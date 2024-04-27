"""
Helper methods for accessing and parsing the extra context that GPT needs to fix code. This includes changelogs.
"""
from collections import defaultdict
from dataclasses import dataclass
import logging
from urlextract import URLExtract
import html2text
from bs4 import BeautifulSoup
from typing import Callable, Dict, List, Optional
import asyncio 
import httpx
import changelogs
from langchain.text_splitter import (
  RecursiveCharacterTextSplitter,
  Language
)

from code_execution.code_files import CodeChunkDelta, CodeExecutionResult, ModuleMap, CodeChunk
from config import TaskConfig
from context.embedding_search_manager import EmbeddingSearchManager
from simple_llm.llm_utils import get_response_from_prompt_one_shot
from simple_llm.utilities import get_logger

@dataclass
class MatchedText:
  source: str
  content: str
  affinity: Optional[float] = None

@dataclass
class ProcessedText:
  raw: str
  processed: List[str]

  def get_processed_matches(self, search_fn: Callable[[str], bool]) -> List[str]:
    return [text for text in self.processed if search_fn(text)]

@dataclass
class HydratedProcessedText:

  body: ProcessedText
  url_to_processed_text: Dict[str, ProcessedText]

class TextExtractor:

  def __init__(self, chunk_size: int = 5000, chunk_overlap: int = 200):
    self.extractor = URLExtract()
    self.html_splitter = RecursiveCharacterTextSplitter.from_language(
      language=Language.HTML, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    self.markdown_splitter = RecursiveCharacterTextSplitter.from_language(
      language=Language.MARKDOWN, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

  async def get_processed_text_from_text(self, text: str) -> ProcessedText:
    split = self.markdown_splitter.create_documents([text])
    return ProcessedText(raw=text, processed=split)


  async def get_processed_text_from_url(self, url: str) -> ProcessedText:
    # NOTE: We push processing down to the initial data ingestion point because we want to present both the raw body text and the html text back to the user with the same structure, and the ingestion point knows about the difference between the text and html structures
    try:
      async with httpx.AsyncClient() as client:
        raw_html = (await client.get(url)).text
    except Exception as e:
      print(f"Error getting text from url: {url}")
      raise e
    split_html = self.html_splitter.create_documents([raw_html])
    processed_split_html = [BeautifulSoup(html_body.page_content, features="lxml").get_text() for html_body in split_html]
    return ProcessedText(raw=raw_html, processed=processed_split_html)


  async def _get_url_to_processed_text(self, text_with_urls: str) -> Dict[str, ProcessedText]:
    urls = self.extractor.find_urls(text_with_urls)
    tasks = []
    for url in urls:
      tasks.append(asyncio.create_task(self.get_processed_text_from_url(url=url)))
    results = await asyncio.gather(*tasks)
    return dict(zip(urls, results))

  async def hydrate_text_with_urls(self, text_with_urls: str) -> HydratedProcessedText:
    body = await self.get_processed_text_from_text(text=text_with_urls)
    url_to_processed_text = await self._get_url_to_processed_text(text_with_urls)
    return HydratedProcessedText(
      body=body,
      url_to_processed_text=url_to_processed_text,
    )

@dataclass
class TextContext:

  key_to_processed_text: Dict[str, ProcessedText]

  def get_processed_matches(self, search_fn: Callable[[str], bool]) -> List[MatchedText]:
    # Get all processed chunks of text
    # TODO: Change this to use a ranking function and a top-k based on a tiktoken-computed budget
    matched_texts = []
    content_set = set()
    for key, processed_text in self.key_to_processed_text.items():
      for matched_text_content in processed_text.get_processed_matches(search_fn=search_fn):

        # We want to avoid returning multiple identical chunks of text
        _matched_text_content_key = matched_text_content.lstrip().rstrip().replace('\n', '')
        if _matched_text_content_key not in content_set:
          content_set.add(_matched_text_content_key)
          matched_texts.append(MatchedText(source=key, content=matched_text_content))
    return matched_texts

  @classmethod
  async def from_text_with_urls_dict(cls, text_with_urls_dict: Dict[str, str], **extractor_kwargs) -> "TextContext":
    extractor = TextExtractor(**extractor_kwargs)
    keys = []
    tasks = []
    for key, text_with_urls in text_with_urls_dict.items():
      keys.append(key)
      tasks.append(asyncio.create_task(extractor.hydrate_text_with_urls(text_with_urls=text_with_urls)))
    results = await asyncio.gather(*tasks)
    key_to_hydrated_processed_text = dict(zip(keys, results))

    key_to_processed_text = {}
    for key, hydrated_processed_text in key_to_hydrated_processed_text.items():
      key_to_processed_text[key] = hydrated_processed_text.body
      for url, processed_text in hydrated_processed_text.url_to_processed_text.items():
        key_to_processed_text[url] = processed_text
    return cls(key_to_processed_text=key_to_processed_text)


def compare_versions(v1: str, v2: str) -> int:
  """
  Return 1 if v1 > v2, -1 if v1 < v2, and 0 if v1 == v2
  """
  v1_split = v1.split('.')
  v2_split = v2.split('.')

  # NOTE: Some version numbers are formatted like `scikit-learn_0.22.2.post1`. We consider the absence of a modifier to be equivalent to the lowest version. So when we compare `scikit-learn_0.22.2.post1` to `scikit-learn_0.22.2` we consider `scikit-learn_0.22.2.post1` to be the greater version.
  v1_split += [''] * (max(len(v1_split), len(v2_split)) - len(v1_split))
  v2_split += [''] * (max(len(v1_split), len(v2_split)) - len(v2_split))

  for i in range(len(v1_split)):
    if v1_split[i] > v2_split[i]:
      return 1
    elif v1_split[i] < v2_split[i]:
      return -1
  return 0

def get_included_versions(version_list: list, start_version: str, end_version: str) -> list:
  """
  Return a list of versions that are between start_version and end_version
  """
  included_versions = []
  for version in version_list:
    # We include all versions greater than or equal to the start version and less than or equal to the end version
    if compare_versions(version, start_version) >= 0 and compare_versions(version, end_version) <= 0:
      included_versions.append(version)
  return included_versions

@dataclass
class ExternalContextManager:

  task_config: TaskConfig
  module_map: ModuleMap
  text_context: TextContext
  embedding_search_manager: EmbeddingSearchManager
  
  def get_all_package_imports(self) -> List[str]:
    # Find all methods and classes that are imported from the changed packages in the task_config
    relevant_imports = []
    for package in self.task_config.upgraded_version_config.package_version_map:
      path_to_imports = self.module_map.get_package_imports(package=package)
      relevant_imports += [imp for import_list in path_to_imports.values() for imp in import_list]
    return relevant_imports

  def get_context_list(self, code_execution_result: CodeExecutionResult, sort_matches: bool = True) -> List[str]:
    # Get all imports from a changed package that show up in the error message, and then grab all context slices that contain one of those imports
    all_package_imports = self.get_all_package_imports()
    relevant_imports = [imp for imp in all_package_imports if imp in code_execution_result.code_execution_logs]
    raw_matches = self.text_context.get_processed_matches(search_fn=lambda x: any(imp in x for imp in relevant_imports))
    
    # Find the embedding distance of each chunk to the error message
    if sort_matches:
      # TODO: Investigate GPT-based sorting techniques that try to sort by prompt
      match_distances = self.embedding_search_manager.get_value_distances(
        key_text=code_execution_result.code_execution_logs,
        value_text_list=[m.content for m in raw_matches])
      matches_with_affinity = [
        MatchedText(affinity=-distance, source=m.source, content=m.content) for m, distance in zip(raw_matches, match_distances)]
      
      # Return the lowest distance matches first
      matches = sorted(matches_with_affinity, key=lambda m: -m.affinity)
    else:
      matches = raw_matches
    return matches

  def get_context_string_full(self, code_execution_result: CodeExecutionResult) -> str:
    # Return a string representation of the context
    context_list = self.get_context_list(code_execution_result=code_execution_result)
    source_to_content_list = defaultdict(list)
    for context in context_list:
      source_to_content_list[context.source].append(context.content)

    context_string = ""
    for source, content_list in source_to_content_list.items():
      joined_content_list = "\n...\n".join(content_list)
      context_string += f"* {source}: {joined_content_list}\n\n"
    return context_string
  
  context_summary_prompt = """
You are a top notch software engineering support agent. I see the following error message after updating some package dependencies in my python program:
--- START ERROR MESSAGE ---
{error_message}
--- END ERROR MESSAGE ---

Here are some relevant snippets of text from the changelogs of the packages that I updated
--- START CHANGELOG ---
{changelog_text}
--- END CHANGELOG ---

Based on this error and the information in the changelogs, can you identify why my code is not passing? Please return all relevant information from the changelogs to describe why my code is failing and how to fix it.
"""
  async def get_context_string_summary(
    self,
    code_execution_result: CodeExecutionResult,
    logger: Optional[logging.Logger] = None
  ) -> str:
    # Return a GPT-summarized string representation of the context
    logger = logger if logger is not None else get_logger()

    context_string = self.get_context_string_full(code_execution_result=code_execution_result)
    prompt = self.context_summary_prompt.format(changelog_text=context_string, error_message=code_execution_result.code_execution_logs)
    return (await get_response_from_prompt_one_shot(prompt=prompt, logger=logger)).unwrap()

  @classmethod
  async def from_task_config(cls, task_config: TaskConfig, module_map: ModuleMap) -> "ExternalContextManager":
    """
    Build the ExternalContextManager from a task_config that specifies the packages whose versions are getting updated. The ExternalContextManager fetches information specifically about these packages
    """
    assert task_config.original_version_config.package_version_map.keys() == task_config.upgraded_version_config.package_version_map.keys()

    embedding_search_manager = EmbeddingSearchManager("microsoft/codebert-base-mlm")

    package_x_version_to_logs = {}
    for package in task_config.upgraded_version_config.package_version_map:
      version_to_logs = changelogs.get(package)
      included_versions = get_included_versions(
        version_to_logs.keys(),
        start_version=task_config.original_version_config.package_version_map[package],
        end_version=task_config.upgraded_version_config.package_version_map[package]
      )
      package_x_version_to_logs.update({f"{package}_{version}": version_to_logs[version] for version in included_versions})

    # Each log entry may contain urls that map to raw changelogs, which is what we want to open and parse
    text_context = await TextContext.from_text_with_urls_dict(text_with_urls_dict=package_x_version_to_logs)


    return cls(task_config=task_config, module_map=module_map, text_context=text_context, embedding_search_manager=embedding_search_manager)