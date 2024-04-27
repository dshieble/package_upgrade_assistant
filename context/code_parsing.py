"""
Helper methods for accessing and parsing the extra context that GPT needs to fix code. This includes changelogs.
"""
from dataclasses import dataclass
from urlextract import URLExtract
from typing import Callable, Dict, List
from langchain.text_splitter import (
  RecursiveCharacterTextSplitter,
  Language
)

from code_execution.code_files import CodeChunkDelta, ModuleMap, CodeChunk

@dataclass
class CodeContext:

  code_chunk_list: List[CodeChunk]

  def get_matched_code_chunks(self, k: int, ranking_affinity_fn: Callable[[str], bool]) -> List[str]:
    # Get all code chunks that are matched by the search_fn
    # TODO: Switch to ranking and return top k
    # TODO: Change the fetch algorithm to remove overlaps at fetch time but allow chunks to have overlaps in the index set

    # try to use a token finder
    return [code_chunk for code_chunk in self.code_chunk_list if ranking_affinity_fn(code_chunk.code_chunk)]
  
  @classmethod
  def from_module_map(cls, module_map: ModuleMap, chunk_size: int = 5000, chunk_overlap: int = 0) -> "CodeContext":
    # Create a CodeContext object from the `path_to_code` field of a ModuleMap object
    python_splitter = RecursiveCharacterTextSplitter.from_language(
      language=Language.PYTHON, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    code_chunk_list = []
    for file_path, file_code in module_map.path_to_code.items():
      code_chunk_list += [
        CodeChunk.from_code_chunk_string(
          file_path=file_path,
          file_code=file_code,
          code_chunk_string=document.page_content
        ) for document in python_splitter.create_documents([file_code])
      ]
    
    assert all(isinstance(code_chunk, CodeChunk) for code_chunk in code_chunk_list)
    return cls(code_chunk_list=code_chunk_list)