import asyncio
from copy import deepcopy
import logging
import os
import re
import traceback
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar
from attrs import define
from dataclasses import dataclass
from io import StringIO 
import sys
from code_execution.docker_helpers import get_docker_client
from code_modification.task_simplification import get_minimal_task_config

import json5
from docker import DockerClient
from context.code_parsing import CodeContext
from simple_llm.llm_utils import DEFAULT_MODEL_NAME, MessageManager
from simple_llm.parsing_utils import add_markdown_formatting, find_json_string, remove_markdown_formatting
from simple_llm.utilities import Maybe, get_logger, to_string_if_not_none

from utilities import is_maximum_context_length_error, token_count
from code_execution.code_files import CodeChunkDelta, CodeExecutionResult, ModuleMap, CodeChunk
from code_execution.execution_engine import FileSwapExecutionEngine

from config import ExecutionConfig, TaskConfig
from context.external_context_parsing import ExternalContextManager

@dataclass
class CodeFixResponse:
  """
  This object stores the information associated with running some code, deriving a prompt from that code's errors, and the LLM's response to that prompt
  """

  # When this is None, the other fields are None as well. This indicates that the code itself could not be found, and therefore the code fix process could not run
  last_code_execution_result: Optional[CodeExecutionResult]

  # This prompt is derived from the last_code_execution_result. It is absent if the last_code_execution_result is None or was successful
  prompt: Optional[str]

  # This is an Optional Maybe since when this is None it indicates that there is no code fix response (since the code was successfully executed) and when this is not None this indicates that we tried to get a response, but that response may have been None
  maybe_code_response: Optional[Maybe[str]]

  status_message: Optional[str] = None

  def display(self):
    print(self.last_code_execution_result.stringify_code_and_result() if self.last_code_execution_result is not None else "last_code_execution_result is None")
    print(f"Prompt [{token_count(self.prompt)} tokens]: {self.prompt}" if self.prompt is not None else "Prompt is None")
    print(self.maybe_code_response.to_string() if self.maybe_code_response is not None else "maybe_code_response is None")
    print(f"Status Message: {self.status_message}" if self.status_message is not None else "Status Message is None")


class AbstractCodeModificationEngine:

  def __init__(self,
    task_config: TaskConfig,
    message_manager: MessageManager,
    execution_engine: FileSwapExecutionEngine,
    external_context_manager: ExternalContextManager,
    num_error_iterations: int,
    use_context_summarization: bool = False,
    logger: Optional[logging.Logger] = None
  ):
    self.task_config = task_config
    self.message_manager = message_manager
    self.execution_engine = execution_engine
    self.external_context_manager = external_context_manager
    self.num_error_iterations = num_error_iterations
    self.use_context_summarization = use_context_summarization
    self.logger = logger or get_logger()


  @classmethod
  async def from_task_config_and_execution_config(
    cls, 
    task_config: TaskConfig,
    execution_config: ExecutionConfig,
    logger: Optional[logging.Logger] = None,
    docker_client: Optional[DockerClient] = None,
    **kwargs
  ) -> "AbstractCodeModificationEngine":
    """
    Args:
      task_config: A description of the migration we are making in terms of packages and python version
      execution_config: A description of the files that we are changing
    """
    docker_client = docker_client if docker_client is not None else get_docker_client()

    execution_engine = FileSwapExecutionEngine(
      version_config=task_config.upgraded_version_config,
      execution_config=execution_config,
      logger=logger,
      docker_client=docker_client
    )
    external_context_manager = await ExternalContextManager.from_task_config(task_config=task_config, module_map=execution_engine.base_module_map)

    # The test code is kept separate from the path_to_code
    assert execution_engine.base_module_map.relative_test_path not in execution_engine.base_module_map.path_to_code 
  
    message_manager = MessageManager(
      model=DEFAULT_MODEL_NAME,
      messages=[{"role": "system", "content": "You are an expert python programmer. Please respond to the following prompt"}],
      logger=(logger or get_logger())
    )
    return cls(
      task_config=task_config,
      message_manager=message_manager,
      execution_engine=execution_engine,
      external_context_manager=external_context_manager,
      logger=logger,
      **kwargs
    )

  
  async def process_code_response(
    self,
    last_prompt: str,
    input_maybe_code_response: Maybe[str],
    **code_kwargs
  ) -> CodeFixResponse:
    """
    Given a raw code response from the LLM, try to interpret it as code. If this fails, re-prompt the LLM to fix the code.
    Args:
      last_prompt: The last prompt that was sent to the LLM
      input_maybe_code_response: The response from the LLM that we are trying to interpret as code
      code_kwargs: The kwargs to pass to the code execution engine
    Returns:
      A CodeFixResponse object that contains the last code execution result, the prompt derived from this result that we sent to the LLM, and the response from the LLM
    """
    if input_maybe_code_response.content is None:
      # The last_prompt didn't succeed due to an LLM error, so we set the prompt to be the last prompt used
  
      # Handle LLM Errors
      if is_maximum_context_length_error(error_log=input_maybe_code_response.error):
        # Break if we hit the maximum context length. In this case we return a failing execution result
        code_fix_response = CodeFixResponse(
          last_code_execution_result=None,
          prompt=None,
          maybe_code_response=None,
          status_message="FAILURE: maximum context length"
        )
      else:
        # For other errors let's just sleep a bit and try again
        await asyncio.sleep(10)
        code_fix_response = CodeFixResponse(
          last_code_execution_result=None,
          prompt=last_prompt,
          maybe_code_response=await self.message_manager.get_response(prompt=prompt)
        )
    else:

      input_maybe_code_response_parsed = input_maybe_code_response.maybe_monad_join(find_json_string)

      if input_maybe_code_response_parsed.content is None:
        # The response was not a valid JSON string that contained code
        prompt = self.get_no_code_found_prompt()
        code_fix_response = CodeFixResponse(
          last_code_execution_result=None,
          prompt=prompt,
          maybe_code_response=await self.message_manager.get_response(prompt=prompt)
        )
      else:

        optional_parsed_response_validation_prompt = self.get_optional_parsed_response_error_prompt(
          parsed_code_response=input_maybe_code_response_parsed.content, **code_kwargs)

        if optional_parsed_response_validation_prompt is not None:
          # The response was a valid JSON string, but it was formatted incorrectly
          prompt = optional_parsed_response_validation_prompt
          code_fix_response = CodeFixResponse(
            last_code_execution_result=None,
            prompt=prompt,
            maybe_code_response=await self.message_manager.get_response(prompt=prompt)
          )

        else:
          code_execution_result = self.run_module_tests_on_parsed_code_response(
            parsed_code_response=input_maybe_code_response_parsed.content, **code_kwargs
          )

          if code_execution_result.code_exited_with_error:
            # Code block was found, we executed it, but it threw an error
            assert code_execution_result.module_map is not None
            assert code_execution_result.code_execution_logs is not None
            prompt = self.get_error_prompt(
              module_map=code_execution_result.module_map,
              code_execution_result=code_execution_result
            )
            code_fix_response = CodeFixResponse(
              last_code_execution_result=code_execution_result,
              prompt=prompt,
              maybe_code_response=await self.message_manager.get_response(prompt=prompt)
            )
          else:
            # Code block was found, we executed it, and it didn't throw an error
            code_fix_response = CodeFixResponse(
              last_code_execution_result=code_execution_result,
              prompt=None,
              maybe_code_response=None,
              status_message="SUCCESS: code executed without error"
            )
    assert isinstance(code_fix_response, CodeFixResponse)
    return code_fix_response


  def get_code_kwargs(self, initial_code_execution_result: CodeExecutionResult) -> Dict[str, Any]:
    return {}


  async def get_code_fix_response_list(
    self,
    logger: Optional[logging.Logger] = None
  ) -> List[CodeFixResponse]:
    """
    Run the code, and then repeatedly ask ChatGPT to fix its code until it works. Returns a list of all CodeExecutionResult objects produced over the course of this function execution. The last CodeExecutionResult will store a module_map that represent's GPT's best attempt at fixing the code (regardless of whether the tests passed) and the logs of that run.
    """
    logger = logger if logger is not None else self.logger
    initial_code_execution_result = self.execution_engine.run_base_tests()

    code_fix_response_list = []
    if not initial_code_execution_result.code_exited_with_error:
      code_fix_response = CodeFixResponse(
        last_code_execution_result=initial_code_execution_result,
        prompt=None,
        maybe_code_response=None
      )
      code_fix_response_list.append(code_fix_response)
    else:
      # The original code produces an error
      logger.info(f"Here is the test that needs to pass:\n{initial_code_execution_result.module_map.stringify_test_code()}\n")
      logger.info(f"Now given this code\n{initial_code_execution_result.module_map.stringify_path_to_code()}\nWe see this error message:\n\n{initial_code_execution_result.display_code_execution_logs()}")

      code_kwargs = self.get_code_kwargs(initial_code_execution_result=initial_code_execution_result)
      prompt = await self.get_initial_prompt(code_execution_result=initial_code_execution_result, **code_kwargs)
      last_code_fix_response = CodeFixResponse(
        last_code_execution_result=initial_code_execution_result,
        prompt=prompt,
        maybe_code_response=await self.message_manager.get_response(prompt=prompt)
      )

      code_fix_response_list.append(last_code_fix_response)
      for _ in range(self.num_error_iterations):
        assert last_code_fix_response.prompt is not None
        assert last_code_fix_response.maybe_code_response is not None
        last_code_fix_response = await self.process_code_response(
          last_prompt=last_code_fix_response.prompt,
          input_maybe_code_response=last_code_fix_response.maybe_code_response,
          **code_kwargs
        )
        assert isinstance(last_code_fix_response, CodeFixResponse)
        code_fix_response_list.append(last_code_fix_response)

        if last_code_fix_response.last_code_execution_result is not None:
          logger.info(f"Now the LLM edits this code to produce the following\n{last_code_fix_response.last_code_execution_result.module_map.stringify_path_to_code()}\nAnd this returns the result:\n{last_code_fix_response.last_code_execution_result.display_code_execution_logs()}")

        if last_code_fix_response.maybe_code_response is None:
          # If the LLM was not prompted to generate a new response, we should break
          break

    return code_fix_response_list

  initial_prompt_template = """ 
The following code was originally written to run with the packages:
{original_packages}
on python {original_python_version}

I want to make it compatible with the upgraded packages:
{upgraded_packages}
on python {upgraded_python_version}

Here are some relevant snippets of text from the changelog of these packages
--- START CHANGELOG ---
{changelog_text}
--- END CHANGELOG ---

Here is a json representation of some snippets of my code:
{base_code_string}

And here is the test I am running:
{test_code_string}

When I run this test with the upgraded packages I see the error:
{original_error}

Please correct this code without changing the test. Format your answer as json in the following format:
{json_response_template_string}

Follow this process:
  Think: Why is this code failing? What changed between the old and new packages?
  Inspect: Look at the information provided in the changelog. Is there a fix suggested?
  Plan: How should you change the code to fix the error?
  Your Answer: <your code in the json format above>

Think:
"""
  async def get_initial_prompt(
    self,
    code_execution_result: CodeExecutionResult,
    **code_kwargs
  ) -> str:

    # This prompt only makes sense is there is an error in the code
    assert code_execution_result.code_exited_with_error

    if self.use_context_summarization:
      context_string = await self.external_context_manager.get_context_string_summary(
        code_execution_result=code_execution_result, logger=self.logger)
    else:
      context_string = self.external_context_manager.get_context_string_full(code_execution_result=code_execution_result)
    return self.initial_prompt_template.format(
      original_python_version=self.task_config.original_version_config.python_version,
      upgraded_python_version=self.task_config.upgraded_version_config.python_version,
      original_packages='\n'.join([f'{k}=={v}' for k,v in self.task_config.original_version_config.package_version_map.items()]),
      upgraded_packages='\n'.join([f'{k}=={v}' for k,v in self.task_config.upgraded_version_config.package_version_map.items()]),
      base_code_string=self.get_base_code_string(**code_kwargs),
      test_code_string=self.get_test_code_string(),
      original_error=add_markdown_formatting(code_execution_result.code_execution_logs),
      changelog_text=context_string,
      json_response_template_string=self.get_json_response_template_string()
    )

  no_code_found_prompt_template = """
I couldn't parse code changes from that message. Did you format your answer as json in the following format:
{json_response_template_string}
"""
  def get_no_code_found_prompt(self) -> str:
    return self.no_code_found_prompt_template.format(
      json_response_template_string=self.get_json_response_template_string()
    )

  # TODO: Add error context if needed
  error_prompt_template = """
I tried running that code you provided, but I got an error.

Here are the logs including the error:
{error}

Please correct this code. Follow this process:
  Reflect: Why is this code failing? Why didn't your previous change work?
  Plan: Think step-by-step about how you can fix this bug. Remember, you cannot fix the test. 
  Your Answer: <your code in the json format above>

Reflect:
"""
  def get_error_prompt(self, module_map: ModuleMap, code_execution_result: CodeExecutionResult) -> str:
    return self.error_prompt_template.format(
      error=add_markdown_formatting(code_execution_result.code_execution_logs)
    )

  def get_optional_parsed_response_error_prompt(self, parsed_code_response: Dict[str, str]) -> Optional[str]:
    raise NotImplementedError()

  def get_base_code_string(self, **code_kwargs) -> str:
    raise NotImplementedError()

  def get_test_code_string(self) -> str:
    raise NotImplementedError()

  def get_json_response_template_string(self) -> str:
    # return json5.dumps({k: f"```<your code for file {k}>```" for k in self.execution_engine.base_module_map.path_to_code.keys()})
    raise NotImplementedError()

  def run_module_tests_on_parsed_code_response(self, parsed_code_response: Dict[str, str]) -> CodeExecutionResult:
    raise NotImplementedError()

class SelectedCodeFileCodeModificationEngine(AbstractCodeModificationEngine):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.code_context = CodeContext.from_module_map(module_map=self.execution_engine.base_module_map)

  def get_code_kwargs(self, initial_code_execution_result: CodeExecutionResult) -> Dict[str, str]:
    # TODO: Search for the relevant code snippets by using the error and then return a dict that contains the data needed
    code_chunk_list = self.code_context.code_chunk_list 
    return {
      "code_chunk_list": code_chunk_list,
    }
  
  def get_optional_parsed_response_error_prompt(
    self,
    parsed_code_response: List[Dict[str, str]],
    code_chunk_list: List[CodeChunk]
  ) -> Optional[str]:
    location_to_original_code_chunk = {code_chunk.get_location_id(): code_chunk for code_chunk in code_chunk_list}

    optional_prompt = None
    if type(parsed_code_response) != list or any(set(r.keys()) != {'location', 'content'} for r in parsed_code_response):
      optional_prompt = self.get_no_code_found_prompt()
    else:
      invalid_code_change_locations = [r['location'] for r in parsed_code_response if r['location']  not in location_to_original_code_chunk]
      if len(invalid_code_change_locations) > 0:
        optional_prompt = f"ERROR: {invalid_code_change_locations} are not valid code change locations. The valid locations are {list(location_to_original_code_chunk.keys())}. Please try again."
    return optional_prompt

  def get_base_code_string(self, code_chunk_list: List[CodeChunk]) -> str:
    base_code_list = [
      {
        'location': code_chunk.get_location_id(),
        'content': code_chunk.get_code_chunk_string()
      } for code_chunk in code_chunk_list
    ]
    return json5.dumps(base_code_list)

  def get_test_code_string(self) -> str:
    # TODO: Change this to only get the test method itself, and not the full test file
    return add_markdown_formatting(self.execution_engine.base_module_map.test_code)

  def get_json_response_template_string(self) -> str:
    json_response_template_list = [
      {
        'location': '<file with line numbers>',
        'content': '<your code>'
      } 
    ]
    return json5.dumps(json_response_template_list)

  def run_module_tests_on_parsed_code_response(
    self,
    parsed_code_response: List[Dict[str, str]],
    code_chunk_list: List[CodeChunk]
  ) -> CodeExecutionResult:

    # We should have already validated that parsed_code_response is a list with the right fields and that suggested_change_dict['location'] is in location_to_original_code_chunk
    assert self.get_optional_parsed_response_error_prompt(parsed_code_response=parsed_code_response, code_chunk_list=code_chunk_list) is None
    location_to_original_code_chunk = {code_chunk.get_location_id(): code_chunk for code_chunk in code_chunk_list}

    code_chunk_delta_list = []
    for suggested_change_dict in parsed_code_response:
      code_chunk_delta_list.append(CodeChunkDelta.from_code_change(
        original_code_chunk=location_to_original_code_chunk[suggested_change_dict['location']],
        new_code=suggested_change_dict['content']
      ))

    module_map = ModuleMap.from_module_map_and_code_chunk_delta_list(
      absolute_directory_path=self.execution_engine.absolute_target_dir,
      base_module_map=self.execution_engine.base_module_map,
      code_chunk_delta_list=code_chunk_delta_list
    )
    return self.execution_engine.run_module_tests(module_map=module_map)


@dataclass
class CodeModificationResult:
  code_fix_response_list: List[CodeFixResponse]
  llm_message_list: List[Dict[str, str]]
  modified_code_dir: str


  def display_code_fix_response_list(self):
    for code_fix_response in self.code_fix_response_list:
      code_fix_response.display()
      print("*********")

  def display_message_list(self):
    for message in self.llm_message_list:
      print(f"---------{message['role']}---------")
      print(message['content'])


async def modify_code(
  task_config: TaskConfig,
  execution_config: ExecutionConfig,
  logger: Optional[logging.Logger] = None,
  **kwargs
) -> CodeModificationResult:
  modification_engine = await SelectedCodeFileCodeModificationEngine.from_task_config_and_execution_config(
    logger=logger,
    task_config=task_config,
    execution_config=execution_config,
    **kwargs
  )


  code_fix_response_list = await modification_engine.get_code_fix_response_list(
    logger=logger
  )
  return CodeModificationResult(
    code_fix_response_list=code_fix_response_list,
    llm_message_list=modification_engine.message_manager.messages,
    modified_code_dir=modification_engine.execution_engine.absolute_target_dir
  )