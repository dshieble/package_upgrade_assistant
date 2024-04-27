import asyncio
from copy import deepcopy
import logging
import os
import re
import json5
import traceback
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar
from dataclasses import dataclass
from io import StringIO 
import sys
from code_parsing.parsing_helpers import get_imports, get_modules_from_package
from simple_llm.llm_utils import MessageManager
from simple_llm.utilities import Maybe, get_logger, to_string_if_not_none
from docker import DockerClient

from config import DockerConfig, ExecutionConfig


@dataclass
class CodeChunk:
  file_path: str
  file_code: str
  start_index: int
  end_index: int

  def get_location_id(self) -> str:
    return f'{self.file_path} [{self.start_index}:{self.end_index}]'

  def get_code_chunk_string(self) -> str:
    return self.file_code[self.start_index:self.end_index]

  @classmethod
  def from_code_chunk_string(cls, file_path: str, file_code: str, code_chunk_string: str) -> "CodeChunk":
    start_index = file_code.find(code_chunk_string)
    end_index = start_index + len(code_chunk_string)
    assert file_code[start_index:end_index] == code_chunk_string
    return cls(
      file_path=file_path,
      file_code=file_code,
      start_index=start_index,
      end_index=end_index
    )


@dataclass
class CodeChunkDelta:
  # Represents a code chunk that has been edited from an original code chunk, perhaps due to GPT selecting that chunk as one to apply an edit to
  original_file_code: str
  new_code_chunk: CodeChunk

  @classmethod
  def from_code_change(cls, original_code_chunk: CodeChunk, new_code: str) -> "CodeChunkDelta":
    new_code_chunk = deepcopy(original_code_chunk)
    new_code_chunk.file_code = new_code
    return cls(
      original_file_code=original_code_chunk.file_code,
      new_code_chunk=new_code_chunk
    )

@dataclass
class ModuleMap:
  absolute_directory_path: str
  relative_test_path: str
  test_code: str
  path_to_code: Dict[str, str]

  def stringify_test_code(self) -> str:
    return f'```\n{self.test_code}\n```'

  def stringify_path_to_code(self) -> str:
    out = ''
    for path, code in self.path_to_code.items():
      out += f'```\n# {path}\n{code}```'
    return out

  def get_package_imports(self, package: str) -> Dict[str, str]:
    # Given a python package, find all classes and methods imported from that package in any of the modules in this module map
    # TODO: Potentially also add in some kind of superclassing logic to fetch all upstream classes of these imports
    included_modules = get_modules_from_package(package)
    if len(included_modules) == 0:
      raise ValueError(f'No modules found for package {package}')

    path_to_imports = {}
    for path, code in self.path_to_code.items():
      imports = get_imports(code)
      split_imports = [imp.split(".") for imp in imports]
      path_to_imports[path] = [imp_list[-1] for imp_list in split_imports if imp_list[0] in included_modules]
    return path_to_imports


  @classmethod
  def from_module_map_and_code_chunk_delta_list(
    cls,
    absolute_directory_path: str,
    base_module_map: str,
    code_chunk_delta_list: List[CodeChunkDelta]
  ) -> 'ModuleMap':
    # TODO: Write tests for this
    """
    Args:
      module_map: The module map that represents the code
      code_chunk_delta_list: A list of code chunk deltas that represent the edits to the code
      absolute_directory_path: The path to the directory that the new module map will mirror. This cannot be the same as the directory of the base module map
    Returns:
      A module map that represents applying the GPT edits to the base_module_map
    """
    if absolute_directory_path == base_module_map.absolute_directory_path:
      raise ValueError(f'Absolute directory path cannot be the same as the base module map directory path. {absolute_directory_path} == {base_module_map.absolute_directory_path}')

    module_map = deepcopy(base_module_map)
    module_map.absolute_directory_path = absolute_directory_path

    assert all(code_chunk_delta.new_code_chunk.file_path in module_map.path_to_code for code_chunk_delta in code_chunk_delta_list)

    # NOTE: We might edit the same file multiple times, so we need to be smart about how we do this to avoid shifting the indices
    for file_path, original_code in module_map.path_to_code.items():
      file_deltas = [
        code_chunk_delta for code_chunk_delta in code_chunk_delta_list
        if code_chunk_delta.new_code_chunk.file_path == file_path]
      sorted_file_deltas = sorted(
        file_deltas, key=lambda code_chunk_delta: code_chunk_delta.new_code_chunk.start_index, reverse=True)
      
      for i in range(len(sorted_file_deltas) - 1):
        assert sorted_file_deltas[i].new_code_chunk.end_index <= sorted_file_deltas[i + 1].new_code_chunk.start_index

      code_index_pointer = 0
      edited_file_string = ""
      for code_chunk_delta in sorted_file_deltas:
        # Assert no overlaps
        assert code_index_pointer <= code_chunk_delta.new_code_chunk.start_index

        # Add the code between the last edit and this edit from the original code
        edited_file_string += original_code[code_index_pointer:code_chunk_delta.new_code_chunk.start_index]

        # Add the edited code
        edited_file_string += code_chunk_delta.new_code_chunk.file_code

        # Move the pointer
        code_index_pointer = code_chunk_delta.new_code_chunk.end_index

      # Add the back of the code
      edited_file_string += original_code[code_index_pointer:]
      module_map.path_to_code[code_chunk_delta.new_code_chunk.file_path] = edited_file_string
    return module_map
  
  @classmethod
  def from_module_map_and_module_to_code_dict(cls, base_module_map: str, module_to_code_dict: Dict[str, str]) -> 'ModuleMap':
    """
    Args:
      module_map: The module map that represents the code
      module_to_code: A dictionary mapping module names to the new code changes to those modules
    Returns:
      A module map that represents applying the GPT edits to the base_module_map
    """
    module_map = deepcopy(base_module_map)

    assert all(key in module_map.path_to_code  for key in module_to_code_dict)

    # Overwrite the module code (this will not incldue changes to the test code)
    for relative_file_path in module_map.path_to_code.keys():
      module_map.path_to_code[relative_file_path] = module_to_code_dict[relative_file_path]
    return module_map

  @classmethod
  def from_execution_config(cls, execution_config: ExecutionConfig) -> 'ModuleMap':

    # Read test code
    absolute_test_file_path = f'{execution_config.absolute_directory_path}/{execution_config.relative_test_path}'
    if not os.path.isfile(absolute_test_file_path):
      raise ValueError(f'File does not exist. {absolute_test_file_path}')
    with open(absolute_test_file_path, 'r') as file:
      test_code = file.read()

    # Read module code
    path_to_code = {}
    for relative_file_path in execution_config.relative_file_path_list:
      absolute_file_path = f'{execution_config.absolute_directory_path}/{relative_file_path}'
      if absolute_file_path == absolute_test_file_path:
        raise ValueError(f'Cannot include test file in module map. {absolute_file_path} == {absolute_test_file_path}')
      if not os.path.isfile(absolute_file_path):
        raise ValueError(f'File does not exist. {absolute_file_path}')
      with open(absolute_file_path, 'r') as file:
        path_to_code[relative_file_path] = file.read()
    return cls(
      absolute_directory_path=execution_config.absolute_directory_path,
      relative_test_path=execution_config.relative_test_path,
      test_code=test_code,
      path_to_code=path_to_code
    )



@dataclass
class CodeExecutionResult:
  module_map: ModuleMap
  code_exited_with_error: Optional[bool] = None
  code_execution_logs: Optional[str] = None

  def stringify_code_and_result(self) -> str:
    return f"""---Code---\n{self.module_map.stringify_path_to_code()}\n---Result---\n{self.code_execution_logs}\n---End---"""

  def display_code_and_result(self):
    print(self.stringify_code_and_result())

  def display_code_execution_logs(self):
    print(f'```\n{self.code_execution_logs}\n```')
  
  def display_result(self):
    print(f'Error: {self.code_exited_with_error}\n------\nLogs:```\n{self.code_execution_logs}```')

