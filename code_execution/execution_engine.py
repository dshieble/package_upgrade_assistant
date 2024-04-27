import asyncio
from copy import deepcopy
import logging
import os
import re
import shutil
import uuid
from code_execution.code_files import CodeExecutionResult, ModuleMap
from code_execution.docker_helpers import build_image, get_docker_client
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
from docker.models.images import Image

from config import DockerConfig, ExecutionConfig, VersionConfig


def get_code_result(docker_client: DockerClient, image: Image, command: str) -> Tuple[bool, str]:
  container = docker_client.containers.run(image.tags[0], command, detach=True)
  wait_result = container.wait()
  code_exited_with_error = wait_result['StatusCode'] != 0
  code_execution_logs = container.logs().decode('utf-8')
  return code_exited_with_error, code_execution_logs


def run_module_map_in_docker(
  version_config: VersionConfig,
  module_map: ModuleMap,
  docker_client: Optional[DockerClient] = None
) -> CodeExecutionResult:
  # Write a module_map
  # Write the code to the file
  docker_client = docker_client if docker_client is not None else get_docker_client()
  exec_absolute_file_path = f'{module_map.absolute_directory_path}/{module_map.relative_test_path}'
  assert os.path.isfile(exec_absolute_file_path)

  # exec_relative_file_path = f'{relative_dir}/{module_map.relative_test_path}'
  # exec_absolute_file_path = f'{docker_config.docker_path}/{exec_relative_file_path}'
  
  code_dir_name = "code"
  image = build_image(
    docker_client=docker_client,
    version_config=version_config,
    path_to_code_dir=module_map.absolute_directory_path,
    code_dir_name=code_dir_name
  )

  code_exited_with_error, code_execution_logs = get_code_result(docker_client=docker_client, image=image, command=f'python {code_dir_name}/{module_map.relative_test_path}')
  return CodeExecutionResult(
    module_map=module_map,
    code_exited_with_error=code_exited_with_error,
    code_execution_logs=code_execution_logs
  )


def run_file_in_docker(
  run_module_map_in_docker: VersionConfig,
  docker_client: DockerClient,
  relative_test_path: str
) -> CodeExecutionResult:
  module_map = ModuleMap(relative_test_path=relative_test_path)
  return run_module_map_in_docker(version_config=run_module_map_in_docker, docker_client=docker_client, module_map=module_map)




class FileSwapExecutionEngine:

  def __init__(
    self,
    version_config: VersionConfig,
    execution_config: ExecutionConfig,
    absolute_target_dir: Optional[str] = None,
    docker_client: Optional[DockerClient] = None,
    logger: Optional[logging.Logger] = None
  ):
    """

    Args:
      version_config: The version of python and python packages that will be used to run the code
      absolute_source_dir: The absolute path to the directory that contains the code that will be run
      absolute_target_dir: The absolute path to the directory that we will copy the source code with modifications to
      relative_test_path: The relative path to the test file, defined from the source directory
      relative_file_path_list: The relative paths to the files that will be modified
      docker_client: The docker client that will be used to run the docker container
      logger: The logger that will be used to log messages
    """
    self.logger = logger if logger is None else get_logger()
    self.docker_client = docker_client if docker_client is not None else get_docker_client()

    self.version_config = version_config
    self.absolute_source_dir = execution_config.absolute_directory_path
    self.absolute_target_dir = absolute_target_dir if absolute_target_dir is not None else f"/tmp/{uuid.uuid4()}"

    self.relative_test_path = execution_config.relative_test_path
    self.relative_file_path_list = execution_config.relative_file_path_list

    self.base_module_map = ModuleMap.from_execution_config(
      execution_config=execution_config
    )

  def overwrite_target_file_with_code(self, relative_file_path: str, code: str):
    # This writes to target dir, not source dir
    exec_absolute_file_path = f'{self.absolute_target_dir}/{relative_file_path}'
    assert os.path.isfile(exec_absolute_file_path)

    # Write the code to the file
    with open(exec_absolute_file_path, 'w') as file:
      file.write(code)


  def run_base_tests(self) -> CodeExecutionResult:
    return self.run_module_tests(module_map=self.base_module_map)

  def run_module_tests(self, module_map: ModuleMap) -> CodeExecutionResult:
    """
    For each file in `module_map.path_to_code`, overwrite the content of that file in the target_dir with the code stored on the module_map and run the test file. Note that many files in target_dir, including the test file itself, might not be overwritten
    """

    # Copy the clean repo to the target directory
    os.system(f'rm -rf {self.absolute_target_dir}')
    os.system(f'cp -r {self.absolute_source_dir} {self.absolute_target_dir}')

    # Write the test file
    for relative_file_path, code in module_map.path_to_code.items():
      self.overwrite_target_file_with_code(relative_file_path=relative_file_path, code=code)


    # TODO: Fix bug where the module map is not getting overridden for some reason
    return run_module_map_in_docker(
      version_config=self.version_config,
      docker_client=self.docker_client,
      module_map=module_map
    )




