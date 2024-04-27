import asyncio
from copy import deepcopy
import functools
import itertools
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

import json5
from context.code_parsing import CodeContext
from simple_llm.llm_utils import DEFAULT_MODEL_NAME, MessageManager
from simple_llm.parsing_utils import add_markdown_formatting, find_json_string, remove_markdown_formatting
from simple_llm.utilities import Maybe, get_logger, to_string_if_not_none

from utilities import is_maximum_context_length_error, token_count
from code_execution.code_files import CodeChunkDelta, CodeExecutionResult, ModuleMap, CodeChunk
from code_execution.execution_engine import run_module_map_in_docker

from config import ExecutionConfig, TaskConfig, VersionConfig
from context.external_context_parsing import ExternalContextManager, compare_versions
from docker import DockerClient
import changelogs



def get_top_sorted_list_of_package_version_maps_from_task_config(task_config: TaskConfig) -> List[Dict[str, str]]:
  """
  TODO: Write unittests for this method


  Given a TaskConfig, sort the packages by version a return a list of
    [
      {p1: v1a, p2: v2a, ...},
      {p1: v1b, p2: v2b, ...},
      {p1: v1c, p2: v2c, ...},
    ] 
  where
    original version <= (v1a, v2a, ...) < (v1b, v2b, ...) < (v1c, v2c, ...) <= upgraded versions
  and 
    (p1, p2, ...) is the full set of all packages that in both the original and upgraded package configs
  """
  package_to_sorted_version_list = {
    package: sorted(list(changelogs.get(package).keys()), key=functools.cmp_to_key(compare_versions))
    
    for package in task_config.original_version_config.package_version_map.keys()
  }
  list_of_package_version_tuple_lists = [
    [
      (package, version) for version in package_to_sorted_version_list[package]
      if compare_versions(version, task_config.original_version_config.package_version_map[package]) >= 0
        and compare_versions(version, task_config.upgraded_version_config.package_version_map[package]) <= 0
    ]
    for package in package_to_sorted_version_list
  ]

  # NOTE: These are topologically sorted, but they are not linearly sorted
  top_sorted_list_of_package_version_tuples_lists = itertools.product(*list_of_package_version_tuple_lists)
  return [{p: v for p, v in package_version_tuples_list} for package_version_tuples_list in top_sorted_list_of_package_version_tuples_lists]


@dataclass
class MinimalTaskConfigReport:
  minimal_task_config: TaskConfig

  # This is the code execution result associated with the upgraded_version_config on the minimal_task_config
  upgraded_version_code_execution_result: CodeExecutionResult

  # This is the code execution result associated with the original_version_config on the minimal_task_config
  original_version_code_execution_result: CodeExecutionResult

def get_minimal_task_config(
  task_config: TaskConfig,
  execution_config: ExecutionConfig,
  docker_client: Optional[DockerClient] = None,
  logger: Optional[logging.Logger] = None
) -> Maybe[MinimalTaskConfigReport]:
  """
  NOTE: One major risk with this method is that when an API change is made, the changelog entry with the change may not lie between the minimal passing version and the new version, since the changelog entry may have been logged at initial deprecation rather than at change time. 


  Given as TaskConfig that specifies an original set of versions and an upgraded set of versions, create a new task_config with the same upgraded package versions, but the original package versions increased to the maximum version configuration that has a different error from the upgraded configuration

  Basically, the version configs are
    old original_version_config -> code execution error A
    new original_version_config -> code execution error B
    <all versions in this space> -> code execution error C
    upgraded_version_config -> code execution error C

  Note that old original_version_config does not need to be a version configuration in which the pipeline succeeds. It is possible that other dependencies has shifted in ways that this version config is no longer viable.
  """
  if task_config.original_version_config.python_version != task_config.upgraded_version_config.python_version:
    raise ValueError("We can only find the minimal task config with a fixed python version!")
  elif set(task_config.original_version_config.package_version_map) != set(task_config.upgraded_version_config.package_version_map):
    raise ValueError("The set of packages in the original and upgraded version maps must be identical")

  logger = logger if logger is not None else get_logger()
  docker_client = docker_client if docker_client is not None else get_docker_client()
  module_map = ModuleMap.from_execution_config(execution_config=execution_config)

  max_version_code_execution_result = run_module_map_in_docker(
    version_config=task_config.upgraded_version_config,
    docker_client=docker_client,
    module_map=module_map
  )

  # Run a topological sort of packages, and find the last package version that fails
  top_sorted_list_of_package_version_maps = get_top_sorted_list_of_package_version_maps_from_task_config(task_config=task_config)
  for package_version_map in reversed(top_sorted_list_of_package_version_maps):
    version_config = VersionConfig(
      python_version=task_config.upgraded_version_config.python_version,
      package_version_map=package_version_map
    )
    logger.info(f"Testing {package_version_map}...")
    code_execution_result = run_module_map_in_docker(
      version_config=version_config,
      docker_client=docker_client,
      module_map=module_map
    )
    if max_version_code_execution_result.code_execution_logs != code_execution_result.code_execution_logs:
      maybe_minimal_task_config_report = Maybe(content=MinimalTaskConfigReport(
        minimal_task_config=TaskConfig(original_version_config=version_config, upgraded_version_config=task_config.upgraded_version_config),
        upgraded_version_code_execution_result=max_version_code_execution_result,
        original_version_code_execution_result=code_execution_result
      ))
      break
  else:
    # NOTE: This should never happen, since if the original version config fails we should catch it above
    maybe_minimal_task_config_report = Maybe(error="No passing code was found")
  return maybe_minimal_task_config_report
