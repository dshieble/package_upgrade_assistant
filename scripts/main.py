# python scripts/main.py
import os
import sys
from unittest import mock

import asyncio
from docker import DockerClient

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from simple_llm.utilities import get_logger
from config import VersionConfig, TaskConfig, DockerConfig
from code_execution.execution_engine import FileSwapExecutionEngine
from code_modification.code_modification_engine import SelectedCodeFileCodeModificationEngine


async def main():
  docker_client = DockerClient().from_env()

  docker_config = DockerConfig(
    docker_path='code_correction/repositories/simple_tool',
    relative_dockerfile='container_upgraded_post_install/Dockerfile',
    docker_tag='simple_tool:container_upgraded_post_install',
  )

  relative_source_dir = 'simple_tool'
  relative_target_dir = 'tmp'
  relative_test_path = 'test.py'
  relative_file_path_list = ['model_training.py']

  # NOTE: Right now every package here will get a changelog fetched
  original_packages = {
    'scikit-learn': '0.19.0'
  }
  upgraded_packages = {
    'scikit-learn': '1.20.0'
  }

  execution_engine = FileSwapExecutionEngine(
    docker_client=docker_client,
    docker_config=docker_config,
    relative_source_dir=relative_source_dir,
    relative_target_dir=relative_target_dir,
    relative_test_path=relative_test_path,
    relative_file_path_list=relative_file_path_list
  )

  task_config = TaskConfig(
    original_version_config=VersionConfig(
      python_version='3.6.0',
      package_version_map=original_packages
    ),
    upgraded_version_config=VersionConfig(
      python_version='3.9.0',
      package_version_map=upgraded_packages
    )
  )

  coding_helper = SelectedCodeFileCodeModificationEngine(
    logger=mock.Mock()
  )
  code_execution_result_list = await coding_helper.get_code_result(
    task_config=task_config,
    execution_engine=execution_engine,
    logger=get_logger()
  )

if __name__ == '__main__':
  asyncio.run(main())
  