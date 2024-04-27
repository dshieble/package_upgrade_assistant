import asyncio
from copy import deepcopy
import logging
import os
import re
import shutil
import uuid
from code_execution.code_files import CodeExecutionResult, ModuleMap
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

from config import DockerConfig, VersionConfig

from docker import DockerClient
import os


def get_docker_client() -> DockerClient:
  try:
    docker_client = DockerClient().from_env()
  except Exception as e:
    print('Docker is not running. Please start docker and try again.')
    raise ValueError()
  else:
    docker_client.login(username=os.environ['DOCKER_USERNAME'], password=os.environ['DOCKER_PASSWORD'])
  return docker_client



def build_image(
  docker_client: DockerClient,
  version_config: VersionConfig,
  path_to_code_dir: str,
  code_dir_name: str,
  tag: Optional[str] = None,
  image_name: str = 'danshiebler/private'
) -> Image:
  tag = tag if tag is not None else str(uuid.uuid4())
  working_dir = f"/tmp/{str(uuid.uuid4())}"
  os.mkdir(working_dir)

  # Add the code that we will be working with
  shutil.copytree(path_to_code_dir, os.path.join(working_dir, code_dir_name))

  # Add helper scripts for image inspection and optimization
  docker_scripts_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "docker_scripts")
  shutil.copytree(docker_scripts_directory, os.path.join(working_dir, "docker_scripts"))

  version_config.write_packages_to_file(os.path.join(working_dir, 'requirements.txt'))

  dockerfile_content = f"""
  FROM python:{version_config.python_version}
  COPY . /app
  WORKDIR /app
  RUN pip install -r requirements.txt
  """

  # create Dockerfile
  with open(f"{working_dir}/Dockerfile", "w") as f:
    f.write(dockerfile_content)

  # build docker image
  image = docker_client.images.build(path=working_dir, tag=f"{image_name}:{tag}")[0]

  # push to Docker hub
  # docker_client.images.push(image_name, tag=tag)
  return image

