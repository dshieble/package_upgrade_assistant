"""
Helper methods for accessing and parsing the extra context that GPT needs to fix code. This includes changelogs.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class DockerConfig:
  docker_path: str
  relative_dockerfile: str
  docker_tag: str

@dataclass
class VersionConfig:
  python_version: str
  package_version_map: Dict[str, str]

  def write_packages_to_file(self, filepath: str) -> str:
    requirements_file_string = "\n".join([f'{k}=={v}' for k, v in self.package_version_map.items()]) + "\n"
    with open(filepath, "w") as f:
      f.write(requirements_file_string)

@dataclass
class ExecutionConfig:
  # The absolute path to the directory that contains the code
  absolute_directory_path: str

  # The path to the test file, relative to the directory
  relative_test_path: str

  # The path to all other files we want to include in the module map, relative to the directory.
  relative_file_path_list: List[str]

@dataclass
class TaskConfig:
  original_version_config: VersionConfig
  upgraded_version_config: VersionConfig
