# python tests/test_code_modification.py
import os
import pathlib
import sys

from context.external_context_parsing import compare_versions
sys.path.append("..")

import unittest
from unittest.mock import patch, Mock
from config import ExecutionConfig, TaskConfig, VersionConfig
from code_modification.task_simplification import get_minimal_task_config, get_top_sorted_list_of_package_version_maps_from_task_config

class TestGetSortedPackagesFromTaskConfig(unittest.TestCase):
  @patch('code_modification.task_simplification.changelogs.get')
  def test_get_top_sorted_list_of_package_version_maps_from_task_config(self, mock_changelogs_get):
    mock_changelogs_get.side_effect = lambda package: (
      {
        "1.0.0": "changelog1",
        "2.0.0": "changelog2"
      } if package == "package1" else (
      {
        "1.5.0": "changelog3",
        "2.0.0": "changelog4"
      } if package == "package2" else None
      )
    )

    original_version_config = VersionConfig(
      python_version="3.9.0",
      package_version_map={
        "package1": "1.0.0",
        "package2": "1.5.0"
      }
    )
    upgraded_version_config = VersionConfig(
      python_version="3.9.0",
      package_version_map={
        "package1": "2.0.0",
        "package2": "2.0.0"
      }
    )

    task_config = TaskConfig(original_version_config, upgraded_version_config)
    top_sorted_list_of_package_version_maps = get_top_sorted_list_of_package_version_maps_from_task_config(task_config)
    expected_result = [
      {"package1": "1.0.0", "package2": "1.5.0"},
      {"package1": "1.0.0", "package2": "2.0.0"},
      {"package1": "2.0.0", "package2": "1.5.0"},
      {"package1": "2.0.0", "package2": "2.0.0"}
    ]
    self.assertEqual(list(top_sorted_list_of_package_version_maps), expected_result)



  def test_get_minimal_task_config(self):
     
    original_version_config = VersionConfig(
      python_version='3.9.0',
      package_version_map={
        # 'numpy': '1.11.5',
        'scikit-learn': '1.1.0'
      }
    )

    upgraded_version_config = VersionConfig(
      python_version='3.9.0',
      package_version_map={
        # 'numpy': '1.25.5',
        'scikit-learn': '1.2.0'
      }
    )

    task_config = TaskConfig(
      original_version_config=original_version_config,
      upgraded_version_config=upgraded_version_config
    )

    absolute_directory_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "simple_tool")
    execution_config = ExecutionConfig(
      absolute_directory_path=absolute_directory_path,
      relative_test_path='test.py',
      relative_file_path_list=['model_training.py']
    )
    minimal_task_config_report = get_minimal_task_config(
      task_config=task_config,
      execution_config=execution_config
    ).content
    assert 'unexpected keyword argument' in minimal_task_config_report.upgraded_version_code_execution_result.code_execution_logs
    assert 'unexpected keyword argument' not in minimal_task_config_report.original_version_code_execution_result.code_execution_logs
    assert compare_versions(
      minimal_task_config_report.minimal_task_config.original_version_config.package_version_map['scikit-learn'], '1.1.0') == 1
    assert compare_versions(
      minimal_task_config_report.minimal_task_config.original_version_config.package_version_map['scikit-learn'], '1.2.0') == -1

if __name__ == '__main__':
    unittest.main()