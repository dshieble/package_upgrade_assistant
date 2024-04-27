# python tests/test_context.py
import sys
sys.path.append("..")

import unittest
from unittest.mock import patch, Mock
from context.external_context_parsing import compare_versions

class TestCompareVersions(unittest.TestCase):
  def test_compare_versions_equal(self):
    self.assertEqual(compare_versions("1.2.3", "1.2.3"), 0)

  def test_compare_versions_greater(self):
    self.assertEqual(compare_versions("2.0.0", "1.0.0"), 1)

  def test_compare_versions_less(self):
    self.assertEqual(compare_versions("1.0.0", "2.0.0"), -1)

  def test_compare_versions_equal_with_modifier(self):
    self.assertEqual(compare_versions("1.2.3.post1", "1.2.3"), 1)

  def test_compare_versions_greater_with_modifier(self):
    self.assertEqual(compare_versions("1.2.3.post2", "1.2.3.post1"), 1)
  
if __name__ == '__main__':
    unittest.main()