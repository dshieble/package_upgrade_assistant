import ast
from typing import List, Set

import pkg_resources

class ImportVisitor(ast.NodeVisitor):
  def __init__(self):
    self.imports = []

  def visit_Import(self, node: ast.Import) -> None:
    for alias in node.names:
      self.imports.append(alias.name)

  def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
    for alias in node.names:
      if node.module:
        full_name = f"{node.module}.{alias.name}"
        self.imports.append(full_name)

def get_imports(code: str) -> List[str]:
  tree = ast.parse(code)
  visitor = ImportVisitor()
  visitor.visit(tree)
  return visitor.imports


def get_modules_from_package(package_name: str) -> Set[str]:
  return set(pkg_resources.get_distribution(package_name)._get_metadata('top_level.txt'))
