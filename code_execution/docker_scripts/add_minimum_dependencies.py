from typing import List
import pkg_resources
import argparse

def get_dependencies(package_name: str) -> List[str]:
  package = pkg_resources.working_set.by_key[package_name]

  dependencies = []

  for requirement in package.requires():
    dependency = requirement.project_name
    if requirement.extras:
      dependency += '[' + ','.join(requirement.extras) + ']'
    for comparator, version in requirement.specs:
      dependency += '==' + version
      dependencies.append(dependency)

  return dependencies

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_requirements_file_path", type=str)
  parser.add_argument("--output_requirements_file_path", type=str)
  args = parser.parse_args()

  with open(args.input_requirements_file_path, 'r') as file:
    requirements = file.readlines()
  # Remove comments and empty lines
  requirements = [line.strip() for line in requirements if line and not line.startswith('#')]
  
  # Extract package names
  package_names = [line.split('==')[0].strip().split('<=')[0].strip().split('>=')[0].strip() for line in requirements]

  # Get minimum version dependencies for each package
  dependencies = []
  for package_name in package_names:
    dependencies += get_dependencies(package_name)

  # Write these dependencies to the output file
  with open(args.output_requirements_file_path, 'w') as file:
    file.writelines('\n'.join(requirements + dependencies))
