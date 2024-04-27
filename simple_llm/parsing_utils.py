import re
from typing import List, Tuple, Any, Optional, Dict, Union
import json5

from simple_llm.utilities import Maybe


def find_matching_pairs(string: str, ldelim: str, rdelim: str) -> List[Tuple[int, int, Any]]:
  """
  Given a string, returns a list of tuples of the form (opening_index, closing_index, [nested pairs])
  """

  def find_pairs_recursive(substring, start_index):
    pairs = []
    stack = []
    for i, char in enumerate(substring):
      if char == ldelim:
        stack.append(i)
      elif char == rdelim:
        if stack:
          opening_index = stack.pop()
          pairs.append(
            (
              opening_index + start_index,
              i + start_index,
              find_pairs_recursive(
                substring[opening_index + 1 : i], opening_index + start_index + 1
              ),
            )
          )
    return pairs

  return find_pairs_recursive(string, 0)


def find_json_string(string_with_json: str) -> Maybe[Union[List[Dict[str, Any]], Dict[str, Any]]]:
  """
  Find the json string in a body of text
  """
  assert type(string_with_json) == str
  pairs = (
    find_matching_pairs(string_with_json, ldelim="{", rdelim="}") + find_matching_pairs(string_with_json, ldelim="[", rdelim="]")
  )
  maybe_parsed = Maybe(error="No json string found")
  for p in reversed(pairs):
    selected_string = string_with_json[p[0] : p[1] + 1]
    try:
      maybe_parsed = Maybe(content=json5.loads(selected_string))
    except Exception as e:
      pass
    if maybe_parsed.content is not None:
      break
  return maybe_parsed


def remove_markdown_formatting(code_with_markdown_formatting: str) -> str:
  no_backticks = (
    code_with_markdown_formatting.split('```')[1] if '```' in code_with_markdown_formatting else code_with_markdown_formatting
  )
  return re.sub(r'^\s*python', '', no_backticks, flags=re.IGNORECASE)

def add_markdown_formatting(code_without_markdown_formatting: str) -> str:
  # We first remove markdown formatting to prevent a double application. Note that this means that this method will strip `python` from the beginning of the code if it is present.
  code_without_markdown_formatting = remove_markdown_formatting(code_without_markdown_formatting)

  # We don't add `python` because this is generic formatting that might be used to wrap an error as well. 
  return f"```\n{code_without_markdown_formatting}\n```"
