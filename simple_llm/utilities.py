import asyncio
import logging
import os
import re
from copy import deepcopy

import git
import json5
import traceback
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Tuple, TypeVar
from dataclasses import dataclass
from lxml import etree
from io import StringIO 
import sys


T = TypeVar('T')
@dataclass
class Maybe(Generic[T]):
  content: Optional[Any] = None
  error: Optional[str] = None

  def unwrap(self):
    if self.content is None:
      raise ValueError(f"Cannot unwrap a Maybe that has no content. Error: {self.error}")
    return self.content

  def to_string(self):
    if self.content is not None:
      return f"Content: {self.content}"
    if self.error is not None:
      return f"Error: {self.error}"
  
  def maybe_apply(self, fn: Callable[[Any, Any], Any], **kwargs):
    out = deepcopy(self)
    if out.content is not None:
      out.content = fn(out.content, **kwargs)
    return out

  def maybe_monad_join(self, fn: "Callable[[Any, Any], Maybe[Any]]", **kwargs):
    # This is a monadic join
    out = deepcopy(self)
    if out.content is not None:
      result = fn(out.content, **kwargs)
      out.content = result.content
      out.error = result.error
    return out

  async def async_maybe_monad_join(self, fn: "Callable[[Any, Any], Maybe[Any]]", **kwargs):
    # This is a monadic join
    return self.maybe_monad_join(fn=fn, **kwargs)


def get_logger() -> logging.Logger:
  logger = logging.getLogger("")
  logger.setLevel(logging.INFO)

  handler = logging.StreamHandler()
  handler.setLevel(logging.INFO)

  # create formatter
  formatter = logging.Formatter('[LOGLINE] %(name)s - %(asctime)s - %(message)s')

  # add formatter to ch
  handler.setFormatter(formatter)

  # add handler to logger
  logger.addHandler(handler)
  
  return logger


def convert_object_to_dict(obj: Any) -> Dict[str, Any]:
  return json5.loads(
    json5.dumps(obj, default=lambda o: getattr(o, '__dict__', str(o)))
  )


def read_files_in_directory(directory_path: str) -> Dict[str, str]:
  """
  Read all the files in a directory and return a dictionary mapping file names to file contents.
  """
  file_contents = {}

  # Get the list of files in the directory
  file_names = os.listdir(directory_path)

  # Iterate over each file in the directory
  for file_name in file_names:
    file_path = os.path.join(directory_path, file_name)

    # Check if the file is a regular file
    if os.path.isfile(file_path):
      # Read the contents of the file
      with open(file_path, 'r') as file:
        contents = file.read()
        file_contents[file_name] = contents

  return file_contents



def to_string_if_not_none(value: Any) -> Optional[str]:
  return str(value) if value is not None else None


def bool_to_string(bool_value: str) -> Optional[bool]:
  # Converts a boolean to a string
  return str(bool_value).lower()


def string_to_bool(string: Optional[str]) -> Optional[bool]:
  # Parses a string that could be "True" or "False" with any capitalization or extra delimiter characters into a boolean
  if string is None:
    out = None
  else:
    string = str(string).lower().strip().replace("'", "").replace('"', "").replace("`","")
    if string == "true":
      out = True
    elif string == "false":
      out = False
    else:
      out = None
  return out


def update_optional_dict(dict_to_update: Optional[Dict[str, Any]], dict_to_update_with: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
  # Updates dict_to_update with dict_to_update_with, if dict_to_update_with is not None
  if dict_to_update is None:
    dict_to_update = {}
  if dict_to_update_with is None:
    dict_to_update_with = {}
  assert type(dict_to_update) == dict
  assert type(dict_to_update_with) == dict
  dict_to_update.update(dict_to_update_with)
  return dict_to_update


def load_json_safe(maybe_json_string: str) -> Optional[Dict[str, Any]]:
  try:
    json_dict = json5.loads(maybe_json_string)
  except Exception as e:
    json_dict = None
  return json_dict


def zip_with_exception(l1: List[Any], l2: List[Any]) -> List[Tuple[Any, Any]]:
  if len(l1) != len(l2):
    raise ValueError(f"Lists must be the same length to zip. Lengths: {len(l1)}, {len(l2)}")
  return list(zip(l1, l2))


async def chunked_gather(awaitable_list: List[Awaitable], chunk_size: int) -> List[Any]:
  """
  Given a list of awaitables, gather them in chunks of 1000 to avoid overloading the event loop.
  """
  out = []
  for i in range(0, len(awaitable_list), chunk_size):
    out += await asyncio.gather(*awaitable_list[i:i+chunk_size])
  return out

