from dataclasses import dataclass
import json
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import openai
from simple_llm.utilities import Maybe, get_logger


DEFAULT_MODEL_NAME = "gpt-3.5-turbo-0613"
# DEFAULT_MODEL_NAME = "gpt-4"


@dataclass
class LLMResponseWithHistory:
  """A response from LLM with the prompt and response history."""
  history: List[str]
  response: str


class MessageManager:
  # A simple chat interface directly with openai
  def __init__(self, messages: List[Dict[str, str]], logger: Optional[logging.Logger] = None, model: str = DEFAULT_MODEL_NAME):
    assert os.environ["OPENAI_API_KEY"] is not None
    openai.api_key = os.environ["OPENAI_API_KEY"]
    self.logger = logger or get_logger()
    self.messages = messages
    self.model = model

  def display_messages(self):
    for m in self.messages:
      print(f"{m['role']}\n{m['content']}")
      print('****************************************************************')


  async def get_response(self, prompt: str, temperature: float = 0, **kwargs) -> Maybe[str]:
    self.messages.append({"role": "user", "content": prompt})
    self.logger.info(f"\n\n=====PROMPT====\n\n{prompt}")
    try:
      raw_response = await openai.ChatCompletion.acreate(
        model=self.model,
        messages=self.messages,
        temperature=temperature,
        **kwargs
      )
    except Exception as e:
      error = traceback.format_exc()
      maybe_response = Maybe(content=None, error=error)
      self.logger.info(error)
    else:
      if raw_response['choices'][0]['message']['content'] is not None:
        content = raw_response['choices'][0]['message']['content']
        self.messages.append({"role": "assistant", "content": content})
        self.logger.info(f"\n\n=====RESPONSE====\n\n{content}")
        maybe_response = Maybe(content=content, error=None)
      elif raw_response['choices'][0]['message']['function_call'] is not None:
        content = raw_response['choices'][0]['message']['function_call'].arguments
        self.messages.append({"role": "assistant", "content": content})
        self.logger.info(f"\n\n=====RESPONSE====\n\n{content}")
        maybe_response = Maybe(content=content, error=None)
      else:
        maybe_response = Maybe(content=None, error="No response from LLM")

    return maybe_response



async def get_response_from_prompt_one_shot(
  prompt: str,
  temperature: float = 0,
  model_name: str = DEFAULT_MODEL_NAME,
  logger: Optional[logging.Logger] = None,
  system_prompt: Optional[str] = None,
  **kwargs
) -> Maybe[str]:
  logger = logger or get_logger()
  try:
    # NOTE: Within a StdoutCatcher, all stdout is captured and stored in output. We should not do any logging
    messages = []
    if system_prompt is not None:
      messages.append({"role": "system", "content": system_prompt})
    message_manager = MessageManager(
      model=model_name,
      messages=messages,
      logger=logger)

    maybe_response = await message_manager.get_response(
      prompt=prompt,
      temperature=temperature,
      **kwargs
    )
  
  except Exception as e:
    maybe_response = Maybe(content=None, error=f"START EXCEPTION\n-----\n{traceback.format_exc()}\n-----\nEND EXCEPTION")
  return maybe_response



