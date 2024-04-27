import tiktoken

def is_maximum_context_length_error(error_log: str):
  return 'maximum context length' in error_log


def token_count(string: str) -> int:
  return len(tiktoken.encoding_for_model('gpt-3.5-turbo').encode(string))
