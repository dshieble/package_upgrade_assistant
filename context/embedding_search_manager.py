from typing import List
import torch
from transformers import AutoTokenizer, AutoModel

class EmbeddingSearchManager:

  def __init__(self, model_name: str):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
  
  def get_embedding(self, text: str) -> torch.Tensor:
    tokens = self.tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor([tokens])
    outputs = self.model(input_ids)
    return outputs.last_hidden_state.mean(dim=1)

  def get_value_distances(self, key_text: str, value_text_list: List[str]) -> List[float]:
    """
    Returns the value texts sorted in reverse distance
    """
    key_embedding = self.get_embedding(text=key_text)
    value_embedding_list = [self.get_embedding(text=value_text) for value_text in value_text_list]
    value_distance_list = [torch.dist(key_embedding, value_embedding) for value_embedding in value_embedding_list]
    return value_distance_list


