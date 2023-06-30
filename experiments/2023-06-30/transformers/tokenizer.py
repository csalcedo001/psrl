from transformers import OpenAIGPTTokenizer, OpenAIGPTConfig
import torch

tokenizer_config = OpenAIGPTConfig()
tokenizer = OpenAIGPTTokenizer(tokenizer_config)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

print('inputs[input_ids]', inputs['input_ids'])
print('inputs[attention_mask]', inputs['attention_mask'])