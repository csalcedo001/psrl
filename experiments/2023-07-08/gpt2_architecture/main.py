from transformers import GPT2Config, GPT2LMHeadModel

# Get model
vocab_size = 79
seq_len = 64

model_config = GPT2Config()

model_config.vocab_size = vocab_size
model_config.n_positions = seq_len
model_config.n_ctx = seq_len

print(model_config)

model = GPT2LMHeadModel(model_config)
print(model)
