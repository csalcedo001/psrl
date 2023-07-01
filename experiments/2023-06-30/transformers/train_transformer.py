import os
import torch
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel

from psrl.config import get_env_config
from psrl.utils import env_name_map

from utils import load_experiment_config




# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("*** CURRENT DEVICE:", device)



# Get experiment configuration
config_path = os.path.join(os.path.dirname(__file__), 'exp_config.yaml')
exp_config = load_experiment_config(config_path)



# Get environment
env_class = env_name_map[exp_config.env]
env_config = get_env_config(exp_config.env)
env_config['gamma'] = exp_config.gamma
env = env_class(env_config)



# Get data (so far toy data)
vocab_size = env.observation_space.n + env.action_space.n
batch_size = 2
seq_length = 10

trajectories = torch.LongTensor(np.random.randint(0, vocab_size, (batch_size, seq_length))).to(device)

attention_mask = torch.ones_like(trajectories)
attention_mask[:, -1] = 0



# Get model
model_config = GPT2Config()

model_config.vocab_size = vocab_size
model_config.n_positions = seq_length
model_config.n_ctx = seq_length
print(model_config)

model = GPT2LMHeadModel(model_config)
print(model)

model.to(device)




# Evaluation before training
model.eval()
output = model(input_ids=trajectories, attention_mask=attention_mask)
y_hat_pre = output.logits.argmax(dim=-1).cpu().numpy()




# Training
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=exp_config.lr)
criterion = torch.nn.CrossEntropyLoss()

for _ in range(exp_config.epochs):
    output = model(input_ids=trajectories, attention_mask=attention_mask)
    loss = criterion(output.logits.view(-1, vocab_size), trajectories.view(-1))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print("Loss:", loss)



# Evaluation after training
model.eval()
output = model(input_ids=trajectories, attention_mask=attention_mask)
y_hat_post = output.logits.argmax(dim=-1).cpu().numpy()



print("trajectories:")
print(trajectories.cpu().numpy())
print()

print("y_hat_pre:")
print(y_hat_pre)
print()

print("y_hat_post:")
print(y_hat_post)
print()