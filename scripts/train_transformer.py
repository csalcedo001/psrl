import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel
import pickle
from tqdm import tqdm

from psrl.config import get_env_config
from psrl.utils import env_name_map

from utils import load_experiment_config, set_seed, get_file_path_from_config



class TrajectoryDataset(Dataset):
    def __init__(self, env, raw_trajectories, seq_len=1024):
        trajectories = []
        for s, a, _, _ in raw_trajectories:
            trajectories.append(s + env.action_space.n)
            trajectories.append(a)
        
        self.trajectories = trajectories
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.trajectories) - self.seq_len) // 2
    
    def __getitem__(self, idx):
        trajectory = self.trajectories[2 * idx: 2 * idx + self.seq_len]
        return torch.LongTensor(trajectory)






# Get experiment configuration
config_path = os.path.join(os.path.dirname(__file__), 'configs', 'exp_config.yaml')
exp_config = load_experiment_config(config_path)



# Setup experiment
set_seed(exp_config.seed)
print("*** SEED:", exp_config.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("*** CURRENT DEVICE:", device)



# Get environment
env_class = env_name_map[exp_config.env]
env_config = get_env_config(exp_config.env)
env_config['gamma'] = exp_config.gamma
env_config['no_goal'] = exp_config.no_goal
env = env_class(env_config)



# Get dataset of trajectories
vocab_size = env.observation_space.n + env.action_space.n

checkpoints_path = os.path.join(os.path.dirname(__file__), exp_config.save_path)
os.makedirs(checkpoints_path, exist_ok=True)

trajectories_path = get_file_path_from_config('trajectories.pkl', exp_config)
with open(trajectories_path, 'rb') as f:
    raw_trajectories = pickle.load(f)

trajectory_dataset = TrajectoryDataset(
    env,
    raw_trajectories,
    seq_len=exp_config.seq_len
)

data_loader = DataLoader(
    trajectory_dataset,
    batch_size=exp_config.batch_size,
    shuffle=True,
    drop_last=True,
)



# Get model
model_config = GPT2Config()

model_config.vocab_size = vocab_size
model_config.n_positions = exp_config.seq_len
model_config.n_ctx = exp_config.seq_len

model = GPT2LMHeadModel(model_config)
model.to(device)
model.train()



# Training
optimizer = torch.optim.Adam(model.parameters(), lr=exp_config.lr)
criterion = torch.nn.CrossEntropyLoss()

losses = []

print("Starting training...")
for epoch in range(exp_config.epochs):
    pbar = tqdm(total=len(data_loader))
    for batch in data_loader:
        batch = batch.to(device)

        attention_mask = torch.ones_like(batch).to(device)
        attention_mask[:, -1:] = 0

        output = model(input_ids=batch, attention_mask=attention_mask)
        loss = criterion(output.logits.view(-1, vocab_size), batch.view(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        pbar.update(1)
        pbar.set_description(f"[{epoch}/{exp_config.epochs}] Loss: {loss.item():.4f}")



# Evaluation after training
sample_batch = next(iter(data_loader)).to(device)

model.eval()
output = model(input_ids=sample_batch, attention_mask=attention_mask)
y_hat_post = output.logits.argmax(dim=-1).cpu().numpy()


print("trajectories:")
print(sample_batch.cpu().numpy()[-10:, -20:])
print()

print("y_hat:")
print(y_hat_post[-10:, -20:])
print()



# Save results
model_path = get_file_path_from_config('model.pt', exp_config, mkdir=True)
torch.save(model.state_dict(), model_path)

losses_path = get_file_path_from_config('losses.pkl', exp_config, mkdir=True)
with open(losses_path, 'wb') as f:
    pickle.dump(losses, f)