import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel
import pickle
from tqdm import tqdm
from accelerate import Accelerator

from psrl.config import get_env_config
from psrl.utils import env_name_map

from arg_utils import get_experiment_parser
from utils import load_experiment_config, set_seed, get_file_path_from_config, get_experiment_path_from_config



class TrajectoryDataset(Dataset):
    def __init__(self, env, raw_trajectories, seq_len=64, beta=8):
        if seq_len % 2 != 0:
            raise ValueError("seq_len must be even")
        
        self.seq_len = seq_len
        self.beta = beta
        self.missing_token = env.observation_space.n + env.action_space.n
        
        trajectories = []
        for s, a, _, _ in raw_trajectories:
            trajectories.append(s + env.action_space.n)
            trajectories.append(a)
        
        self.trajectories = trajectories
    
    def get_vocab_size(self):
        return self.missing_token + 1

    def __len__(self):
        return (len(self.trajectories) - self.seq_len) // 2
    
    def __getitem__(self, idx):
        # Get trajectory of length seq_len
        trajectory = np.array(self.trajectories[2 * idx: 2 * idx + self.seq_len])

        y = torch.LongTensor(trajectory)

        # Mask some actions
        missing_actions = 1 + min(int(np.random.exponential(self.beta)), self.seq_len // 2)
        missing_s_and_a = 2 * missing_actions - 1

        x = torch.LongTensor(trajectory.copy())
        x[-missing_s_and_a:] = self.missing_token

        return x, y






# Get experiment configuration
parser = get_experiment_parser()
args = parser.parse_args()
config_path = args.config
exp_config = load_experiment_config(config_path)



# Setup experiment
set_seed(exp_config.seed)
print("*** SEED:", exp_config.seed)

data_dir = get_experiment_path_from_config(exp_config, mkdir=True, root_type='data')
accelerator = Accelerator(project_dir=data_dir)



# Get environment
env_class = env_name_map[exp_config.env]
env_config = get_env_config(exp_config.env)
env_config['gamma'] = exp_config.gamma
env_config['no_goal'] = exp_config.no_goal
env = env_class(env_config)



# Get dataset of trajectories
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
vocab_size = trajectory_dataset.get_vocab_size()

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
model.train()



# Training
optimizer = torch.optim.Adam(model.parameters(), lr=exp_config.lr)
criterion = torch.nn.CrossEntropyLoss()

model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

losses = []

print("Starting training...")
for epoch in range(exp_config.epochs):
    pbar = tqdm(total=len(data_loader))
    for batch in data_loader:
        x, y = batch
        
        output = model(input_ids=x)
        y_hat = output.logits.view(-1, vocab_size)
        loss = criterion(y_hat, y.view(-1))

        accelerator.backward(loss)

        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        pbar.update(1)
        pbar.set_description(f"[{epoch}/{exp_config.epochs}] Loss: {loss.item():.4f}")
    
    if epoch % 10 == 0:
        checkpoints_dir = get_file_path_from_config('checkpoints', exp_config)
        accelerator.save_state(checkpoints_dir)



# Evaluation after training
device = accelerator.device

x, y = next(iter(data_loader))
x = x.to(device)

model.eval()
output = model(input_ids=x)
y_hat_post = output.logits.argmax(dim=-1).cpu().numpy()

print("x:")
print(x.cpu().detach().numpy()[-10:, -20:])
print()

print("y_hat:")
print(y_hat_post[-10:, -20:])
print()

print("y:")
print(y.cpu().numpy()[-10:, -20:])
print()



# Save results
model_path = get_file_path_from_config('model.pt', exp_config, mkdir=True)
torch.save(model.state_dict(), model_path)

losses_path = get_file_path_from_config('losses.pkl', exp_config, mkdir=True)
with open(losses_path, 'wb') as f:
    pickle.dump(losses, f)