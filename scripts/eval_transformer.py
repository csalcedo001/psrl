import copy
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel

from setup_script import setup_script
from file_system import load_pickle, save_json
from trajectory_dataset import TrajectoryDataset
from metrics import compute_metrics
from utils import get_file_path_from_config




# Setup script
exp_config, env, _, accelerator = setup_script('debug')



# Get dataset of trajectories
data_config_pattern = copy.deepcopy(exp_config)
data_config_pattern.seed = '****'
data_path_pattern = get_file_path_from_config('trajectories.pkl', data_config_pattern)
paths = glob.glob(data_path_pattern)
paths.sort()

# Load trajectories
print("Loading trajectories...")
trajectories = []
for trajectory_path in paths:
    trajectory = load_pickle(trajectory_path)
    trajectories.append(trajectory)

# Split into train and val sets
print("Splitting into train and val sets...")
train_val_split_ratio = 0.9
val_trajectories = trajectories[int(len(trajectories) * train_val_split_ratio):]

# Datasets
print("Setting up validation set...")
val_dataset = TrajectoryDataset(
    env,
    val_trajectories,
    seq_len=exp_config.seq_len
)
val_data_loader = DataLoader(
    val_dataset,
    batch_size=exp_config.batch_size,
    shuffle=True,
    drop_last=True,
)
print("Finished setting up datasets.")




# Get model
vocab_size = val_dataset.get_vocab_size()

model_config = GPT2Config()

model_config.vocab_size = vocab_size
model_config.n_positions = exp_config.seq_len
model_config.n_ctx = exp_config.seq_len

model = GPT2LMHeadModel(model_config)

checkpoints_dir = get_file_path_from_config('checkpoints', exp_config)
model = accelerator.prepare(model)
accelerator.load_state(checkpoints_dir)




# Evaluation loop
criterion = nn.CrossEntropyLoss()

model, val_data_loader = accelerator.prepare(model, val_data_loader)
model.eval()
with torch.no_grad():
    eval_metrics = compute_metrics(model, val_data_loader, criterion)



print("Raw accuracy:", eval_metrics['accuracy'])
print("Last action accuracy:", eval_metrics['last_action_accuracy'])

# device = accelerator.device

# x, y = next(iter(data_loader))
# x = x.to(device)

# model.eval()
# output = model(input_ids=x)
# y_hat_post = output.logits.argmax(dim=-1).cpu().numpy()

# print("x:")
# print(x.cpu().detach().numpy()[-10:, -20:])
# print()

# print("y_hat:")
# print(y_hat_post[-10:, -20:])
# print()

# print("y:")
# print(y.cpu().numpy()[-10:, -20:])
# print()



# Save results
eval_metrics_path = get_file_path_from_config('eval_metrics.json', exp_config, root_type='plots', mkdir=True)
save_json(eval_metrics, eval_metrics_path)