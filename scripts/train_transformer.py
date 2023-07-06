import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm
import copy
import glob

from setup_script import setup_script
from file_system import load_pickle, save_pickle
from trajectory_dataset import TrajectoryDataset
from metrics import compute_raw_accuracy, compute_last_action_accuracy
from utils import get_file_path_from_config




# Setup script
exp_config, env, _, accelerator = setup_script()



# Get dataset of trajectories
data_config_pattern = copy.deepcopy(exp_config)
data_config_pattern.seed = '****'
data_path_pattern = get_file_path_from_config('trajectories.pkl', data_config_pattern)
paths = glob.glob(data_path_pattern)
paths.sort()

# Load trajectories
trajectories = []
for trajectory_path in paths:
    trajectory = load_pickle(trajectory_path)
    trajectories.append(trajectory)

# Split into train and val sets
train_val_split_ratio = 0.9
train_trajectories = trajectories[:max(int(len(trajectories) * train_val_split_ratio), 1)]
val_trajectories = trajectories[int(len(trajectories) * train_val_split_ratio):]

# Datasets
train_dataset = TrajectoryDataset(
    env,
    train_trajectories,
    seq_len=exp_config.seq_len
)
val_dataset = TrajectoryDataset(
    env,
    val_trajectories,
    seq_len=exp_config.seq_len
)

train_data_loader = DataLoader(
    train_dataset,
    batch_size=exp_config.batch_size,
    shuffle=True,
    drop_last=True,
)
val_data_loader = DataLoader(
    val_dataset,
    batch_size=exp_config.batch_size,
    shuffle=True,
    drop_last=True,
)



# Get model
vocab_size = train_dataset.get_vocab_size()

model_config = GPT2Config()

model_config.vocab_size = vocab_size
model_config.n_positions = exp_config.seq_len
model_config.n_ctx = exp_config.seq_len

model = GPT2LMHeadModel(model_config)



# Training
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=exp_config.lr)
criterion = torch.nn.CrossEntropyLoss()

model, optimizer, train_data_loader, val_data_loader = accelerator.prepare(model, optimizer, train_data_loader, val_data_loader)

metrics = {
    'loss': [],
    'raw_accuracy': [],
    'last_action_accuracy': [],
}

print("Starting training...")

epoch_pbar = tqdm(total=exp_config.epochs)
epoch_pbar.set_description(f"* EPOCH LOOP. Accuracy: NA")
print()
for epoch in range(exp_config.epochs):

    batch_pbar = tqdm(total=len(train_data_loader))
    batch_pbar.set_description(f"  - BATCH LOOP. Loss: NA")
    for i, batch in enumerate(train_data_loader):
        x, y = batch
        
        output = model(input_ids=x)
        y_hat = output.logits.view(-1, vocab_size)
        loss = criterion(y_hat, y.view(-1))

        accelerator.backward(loss)

        optimizer.step()
        optimizer.zero_grad()

        metrics['loss'].append(loss.item())

        batch_pbar.update(1)
        batch_pbar.set_description(f"  - BATCH LOOP. Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        raw_accuracy = compute_raw_accuracy(model, val_data_loader)
        last_action_accuracy = compute_last_action_accuracy(model, val_data_loader)

        metrics['raw_accuracy'].append(raw_accuracy)
        metrics['last_action_accuracy'].append(last_action_accuracy)
    
    model.train()
    
    if (epoch - 1) % 10 == 0:
        checkpoints_dir = get_file_path_from_config('checkpoints', exp_config)
        accelerator.save_state(checkpoints_dir)
    
    epoch_pbar.update(1)
    epoch_pbar.set_description(f"* EPOCH LOOP. Accuracy: {last_action_accuracy:.4f}")
    print()



# Evaluation after training
device = accelerator.device

x, y = next(iter(train_data_loader))
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
checkpoints_dir = get_file_path_from_config('checkpoints', exp_config)
accelerator.save_state(checkpoints_dir)

model_path = get_file_path_from_config('model.pt', exp_config, mkdir=True)
torch.save(model.state_dict(), model_path)

metrics_path = get_file_path_from_config('metrics.pkl', exp_config, mkdir=True)
save_pickle(metrics, metrics_path)