import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm

from setup_script import setup_script
from file_system import load_pickle, save_pickle
from trajectory_dataset import TrajectoryDataset
from metrics import compute_raw_accuracy, compute_last_action_accuracy
from utils import get_file_path_from_config




# Setup script
exp_config, env, _, accelerator = setup_script()



# Get dataset of trajectories
trajectories_path = get_file_path_from_config('trajectories.pkl', exp_config)
raw_trajectory = load_pickle(trajectories_path)
raw_trajectories = [raw_trajectory]

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



# Training
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=exp_config.lr)
criterion = torch.nn.CrossEntropyLoss()

model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

metrics = {
    'loss': [],
    'raw_accuracy': [],
    'last_action_accuracy': [],
}

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

        metrics['loss'].append(loss.item())

        pbar.update(1)
        pbar.set_description(f"[{epoch}/{exp_config.epochs}] Loss: {loss.item():.4f}")
    
    if epoch % 10 == 0:
        checkpoints_dir = get_file_path_from_config('checkpoints', exp_config)
        accelerator.save_state(checkpoints_dir)

        model.eval()
        with torch.no_grad():
            raw_accuracy = compute_raw_accuracy(model, data_loader)
            last_action_accuracy = compute_last_action_accuracy(model, data_loader)

            metrics['raw_accuracy'].append(raw_accuracy)
            metrics['last_action_accuracy'].append(last_action_accuracy)
        
        model.train()



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
checkpoints_dir = get_file_path_from_config('checkpoints', exp_config)
accelerator.save_state(checkpoints_dir)

model_path = get_file_path_from_config('model.pt', exp_config, mkdir=True)
torch.save(model.state_dict(), model_path)

metrics_path = get_file_path_from_config('metrics.pkl', exp_config, mkdir=True)
save_pickle(metrics, metrics_path)