
import copy
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import TrajectoryTransformerModel, TrajectoryTransformerConfig
from tqdm import tqdm
import wandb

from setup_script import setup_script
from file_system import load_pickle, save_pickle
from trajectory_dataset import TrajectoryTransformerDataset
from metrics import compute_metrics
from utils import get_file_path_from_config




# Setup script
exp_config, env, _, accelerator = setup_script('run')
max_ep_len = 1000



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
train_trajectories = trajectories[:max(int(len(trajectories) * train_val_split_ratio), 1)]
val_trajectories = trajectories[int(len(trajectories) * train_val_split_ratio):]

# Datasets
print("Setting up training set...")
train_dataset = TrajectoryTransformerDataset(
    env,
    train_trajectories,
    seq_len=exp_config.seq_len
)
print("Setting up validation set...")
val_dataset = TrajectoryTransformerDataset(
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
print("Finished setting up datsets.")




# Get model
vocab_size = train_dataset.get_vocab_size()

model_config = TrajectoryTransformerConfig()

model_config.vocab_size = vocab_size
model_config.action_dim = 1
model_config.observation_dim = 1

# One transition is the embeddings of states, actions, reward and reward-to-go.
# The state and action spaces are already discrete (tokenized), so each count
# for one dimension of the transition. The reward and return-to-go are also
# one dimensional.
model_config.transition_dim = 4

model = TrajectoryTransformerModel(model_config)

print("Model config:")
print(model.config)

print("Model:")
print(model)




# Training
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=exp_config.lr)
criterion = torch.nn.CrossEntropyLoss()

model, optimizer, train_data_loader, val_data_loader = accelerator.prepare(model, optimizer, train_data_loader, val_data_loader)

# Define metrics
wandb.define_metric('iteration')
wandb.define_metric('epoch')

wandb.define_metric('train/batch_loss', step_metric='iteration')
wandb.define_metric('train/loss', step_metric='epoch')
wandb.define_metric('val/loss', step_metric='epoch')
wandb.define_metric('val/accuracy', step_metric='epoch')
wandb.define_metric('val/last_action_accuracy', step_metric='epoch')


# Start training loop
metrics = {
    'train/batch_loss': [],
    'train/loss': [],
    'val/loss': [],
    'val/accuracy': [],
    'val/last_action_accuracy': [],
}

print("Starting training...")

total_train_iter = 0
epoch_pbar = tqdm(total=exp_config.epochs)
epoch_pbar.set_description(f"* EPOCH LOOP. Acc: NA. LAA: NA")
print()
for epoch in range(exp_config.epochs):
    # Training loop
    batch_pbar = tqdm(total=len(train_data_loader))
    batch_pbar.set_description(f"  - BATCH LOOP. Loss: NA")

    for i, batch in enumerate(train_data_loader):
        x, y = batch
        
        output = model(
            x,
            targets=y,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        loss = output.loss

        accelerator.backward(loss)

        optimizer.step()
        optimizer.zero_grad()

        # Log batch metrics
        metrics['train/batch_loss'].append(loss.item())
        wandb.log({
            'train/batch_loss': loss.item(),
            'iteration': total_train_iter,
        })


        total_train_iter += 1
        batch_pbar.update(1)
        batch_pbar.set_description(f"  - BATCH LOOP. Loss: {loss.item():.4f}")

    print()
    
    # Validation loop
    model.eval()

    with torch.no_grad():
        loss = 0
        total_batch_loss = 0
        hits = 0
        hits_and_misses = 0
        last_action_hits = 0
        last_action_hits_and_misses = 0

        pbar = tqdm(total=len(val_data_loader))
        for i, batch in enumerate(val_data_loader):
            x, y = batch
            
            output = model(
                x,
                targets=y,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

            # Compute loss
            batch_loss = output.loss
            total_batch_loss += batch_loss.item()
            loss = total_batch_loss / (i + 1)

            # Compute accuracy
            y_hat = output.logits.argmax(dim=-1)
            hits += torch.sum(y == y_hat).item()
            hits_and_misses += torch.ones_like(y).sum().item()
            accuracy = hits / hits_and_misses

            # Compute last action accuracy

            batch_hits = torch.sum(y[:, -4:] == y_hat[:, -4:]).item()
            last_action_hits += batch_hits
            last_action_hits_and_misses += torch.ones_like(y[:, -4:]).sum().item()
            last_action_accuracy = last_action_hits / last_action_hits_and_misses

            pbar.update(1)
            pbar.set_description(f"  - RAW ACCURACY. Loss: {loss:.4f}. Acc: {accuracy:.4f}. LAA: {last_action_accuracy:.4f}")
        
        pbar.close()
    
        val_metrics = {
            'val/loss': loss,
            'val/accuracy': accuracy,
            'val/last_action_accuracy': last_action_accuracy
        }

        for metric_name, metric_val in val_metrics.items():
            metrics[metric_name].append(metric_val)
    
    model.train()

    # Log epoch metrics
    epoch_metrics = {
        'epoch': epoch,
        'train/loss': metrics['train/batch_loss'][-1],
        **val_metrics,
    }
    wandb.log(epoch_metrics)
    
    # Save checkpoint every 5 epochs
    if (epoch - 1) % 10 == 0:
        checkpoints_dir = get_file_path_from_config('tt_checkpoints', exp_config)
        accelerator.save_state(checkpoints_dir)

        metrics_path = get_file_path_from_config('tt_metrics.pkl', exp_config, mkdir=True)
        save_pickle(metrics, metrics_path)

    epoch_pbar.update(1)
    epoch_pbar.set_description(f"* EPOCH LOOP. Acc: {metrics['val/accuracy'][-1]:.4f}. LAA: {metrics['val/last_action_accuracy'][-1]:.4f}")
    print()



# Save results
checkpoints_dir = get_file_path_from_config('tt_checkpoints', exp_config)
accelerator.save_state(checkpoints_dir)

model_path = get_file_path_from_config('tt_model.pt', exp_config, mkdir=True)
torch.save(model.state_dict(), model_path)

metrics_path = get_file_path_from_config('tt_metrics.pkl', exp_config, mkdir=True)
save_pickle(metrics, metrics_path)