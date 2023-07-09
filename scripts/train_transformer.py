import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm
import copy
import glob
import wandb

from setup_script import setup_script
from file_system import load_pickle, save_pickle
from trajectory_dataset import TrajectoryDataset
from metrics import compute_metrics
from utils import get_file_path_from_config, get_experiment_dir_from_config




# Setup script
exp_config, env, _, accelerator = setup_script()
print('Experiment dir: ', get_experiment_dir_from_config(exp_config))



# Get dataset of trajectories
data_config_pattern = copy.deepcopy(exp_config)
data_config_pattern.seed = '****'
data_config_pattern.exp_naming_strategy = 'default'
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
train_dataset = TrajectoryDataset(
    env,
    train_trajectories,
    seq_len=exp_config.seq_len
)
print("Setting up validation set...")
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
print("Finished setting up datasets.")



# Get model
vocab_size = train_dataset.get_vocab_size()

model_config = GPT2Config()

model_config.vocab_size = vocab_size
model_config.n_positions = exp_config.seq_len
model_config.n_ctx = exp_config.seq_len
model_config.n_layer = exp_config.model.n_layer
model_config.n_head = exp_config.model.n_head
model_config.n_embd = exp_config.model.n_embd

model = GPT2LMHeadModel(model_config)



# Training
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=exp_config.lr, **exp_config.adam)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    steps_per_epoch=len(train_data_loader),
    epochs=exp_config.epochs,
    **exp_config.lr_scheduler,
)
criterion = torch.nn.CrossEntropyLoss()

model, optimizer, train_data_loader, val_data_loader, scheduler = accelerator.prepare(model, optimizer, train_data_loader, val_data_loader, scheduler)

# Define metrics
wandb.define_metric('iteration')
wandb.define_metric('epoch')

wandb.define_metric('train/batch_loss', step_metric='iteration')
wandb.define_metric('train/lr', step_metric='iteration')
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
        
        output = model(input_ids=x)
        y_hat = output.logits.view(-1, vocab_size)
        loss = criterion(y_hat, y.view(-1))

        accelerator.backward(loss)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Log batch metrics
        metrics['train/batch_loss'].append(loss.item())
        wandb.log({
            'train/batch_loss': loss.item(),
            'train/lr': scheduler.get_last_lr()[0],
            'iteration': total_train_iter,
        })


        total_train_iter += 1
        batch_pbar.update(1)
        batch_pbar.set_description(f"  - BATCH LOOP. Loss: {loss.item():.4f}")

    print()
    
    # Validation loop
    model.eval()
    with torch.no_grad():
        val_metrics_raw = compute_metrics(model, val_data_loader, criterion)

        val_metrics = {}
        for metric_name in val_metrics_raw:
            val_metrics['val/' + metric_name] = val_metrics_raw[metric_name]

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
        checkpoints_dir = get_file_path_from_config('checkpoints', exp_config, mkdir=True)
        accelerator.save_state(checkpoints_dir)

        metrics_path = get_file_path_from_config('metrics.pkl', exp_config, mkdir=True)
        save_pickle(metrics, metrics_path)

    epoch_pbar.update(1)
    epoch_pbar.set_description(f"* EPOCH LOOP. Acc: {metrics['val/accuracy'][-1]:.4f}. LAA: {metrics['val/last_action_accuracy'][-1]:.4f}")
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