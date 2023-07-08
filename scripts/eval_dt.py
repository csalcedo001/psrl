import glob
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datasets import DatasetDict, Dataset
from transformers import DecisionTransformerConfig, DecisionTransformerModel
from tqdm import tqdm

from setup_script import setup_script
from file_system import load_pickle, save_json
from trajectory_dataset import trajectories_to_dt_dataset_format, DecisionTransformerGymDataCollator
from utils import get_file_path_from_config




# Setup script
exp_config, env, _, accelerator = setup_script('debug')
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
val_trajectories = trajectories[int(len(trajectories) * train_val_split_ratio):]

print("Setting up validation set...")
val_trajectory_data = trajectories_to_dt_dataset_format(
    val_trajectories,
    env.observation_space.n,
    env.action_space.n,
    max_ep_len,
)
val_trajectory_dataset = Dataset.from_dict(val_trajectory_data)

trajectory_dataset = DatasetDict({
    'val': val_trajectory_dataset,
})
print("Finished setting up dataset.")

print("Dataset:", val_trajectory_dataset)
print("Train set length:", len(val_trajectory_dataset[0]['observations']))
print("Observations shape:", np.array(val_trajectory_dataset[0]['observations']).shape)
print("Actions shape:     ", np.array(val_trajectory_dataset[0]['actions']).shape)
print("Rewards shape:     ", np.array(val_trajectory_dataset[0]['rewards']).shape)
print("Dones shape:       ", np.array(val_trajectory_dataset[0]['dones']).shape)

data_collator = DecisionTransformerGymDataCollator(trajectory_dataset['val'])
print("Finished data processing.")

data_loader = DataLoader(
    val_trajectory_dataset,
    batch_size=exp_config.batch_size,
    collate_fn=data_collator,
    shuffle=True,
    drop_last=True,
)



# Get model
model_config = DecisionTransformerConfig()

model_config.max_length = max_ep_len
model_config.state_dim = env.observation_space.n
model_config.act_dim = env.action_space.n
model_config.max_ep_len = exp_config.training_steps

model = DecisionTransformerModel(model_config)
checkpoints_dir = get_file_path_from_config('dt_checkpoints', exp_config)
accelerator.load_state(checkpoints_dir)
print("Model:")
print(model)




# Evaluation loop
model.eval()

batch = next(iter(data_loader))
print("batch.keys():", batch.keys())
print("batch['actions'].shape:", batch['actions'].shape)

output = model(**batch)
criterion = nn.CrossEntropyLoss()

print("output.keys():", output.keys())
print("output.action_preds.shape():", output.action_preds.shape)

# with torch.no_grad():
#     for batch in tqdm(data_loader):
#         output = model(**batch)

#         loss = criterion(output.action_preds, batch['actions'])

#         print("Loss:", loss.item())

    
with torch.no_grad():
    # Validation loop
    model.eval()

    with torch.no_grad():
        loss = 0
        total_batch_loss = 0
        hits = 0
        hits_and_misses = 0

        pbar = tqdm(total=len(data_loader))
        for i, batch in enumerate(data_loader):
            x = batch['actions']
            y = batch['actions']

            b_size = x.shape[0]

            # Compute loss
            output = model(**batch)
            y_logits = output.action_preds
            # print("Y:", y.shape)
            # print("Y_logits:", y_logits.shape)

            # y_logits = output.logits.view(b_size, -1)
            # batch_loss = criterion(y_logits, y)
            # total_batch_loss += batch_loss.item()
            # loss = total_batch_loss / (i + 1)

            action_preds = output[1]
            action_targets = batch['actions']
            attention_mask = batch['attention_mask']
            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            
            batch_loss = torch.mean((action_preds - action_targets) ** 2)
            total_batch_loss += batch_loss.item()
            loss = total_batch_loss / (i + 1)

            # Compute accuracy
            y_hat = y_logits
            hits += torch.sum(y == y_hat).item()
            hits_and_misses += torch.ones_like(y).sum().item()
            accuracy = hits / hits_and_misses

            pbar.update(1)
            pbar.set_description(f"  - RAW ACCURACY. Loss: {loss:.4f}. Acc: {accuracy:.4f}")
        
        print("Y:", y[:2])
        print("Y_HAT:", y_hat[:2])
        
        pbar.close()
    
        val_metrics = {
            'val/loss': loss,
            'val/accuracy': accuracy,
        }
    
    model.train()

for metric in val_metrics:
    print(metric, ":", val_metrics[metric])

eval_metrics_path = get_file_path_from_config('dt_eval_metrics.json', exp_config, root_type='plots', mkdir=True)
save_json(val_metrics, eval_metrics_path)


# output = model()
# with torch.no_grad():
#     raw_accuracy = compute_raw_accuracy(model, data_loader)
#     last_action_accuracy = compute_last_action_accuracy(model, data_loader)


# print("Raw accuracy:", raw_accuracy)
# print("Last action accuracy:", last_action_accuracy)

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



# # Save results
# eval_metrics = {
#     'raw_accuracy': raw_accuracy,
#     'last_action_accuracy': last_action_accuracy,
# }

# eval_metrics_path = get_file_path_from_config('eval_metrics.json', exp_config, root_type='plots', mkdir=True)
# save_json(eval_metrics, eval_metrics_path)