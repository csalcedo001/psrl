import glob
import copy
import numpy as np
from datasets import Dataset
from transformers import DecisionTransformerConfig, DecisionTransformerModel
from transformers.data.data_collator import DataCollatorWithPadding

from setup_script import setup_script
from file_system import load_pickle, save_json
from trajectory_dataset import trajectories_to_dt_dataset_format
from metrics import compute_raw_accuracy, compute_last_action_accuracy
from utils import get_file_path_from_config




# Setup script
exp_config, env, _, accelerator = setup_script()
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
print("Finished setting up dataset.")

print("Dataset:", val_trajectory_dataset)
print("Train set length:", len(val_trajectory_dataset[0]['observations']))
print("Observations shape:", np.array(val_trajectory_dataset[0]['observations']).shape)
print("Actions shape:     ", np.array(val_trajectory_dataset[0]['actions']).shape)
print("Rewards shape:     ", np.array(val_trajectory_dataset[0]['rewards']).shape)
print("Dones shape:       ", np.array(val_trajectory_dataset[0]['dones']).shape)

data_collator = DataCollatorWithPadding(val_trajectory_data)
print("Finished data processing.")



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




# # Evaluation loop
# model.eval()
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