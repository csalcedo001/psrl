import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel

from setup_script import setup_script
from file_system import load_pickle, save_json
from trajectory_dataset import TrajectoryDataset
from metrics import compute_raw_accuracy, compute_last_action_accuracy
from utils import get_file_path_from_config




# Setup script
exp_config, env, _, accelerator = setup_script()



# Get dataset of trajectories
trajectories_path = get_file_path_from_config('trajectories.pkl', exp_config)
raw_trajectories = load_pickle(trajectories_path)

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
)

data_loader = accelerator.prepare(data_loader)




# Get model
model_config = GPT2Config()

model_config.vocab_size = vocab_size
model_config.n_positions = exp_config.seq_len
model_config.n_ctx = exp_config.seq_len

model = GPT2LMHeadModel(model_config)

checkpoints_dir = get_file_path_from_config('checkpoints', exp_config)
model = accelerator.prepare(model)
accelerator.load_state(checkpoints_dir)




# Evaluation loop
model.eval()
with torch.no_grad():
    raw_accuracy = compute_raw_accuracy(model, data_loader)
    last_action_accuracy = compute_last_action_accuracy(model, data_loader)



print("Raw accuracy:", raw_accuracy)
print("Last action accuracy:", last_action_accuracy)

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
eval_metrics = {
    'raw_accuracy': raw_accuracy,
    'last_action_accuracy': last_action_accuracy,
}

eval_metrics_path = get_file_path_from_config('eval_metrics.json', exp_config, root_type='plots', mkdir=True)
save_json(eval_metrics, eval_metrics_path)