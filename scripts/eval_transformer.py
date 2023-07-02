import os
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
import pickle
from tqdm import tqdm
from accelerate import Accelerator

from psrl.config import get_env_config
from psrl.utils import env_name_map

from arg_utils import get_experiment_parser, process_experiment_config
from trajectory_dataset import TrajectoryDataset
from metrics import compute_raw_accuracy, compute_last_action_accuracy
from utils import load_experiment_config, set_seed, get_file_path_from_config, get_experiment_path_from_config






# Get experiment configuration
parser = get_experiment_parser()
args = parser.parse_args()
config_path = args.config
exp_config = load_experiment_config(config_path)
exp_config = process_experiment_config(args, exp_config)



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