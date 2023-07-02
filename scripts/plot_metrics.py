import os
import pickle

from plotting import save_losses_plot, save_accuracy_plot
from arg_utils import get_experiment_parser, process_experiment_config
from utils import load_experiment_config, set_seed, get_file_path_from_config




# Get experiment configuration
parser = get_experiment_parser()
args = parser.parse_args()
config_path = args.config
exp_config = load_experiment_config(config_path)
exp_config = process_experiment_config(args, exp_config)



# Setup experiment
set_seed(exp_config.seed)
print("*** SEED:", exp_config.seed)



# Get losses data
metrics_path = get_file_path_from_config('metrics.pkl', exp_config)
with open(metrics_path, 'rb') as f:
    metrics = pickle.load(f)



# Save plots
print('Saving plots...')
save_losses_plot(
    metrics['loss'],
    get_file_path_from_config('training_loss.png', exp_config, mkdir=True, root_type='plots'),
    title='Training Loss',
)
save_accuracy_plot(
    metrics['raw_accuracy'],
    get_file_path_from_config('training_raw_accuracy.png', exp_config, root_type='plots'),
    title='Raw Accuracy',
)
save_accuracy_plot(
    metrics['last_action_accuracy'],
    get_file_path_from_config('training_last_action_accuracy.png', exp_config, root_type='plots'),
    title='Last Action Accuracy',
)