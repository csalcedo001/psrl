import os

from psrl.config import get_env_config
from psrl.utils import env_name_map

from plotting import save_gridworld_plot

from arg_utils import get_experiment_parser, process_experiment_config
from utils import load_experiment_config, set_seed




# Get experiment configuration
parser = get_experiment_parser()
args = parser.parse_args()
config_path = args.config
exp_config = load_experiment_config(config_path)
exp_config = process_experiment_config(args, exp_config)




for env_name in ['tworoom', 'fourroom']:
    # Get environment
    env_class = env_name_map[env_name]
    env_config = get_env_config(env_name)
    env = env_class(env_config)



    # Plot gridworld
    filename = env_name + '.png'
    root = exp_config.plots_path
    experiment_dir = 'gridworlds'

    file_dir = os.path.join(root, experiment_dir)
    file_path = os.path.join(file_dir, filename)

    os.makedirs(file_dir, exist_ok=True)

    save_gridworld_plot(
        env,
        env_name,
        file_path
    )
