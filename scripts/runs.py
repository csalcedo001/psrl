import os
import warnings
import datetime
import uuid

DEFAULT_RUNS_DIR = "./runs/"

def get_runs_dir(setup=False):
    runs_dir = os.environ.get("RUNS_DIR", DEFAULT_RUNS_DIR)

    if setup:
        os.makedirs(runs_dir, exist_ok=True)
    
    return os.path.realpath(runs_dir)

def get_experiment_config(config, warn=True):
    runs_dir = get_runs_dir(setup=True)

    experiment_name = config.get("experiment_name", None)
    if experiment_name == None:
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        uuid_str = uuid.uuid4().hex[:8]

        experiment_name = datetime_str + "_" + uuid_str

    experiment_dir = config.get("experiment_dir", None)
    if experiment_dir == None:
        experiment_dir = os.path.join(runs_dir, experiment_name)

    if warn and os.path.exists(experiment_dir):
        warnings.warn(f"Experiment name {experiment_dir} already exists, overwriting...")

    os.makedirs(experiment_dir, exist_ok=True)

    return {
        'experiment_dir': experiment_dir,
        'experiment_name': experiment_name,
    }

def get_experiment_dir(config):
    if 'experiment_dir' in config:
        return config['experiment_dir']
    
    experiment_conf = get_experiment_config(config, warn=False)
    return experiment_conf['experiment_dir']