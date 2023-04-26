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

def get_experiment_dir(config):
    runs_dir = get_runs_dir(setup=True)

    experiment_name = config.get("experiment_name", None)
    if experiment_name == None:
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        uuid_str = uuid.uuid4().hex[:8]

        experiment_name = datetime_str + "_" + uuid_str

    experiment_dir = os.path.join(runs_dir, experiment_name)

    if os.path.exists(experiment_dir):
        warnings.warn(f"Experiment name {experiment_dir} already exists, overwriting...")

    os.makedirs(experiment_dir, exist_ok=True)

    return experiment_dir