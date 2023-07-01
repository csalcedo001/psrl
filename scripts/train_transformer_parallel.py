import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm
import pickle

from psrl.config import get_env_config
from psrl.utils import env_name_map

from arg_utils import get_experiment_parser
from utils import load_experiment_config, set_seed, get_file_path_from_config



class TrajectoryDataset(Dataset):
    def __init__(self, env, raw_trajectories, seq_len=64, beta=8):
        if seq_len % 2 != 0:
            raise ValueError("seq_len must be even")
        
        self.seq_len = seq_len
        self.beta = beta
        self.missing_token = env.observation_space.n + env.action_space.n
        
        trajectories = []
        for s, a, _, _ in raw_trajectories:
            trajectories.append(s + env.action_space.n)
            trajectories.append(a)
        
        self.trajectories = trajectories
    
    def get_vocab_size(self):
        return self.missing_token + 1

    def __len__(self):
        return (len(self.trajectories) - self.seq_len) // 2
    
    def __getitem__(self, idx):
        # Get trajectory of length seq_len
        trajectory = np.array(self.trajectories[2 * idx: 2 * idx + self.seq_len])

        y = torch.LongTensor(trajectory)

        # Mask some actions
        missing_actions = 1 + min(int(np.random.exponential(self.beta)), self.seq_len // 2)
        missing_s_and_a = 2 * missing_actions - 1

        x = torch.LongTensor(trajectory.copy())
        x[-missing_s_and_a:] = self.missing_token

        return x, y


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()






# Get experiment configuration
parser = get_experiment_parser()
args = parser.parse_args()
config_path = args.config
exp_config = load_experiment_config(config_path)



def train_parallel(rank, world_size, vocab_size, data_loader):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # Get model
    model_config = GPT2Config()

    model_config.vocab_size = vocab_size
    model_config.n_positions = exp_config.seq_len
    model_config.n_ctx = exp_config.seq_len

    model = GPT2LMHeadModel(model_config)
    # model.parallelize()
    model = DDP(model)
    model.train()



    # # Training
    # optimizer = torch.optim.Adam(model.parameters(), lr=exp_config.lr)
    # criterion = torch.nn.CrossEntropyLoss()

    # losses = []

    # print("Starting training...")
    # for epoch in range(exp_config.epochs):
    #     pbar = tqdm(total=len(data_loader))
    #     for batch in data_loader:
    #         x, y = batch

    #         # x = x.to(device)
    #         # y = y.to(device)
            
    #         output = model(input_ids=x)
    #         y_hat = output.logits.view(-1, vocab_size)
    #         loss = criterion(y_hat, y.view(-1))

    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()

    #         losses.append(loss.item())

    #         pbar.update(1)
    #         pbar.set_description(f"[{epoch}/{exp_config.epochs}] Loss: {loss.item():.4f}")
    
    # # model.deparallelize()



    # # Evaluation after training
    # x, y = next(iter(data_loader))

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
    # print(y.numpy()[-10:, -20:])
    # print()



    # # Save results
    # model_path = get_file_path_from_config('model.pt', exp_config, mkdir=True)
    # torch.save(model.state_dict(), model_path)

    # losses_path = get_file_path_from_config('losses.pkl', exp_config, mkdir=True)
    # with open(losses_path, 'wb') as f:
    #     pickle.dump(losses, f)


    cleanup()


if __name__ == '__main__':
    n_gpus = torch.cuda.device_count()
    print("DEVICE COUNT:", n_gpus)
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    # run_demo(demo_basic, world_size)
    # run_demo(demo_checkpoint, world_size)
    world_size = n_gpus//2




    # Setup experiment
    set_seed(exp_config.seed)
    print("*** SEED:", exp_config.seed)



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
        drop_last=True,
    )



    # Run parallel training
    mp.spawn(train_parallel,
        args=(world_size, vocab_size, data_loader),
        nprocs=world_size,
        join=True)