import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.utils.model_parallel_utils import get_device_map
import time

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)


def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = GPT2LMHeadModel(GPT2Config(
        n_embd=32, n_layer=2, n_head=16, n_positions=128
    ))
    device_map = get_device_map(len(mp_model.transformer.h), range(2))
    mp_model.parallelize(device_map)
    # mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)


    print("Training starting. Run gpustat to see CUDA memory allocation.")
    for i in range(10):
        print("iteration: ", i)
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        # outputs will be on dev1
        outputs = ddp_mp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(dev1)
        loss_fn(outputs, labels).backward()
        optimizer.step()

        time.sleep(1)

    cleanup()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    # run_demo(demo_basic, world_size)
    # run_demo(demo_checkpoint, world_size)
    world_size = n_gpus//2
    run_demo(demo_model_parallel, world_size)