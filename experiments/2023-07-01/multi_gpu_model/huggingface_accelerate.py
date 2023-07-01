import torch
import torch.nn.functional as F
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

accelerator = Accelerator()
device = accelerator.device

model = torch.nn.Transformer().to(device)
optimizer = torch.optim.Adam(model.parameters())

my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets)

tensor_x = torch.Tensor(my_x) # transform to torch tensor
tensor_y = torch.Tensor(my_y)

dataset = TensorDataset(tensor_x,tensor_y)
data = torch.utils.data.DataLoader(dataset, shuffle=True)

model, optimizer, data = accelerator.prepare(model, optimizer, data)

model.train()
for epoch in range(10):
    for source, targets in data:
        source = source.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = model(source)
        loss = F.cross_entropy(output, targets)

        accelerator.backward(loss)

        optimizer.step()