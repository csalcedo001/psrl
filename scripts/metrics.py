import torch
from tqdm import tqdm

def compute_accuracy(model, data_loader):
    accuracies = []
    for x, y in tqdm(data_loader):
        output = model(input_ids=x)
        y_hat = output.logits.argmax(dim=-1)

        accuracy = torch.sum(y == y_hat, axis=1).float() / (y.shape[0] * y.shape[1])
        accuracies.append(accuracy)
    accuracy = torch.mean(torch.concatenate(accuracies))

    return accuracy