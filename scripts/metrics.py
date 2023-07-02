import torch
from tqdm import tqdm

def compute_raw_accuracy(model, data_loader):
    accuracies = []
    for x, y in tqdm(data_loader):
        output = model(input_ids=x)
        y_hat = output.logits.argmax(dim=-1)

        accuracy = torch.sum(y == y_hat, axis=1).float() / (y.shape[0] * y.shape[1])
        accuracies.append(accuracy)
    
    accuracy = torch.mean(torch.concatenate(accuracies))

    return accuracy.item()

def compute_last_action_accuracy(model, data_loader):
    hits = 0
    for _, y in tqdm(data_loader):
        x = y.clone().detach()
        x[:, -1] = data_loader.dataset.missing_token

        output = model(input_ids=x)
        y_hat = output.logits.argmax(dim=-1)

        batch_hits = torch.sum(y[:, -1] == y_hat[:, -1])
        hits += batch_hits
    
    hits_and_misses = len(data_loader.dataset)
    accuracy = hits / hits_and_misses

    return accuracy.item()