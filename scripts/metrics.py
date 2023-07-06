import torch
from tqdm import tqdm

def compute_raw_accuracy(model, data_loader):
    accuracies = []
    pbar = tqdm(total=len(data_loader.dataset))
    for x, y in data_loader:
        output = model(input_ids=x)
        y_hat = output.logits.argmax(dim=-1)

        accuracy = torch.sum(y == y_hat, axis=1).float() / (y.shape[0] * y.shape[1])
        accuracies.append(accuracy)

        pbar.update(1)
        pbar.set_description(f"  - RAW ACCURACY. Accuracy: {torch.mean(accuracy).item():.4f}")
    
    accuracy = torch.mean(torch.concatenate(accuracies))

    return accuracy.item()

def compute_last_action_accuracy(model, data_loader):
    hits = 0
    pbar = tqdm(total=len(data_loader.dataset))
    for _, y in data_loader:
        x = y.clone().detach()
        x[:, -1] = data_loader.dataset.missing_token

        output = model(input_ids=x)
        y_hat = output.logits.argmax(dim=-1)

        batch_hits = torch.sum(y[:, -1] == y_hat[:, -1])
        hits += batch_hits

        pbar.update(1)
        pbar.set_description(f"  - LAST ACTION ACCURACY. Hits: {hits}/{len(data_loader.dataset)}")
    
    hits_and_misses = len(data_loader.dataset)
    accuracy = hits / hits_and_misses

    return accuracy.item()