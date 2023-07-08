import torch
from tqdm import tqdm

def compute_metrics(model, data_loader, criterion):
    vocab_size = data_loader.dataset.get_vocab_size()

    loss = 0
    total_batch_loss = 0
    hits = 0
    hits_and_misses = 0
    last_action_hits = 0
    last_action_hits_and_misses = 0

    pbar = tqdm(total=len(data_loader))
    for i, (x, y) in enumerate(data_loader):
        # Compute loss
        output = model(input_ids=x)
        y_logits = output.logits
        vocab_size = y_logits.shape[-1]
        y_logits = y_logits.reshape(-1, vocab_size)
        batch_loss = criterion(y_logits, y.reshape(-1))
        total_batch_loss += batch_loss.item()
        loss = total_batch_loss / (i + 1)

        # Compute accuracy
        y_hat = output.logits.argmax(dim=-1)
        hits += torch.sum(y == y_hat).item()
        hits_and_misses += torch.ones_like(y).sum().item()
        accuracy = hits / hits_and_misses

        # Compute last action accuracy
        x = y.clone().detach()
        x[:, -1] = data_loader.dataset.missing_token

        output = model(input_ids=x)
        y_hat = output.logits.argmax(dim=-1)

        batch_hits = torch.sum(y[:, -1] == y_hat[:, -1]).item()
        last_action_hits += batch_hits
        last_action_hits_and_misses += torch.ones_like(y[:, -1]).sum().item()
        last_action_accuracy = last_action_hits / last_action_hits_and_misses

        pbar.update(1)
        pbar.set_description(f"  - RAW ACCURACY. Loss: {loss:.4f}. Acc: {accuracy:.4f}. LAA: {last_action_accuracy:.4f}")
    
    pbar.close()

    return {
        'loss': loss,
        'accuracy': accuracy,
        'last_action_accuracy': last_action_accuracy
    }


def compute_raw_accuracy(model, data_loader):
    accuracies = []
    pbar = tqdm(total=len(data_loader.dataset))
    for x, y in data_loader:
        output = model(input_ids=x)
        y_hat = output.logits.argmax(dim=-1)

        accuracy = torch.sum(y == y_hat, axis=1).float() / (y.shape[0] * y.shape[1])
        accuracies.append(accuracy)

        pbar.update(1)
        pbar.set_description(f"  - RAW ACCURACY. Acc: {torch.mean(accuracy).item():.4f}")
    
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
        pbar.set_description(f"  - LAST ACTION ACCURACY. LAA: {hits}/{len(data_loader.dataset)}")
    
    hits_and_misses = len(data_loader.dataset)
    accuracy = hits / hits_and_misses

    return accuracy.item()