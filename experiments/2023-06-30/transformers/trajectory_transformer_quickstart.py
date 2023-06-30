from transformers import TrajectoryTransformerModel
import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CURRENT DEVICE:", device)

model = TrajectoryTransformerModel.from_pretrained(
    "CarlCochet/trajectory-transformer-halfcheetah-medium-v2"
)

model.to(device)
model.eval()

observations_dim, action_dim, batch_size = 17, 6, 256
seq_length = observations_dim + action_dim + 1
vocab_size = model.config.vocab_size
n_head = model.config.n_head
hidden_size = model.config.n_embd
embed_size_per_head = hidden_size / n_head
num_hidden = model.config.n_layer

trajectories = torch.LongTensor(np.array([np.random.permutation(seq_length) for _ in range(batch_size)])).to(
    device
)
attention_mask = torch.ones_like(trajectories)
targets = torch.LongTensor(np.array([np.random.permutation(seq_length) for _ in range(batch_size)])).to(device)

outputs = model(
    trajectories,
    attention_mask=attention_mask,
    targets=targets,
    use_cache=True,
    output_attentions=True,
    output_hidden_states=True,
    return_dict=True,
)


loss = outputs.loss
logits = outputs.logits
past_key_values = past_key_values = torch.Tensor([[t2.cpu().detach().numpy().tolist() for t2 in t1] for t1 in outputs.past_key_values])
hidden_states = torch.Tensor([t1.cpu().detach().numpy().tolist() for t1 in outputs.hidden_states])
attentions = torch.Tensor([t1.cpu().detach().numpy().tolist() for t1 in outputs.attentions])


print("******** CONFIG ********")
print(model.config)
print()

print("******** MODEL ********")
print(model)
print()


print("******** INPUT ********")
print("Trajectories shape (batch_size, sequence_length):")
print("   ", trajectories.shape)

print()


print("******** OUTPUT ********")

# Single float
print("Loss (scalar):")
print("   ", loss)

# Shape: (batch_size, sequence_length, vocab_size)
print("Logits shape (batch_size, sequence_length, vocab_size):")
print("   ", logits.shape)

# For each of the n layers
# Shape: (batch_size, num_heads, sequence_length, embed_size_per_head)
print("Past key values shape (n_layers, ?, batch_size, num_heads, sequence_length, embed_size_per_head):")
print("   ", past_key_values.shape)

# For each hidden state
# Shape: (batch_size, sequence_length, hidden_size)
print("Hidden states shape (n_layers + 1, batch_size, sequence_length, hidden_size):")
print("   ", hidden_states.shape)

# For each hidden layer
# Shape: (batch_size, num_heads, sequence_length, sequence_length)
print("Attentions shape (n_layers, batch_size, num_heads, sequence_length, sequence_length):")
print("   ", attentions.shape)
print()


y = nn.Softmax(dim=-1)(logits).argmax(dim=-1).cpu().detach()
print("Output sequence shape:")
print("   ", y.shape)

print("Output sequence:")
print("   ", y)