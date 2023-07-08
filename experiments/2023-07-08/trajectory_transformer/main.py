from transformers import TrajectoryTransformerModel
import torch
import numpy as np



# Get sample data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
seq_len = 100

trajectories = torch.LongTensor([np.random.permutation(seq_len) for _ in range(batch_size)]).to(device)
targets = torch.LongTensor([np.random.permutation(seq_len) for _ in range(batch_size)]).to(device)

print("trajectories.shape:", trajectories.shape)
print("targets.shape:", trajectories.shape)



# Get model
model = TrajectoryTransformerModel.from_pretrained(
    "CarlCochet/trajectory-transformer-halfcheetah-medium-v2"
)
model.to(device)

print("Model config:")
print(model.config)

print("Model:")
print(model)



# Run evaluation loop
model.eval()
outputs = model(
    trajectories,
    targets=targets,
    use_cache=True,
    output_attentions=True,
    output_hidden_states=True,
    return_dict=True,
)

print("output.keys()", outputs.keys())
print("output.loss:", outputs.loss)
print("outputs.logits.shape:", outputs.logits.shape)