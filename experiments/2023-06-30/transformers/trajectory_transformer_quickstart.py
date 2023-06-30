from transformers import TrajectoryTransformerModel
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CURRENT DEVICE:", device)

model = TrajectoryTransformerModel.from_pretrained(
    "CarlCochet/trajectory-transformer-halfcheetah-medium-v2"
)

print("Model config:", model.config)

model.to(device)
model.eval()

observations_dim, action_dim, batch_size = 17, 6, 256
seq_length = observations_dim + action_dim + 1

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

print("Loss:", outputs.loss)
print("Logits shape:", [t1.shape for t1 in outputs.logits])
print("Past key values:", [[t2.shape for t2 in t1] for t1 in outputs.past_key_values])
print("Hidden states shape:", [t1.shape for t1 in outputs.hidden_states])
print("Attentions shape:", [t1.shape for t1 in outputs.attentions])