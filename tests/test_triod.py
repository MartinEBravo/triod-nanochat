import torch
import numpy as np

from nanochat.gpt import GPT, GPTConfig
from triod.utils import test_prefix_od
from nanochat.tokenizer import get_tokenizer, get_token_bytes


# Model configuration (same defaults as base_train.py)
depth = 20
aspect_ratio = 64
head_dim = 128
max_seq_len = 2048
window_pattern = "SSSL"

# Model kwargs are derived from the desired depth of the model
num_layers = depth
model_dim = depth * aspect_ratio

def find_num_heads(model_dim, target_head_dim):
    # Find num_heads that divides model_dim evenly, with head_dim closest to target.
    ideal = max(1, round(model_dim / target_head_dim))
    for offset in range(model_dim):
        for candidate in [ideal + offset, ideal - offset]:
            if candidate > 0 and model_dim % candidate == 0:
                return candidate
    return 1

num_heads = find_num_heads(model_dim, head_dim)
num_kv_heads = num_heads  # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)

# TriOD parameters (disabled by default like base_train)
triangular = True
num_models = 4
min_p = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Model dimension: {model_dim}, Number of heads: {num_heads}, Number of KV heads: {num_kv_heads}, Number of layers: {num_layers}")

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()

# TRIOD parameters
p_s = np.linspace(min_p, 1.0, num_models)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim, window_pattern=window_pattern, triangular=triangular, p_s=p_s)

model_config = GPTConfig(**model_config_kwargs)

amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model = GPT(model_config).to(device).to(dtype=amp_dtype)
model.init_weights()

T = 2048
B = 4
x = torch.randint(0, vocab_size, (B, T), device=device, dtype=torch.long)
y = torch.randint(0, vocab_size, (B, T), device=device, dtype=torch.long)

dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x, y), batch_size=B
)

print("model device:", next(model.parameters()).device)
print("model dtype:", next(model.parameters()).dtype)
print("x device:", x.device)

model.eval()

with torch.inference_mode():
    9(
        model=model,
        device=device,
        dataloader=dataloader,
        p_s=p_s,
    )