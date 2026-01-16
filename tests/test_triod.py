import torch
import numpy as np

from nanochat.gpt import GPT, GPTConfig
from triod.utils import test_prefix_od
from nanochat.base_train import args, find_num_heads
from nanochat.tokenizer import get_tokenizer, get_token_bytes




# Model kwargs are derived from the desired depth of the model
model_dim = args.depth * args.aspect_ratio
num_heads = find_num_heads(model_dim, args.head_dim)
num_kv_heads = num_heads # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
num_layers = args.depth
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()

# TRIOD parameters
p_s = np.linspace(args.min_p, 1.0, args.num_models)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config_kwargs = dict(sequence_len=args.max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim, window_pattern=args.window_pattern, triangular=args.triangular, p_s=p_s)

    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)

    config = GPTConfig(

    )

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model = GPT(config).to(device).to(dtype=amp_dtype)
    model.init_weights()

    B, T = 2, 128
    x = torch.randint(0, config.vocab_size, (B, T), device=device, dtype=torch.long)
    y = torch.randint(0, config.vocab_size, (B, T), device=device, dtype=torch.long)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=B
    )

    print("model device:", next(model.parameters()).device)
    print("model dtype:", next(model.parameters()).dtype)
    print("x device:", x.device)

    model.eval()

    with torch.inference_mode():
        test_prefix_od(
            model=model,
            device=device,
            dataloader=dataloader,
            p_s=np.linspace(0.2, 1.0, 5),
        )