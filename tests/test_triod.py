import torch
import numpy as np

from nanochat.gpt import GPT, GPTConfig
from triod.utils import test_prefix_od


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = GPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=4,
        n_head=8,
        n_kv_head=8,
        n_embd=768,
        window_pattern="SL",
        triangular=True,
        min_p=0.2,
        num_models=1,
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