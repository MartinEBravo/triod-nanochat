"""
Loads a checkpoint, and:
- Evaluates the loss on a larger chunk of train/val splits
- Samples from the model

Example run as:
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss

To evaluate a HuggingFace model:
python -m scripts.base_loss --hf-path openai-community/gpt2

TriOD evaluation (evaluates all submodels):
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss -- --triangular --num-models=10 --min-p=0.1
"""
import argparse
from contextlib import nullcontext
import numpy as np
import torch
from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init, print0, compute_cleanup, autodetect_device_type
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.tokenizer import get_token_bytes, HuggingFaceTokenizer
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine

# -----------------------------------------------------------------------------
# HuggingFace loading utilities, making the APIs match up to those of nanochat

class ModelWrapper:
    """Lightweight wrapper for a HuggingFace model"""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        else:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss

    def get_device(self):
        return next(self.model.parameters()).device

def load_hf_model(hf_path: str, device):
    print0(f"Loading model from: {hf_path}")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "openai-community/gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer

def get_hf_token_bytes(tokenizer, device="cpu"):
    """Compute token_bytes tensor for a HuggingFace tokenizer."""
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    for token_id in range(vocab_size):
        token_str = tokenizer.tokenizer.decode([token_id])
        token_bytes[token_id] = len(token_str.encode('utf-8')) # Count UTF-8 bytes
    return token_bytes

# CLI arguments
parser = argparse.ArgumentParser(description="Evaluate loss on train/val splits and sample from model")
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--split-tokens", type=int, default=40*524288, help="number of tokens to evaluate per split")
parser.add_argument("--model-tag", type=str, default=None, help="model tag for checkpoint directory")
parser.add_argument("--model-step", type=int, default=None, help="model step to load")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--hf-path", type=str, default=None, help="HuggingFace model path (e.g. openai-community/gpt2)")
# TriOD parameters
parser.add_argument("--triangular", action="store_true", help="enable TriOD evaluation across all submodels")
parser.add_argument("--num-models", type=int, default=0, help="number of submodels in TriOD (0 = disabled)")
parser.add_argument("--min-p", type=float, default=1.0, help="smallest submodel in TriOD")
args = parser.parse_args()

# Load the base model and the tokenizer
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
print0(f"Device: {device} | DDP rank: {ddp_rank} | DDP local rank: {ddp_local_rank} | DDP world size: {ddp_world_size}")

if args.hf_path is not None:
    # Load HuggingFace model
    model, tokenizer = load_hf_model(args.hf_path, device)
    sequence_len = model.max_seq_len if model.max_seq_len else 1024
    token_bytes = get_hf_token_bytes(tokenizer, device=device)
    model_name = args.hf_path
else:
    # Load local nanochat model
    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.model_step)
    sequence_len = meta["model_config"]["sequence_len"]
    token_bytes = get_token_bytes(device=device)
    model_name = f"base_model (step {meta['step']})"

autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

print0(f"Evaluating model: {model_name}")

# TriOD: Determine p values to evaluate
if args.triangular and args.num_models > 0:
    p_values = np.linspace(args.min_p, 1.0, args.num_models).tolist()
    print0(f"TriOD enabled: evaluating {len(p_values)} submodels with p values: {p_values}")
else:
    p_values = [None]  # Single evaluation with full model

# Evaluate the loss on each split for each p value
tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
assert args.split_tokens % tokens_per_step == 0, "split_tokens must be divisible by tokens_per_step"
steps = args.split_tokens // tokens_per_step
all_bpb_results = {}  # {p: {split: bpb}}
for p in p_values:
    p_label = f"p={p:.2f}" if p is not None else "full"
    print0(f"\n{'='*50}")
    print0(f"Evaluating submodel: {p_label}")
    print0(f"{'='*50}")
    bpb_results = {}
    for split_name in ["train", "val"]:
        loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, sequence_len, split_name, device=device)
        with autocast_ctx:
            bpb = evaluate_bpb(model, loader, steps, token_bytes, p=p)
        print0(f"{split_name} bpb: {bpb:.4f}")
        bpb_results[split_name] = bpb
        print0(f"Model: {model_name}, {p_label}, {split_name} bpb: {bpb:.6f}")
    all_bpb_results[p_label] = bpb_results

# Master process also samples from the model (only for nanochat models)
all_samples = {}  # {p: [samples]}
if ddp_rank == 0 and args.hf_path is None:
    prompts = [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
        "The planets of the solar system are:",
        "My favorite color is",
        "If 5*x + 3 = 13, then x is",
    ]
    for p in p_values:
        p_label = f"p={p:.2f}" if p is not None else "full"
        print0(f"\n{'='*50}")
        print0(f"Samples from submodel: {p_label}")
        print0(f"{'='*50}")
        engine = Engine(model, tokenizer, p=p)
        samples = []
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            sample_str = tokenizer.decode(sample[0])
            print0(sample_str)
            samples.append(sample_str)
        all_samples[p_label] = samples

# Log to report
from nanochat.report import get_report
report_data = []
for p_label, bpb_results in all_bpb_results.items():
    report_entry = {
        "model": model_name,
        "submodel": p_label,
        "train bpb": bpb_results["train"],
        "val bpb": bpb_results["val"],
    }
    report_data.append(report_entry)
    # Add samples for this p value if available
    if p_label in all_samples:
        report_data.append({f"sample {i} ({p_label})": sample for i, sample in enumerate(all_samples[p_label])})

get_report().log(section="Base model loss", data=report_data)

# Cleanup
compute_cleanup()
