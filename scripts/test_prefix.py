"""
Test prefix invariance property of TriOD GPT model.

This script verifies that the model satisfies the prefix invariance property:
for any p value, the first p% of neurons in the output should match
the output when running the full model and taking the first p% of neurons.

Run as:
    python -m scripts.test_prefix

Or with custom settings:
    python -m scripts.test_prefix --depth=12 --num_models=5 --min_p=0.2
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import print0, autodetect_device_type

# -----------------------------------------------------------------------------
# Settings
depth = 12  # smaller model for quick testing
max_seq_len = 128
vocab_size = 1024  # small vocab for testing
batch_size = 2
seq_len = 32
# TriOD settings
triangular = True
min_p = 0.2
num_models = 5
# Test settings
rtol = 1e-4  # relative tolerance for comparison
atol = 1e-5  # absolute tolerance for comparison

# Allow CLI overrides
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())

# -----------------------------------------------------------------------------


def test_prefix_invariance():
    """Test that the model satisfies prefix invariance."""
    
    print0("=" * 60)
    print0("Testing TriOD Prefix Invariance")
    print0("=" * 60)
    
    # Device setup
    device_type = autodetect_device_type()
    device = torch.device(device_type)
    print0(f"Using device: {device}")
    
    # Model setup
    print0(f"\nCreating model with depth={depth}, triangular={triangular}")
    print0(f"  min_p={min_p}, num_models={num_models}")
    
    num_layers = depth
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    
    config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        triangular=triangular,
        min_p=min_p,
        num_models=num_models,
    )
    
    model = GPT(config)
    model.to(device)
    model.init_weights()
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print0(f"  Model parameters: {num_params:,}")
    
    # Generate random input
    print0(f"\nGenerating random input: batch_size={batch_size}, seq_len={seq_len}")
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Get p values
    p_s = np.linspace(min_p, 1.0, num_models)
    print0(f"Testing p values: {p_s}")
    
    # Collect outputs for each p value
    print0("\n" + "-" * 60)
    print0("Running forward passes with return_prelast=True")
    print0("-" * 60)
    
    outputs = []
    with torch.no_grad():
        for p in p_s:
            output = model(input_ids, p=p, return_prelast=True)
            # Reshape to 2D for easier comparison
            output_2d = output.reshape(-1, output.shape[-1])
            outputs.append(output_2d.cpu().numpy())
            print0(f"  p={p:.2f}: output shape = {output.shape} -> {output_2d.shape}")
    
    # Get full model output (p=1.0, which is the last one)
    full_output = outputs[-1]
    
    # Test prefix invariance
    print0("\n" + "-" * 60)
    print0("Testing prefix invariance")
    print0("-" * 60)
    
    all_passed = True
    for i, (p, out) in enumerate(zip(p_s, outputs)):
        inner_dim = out.shape[-1]
        
        # Compare with full model's first inner_dim neurons
        full_prefix = full_output[:, :inner_dim]
        
        # Compute differences
        abs_diff = np.abs(full_prefix - out)
        rel_diff = abs_diff / (np.abs(out) + 1e-8)
        
        max_abs_diff = abs_diff.max()
        mean_abs_diff = abs_diff.mean()
        max_rel_diff = rel_diff.max()
        mean_rel_diff = rel_diff.mean()
        
        # Check if passes tolerance
        passes = np.allclose(full_prefix, out, rtol=rtol, atol=atol)
        status = "✓ PASS" if passes else "✗ FAIL"
        all_passed = all_passed and passes
        
        print0(f"\n  p={p:.2f} (dim={inner_dim}):")
        print0(f"    Max  absolute diff: {max_abs_diff:.2e}")
        print0(f"    Mean absolute diff: {mean_abs_diff:.2e}")
        print0(f"    Max  relative diff: {max_rel_diff:.2e}")
        print0(f"    Mean relative diff: {mean_rel_diff:.2e}")
        print0(f"    Status: {status}")
    
    # Summary
    print0("\n" + "=" * 60)
    if all_passed:
        print0("✓ ALL TESTS PASSED - Prefix invariance verified!")
    else:
        print0("✗ SOME TESTS FAILED - Prefix invariance NOT satisfied!")
    print0("=" * 60)
    
    return all_passed


def test_all_models_output():
    """Test that all_models=True produces correct output shapes."""
    
    print0("\n" + "=" * 60)
    print0("Testing all_models=True output")
    print0("=" * 60)
    
    device_type = autodetect_device_type()
    device = torch.device(device_type)
    
    num_layers = depth
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    
    config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        triangular=triangular,
        min_p=min_p,
        num_models=num_models,
    )
    
    model = GPT(config)
    model.to(device)
    model.init_weights()
    model.eval()
    
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        # Test all_models output
        logits = model(input_ids, all_models=True)
        
        expected_batch = num_models * batch_size
        print0(f"\n  Input shape: {input_ids.shape}")
        print0(f"  Output shape: {logits.shape}")
        print0(f"  Expected batch dim: {expected_batch} (num_models={num_models} × batch={batch_size})")
        
        passes = logits.shape[0] == expected_batch
        status = "✓ PASS" if passes else "✗ FAIL"
        print0(f"  Status: {status}")
    
    print0("=" * 60)
    return passes


def test_generation():
    """Test that generation works with different p values."""
    
    print0("\n" + "=" * 60)
    print0("Testing generation with submodels")
    print0("=" * 60)
    
    device_type = autodetect_device_type()
    device = torch.device(device_type)
    
    num_layers = depth
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    
    config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        triangular=triangular,
        min_p=min_p,
        num_models=num_models,
    )
    
    model = GPT(config)
    model.to(device)
    model.init_weights()
    model.eval()
    
    p_s = np.linspace(min_p, 1.0, num_models)
    start_tokens = [1, 2, 3]  # dummy start tokens
    max_tokens = 5
    
    print0(f"\n  Generating {max_tokens} tokens from start={start_tokens}")
    
    all_passed = True
    for p in p_s:
        try:
            generated = []
            for token in model.generate(start_tokens, max_tokens=max_tokens, temperature=1.0, p=p):
                generated.append(token)
            print0(f"  p={p:.2f}: generated {len(generated)} tokens: {generated[:5]}... ✓")
        except Exception as e:
            print0(f"  p={p:.2f}: FAILED with error: {e}")
            all_passed = False
    
    status = "✓ PASS" if all_passed else "✗ FAIL"
    print0(f"\n  Status: {status}")
    print0("=" * 60)
    return all_passed


if __name__ == "__main__":
    print0("\n" + "#" * 60)
    print0("# TriOD PREFIX INVARIANCE TEST SUITE")
    print0("#" * 60)
    
    results = {}
    
    # Run tests
    results["prefix_invariance"] = test_prefix_invariance()
    results["all_models_output"] = test_all_models_output()
    results["generation"] = test_generation()
    
    # Final summary
    print0("\n" + "#" * 60)
    print0("# FINAL SUMMARY")
    print0("#" * 60)
    
    all_passed = all(results.values())
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print0(f"  {test_name}: {status}")
    
    print0("\n" + ("✓ ALL TESTS PASSED!" if all_passed else "✗ SOME TESTS FAILED!"))
    print0("#" * 60 + "\n")
    
    exit(0 if all_passed else 1)
