"""
GPT model with TriOD (Triangular Ordered Dropout) support.
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- TriOD: Triangular connectivity for prefix invariance
- TriOD: Support for p parameter to select submodel size
- TriOD: all_models mode for training multiple submodels
"""

import math
from functools import partial
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from triod.layers.linear import TriODLinear
from triod.layers.layer_norm import TriODHeadLayerNorm
from triod.utils import generate_structured_masked_x

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    # TriOD parameters
    triangular: bool = True  # Enable triangular ordered dropout
    min_p: float = 0.2  # Minimum p for smallest submodel
    num_models: int = 5  # Number of submodels to train


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.triangular = config.triangular
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # TriOD linear layers with triangular connectivity
        self.c_q = TriODLinear(self.n_embd, self.n_head * self.head_dim, bias=False, triangular=self.triangular)
        self.c_k = TriODLinear(self.n_embd, self.n_kv_head * self.head_dim, bias=False, triangular=self.triangular)
        self.c_v = TriODLinear(self.n_embd, self.n_kv_head * self.head_dim, bias=False, triangular=self.triangular)
        self.c_proj = TriODLinear(self.n_embd, self.n_embd, bias=False, triangular=self.triangular)
        # TriOD LayerNorm for prefix-invariant normalization
        self.ln = TriODHeadLayerNorm(self.n_embd, self.n_head, triangular=self.triangular)

    def forward(self, x, cos_sin, kv_cache, p=None):
        B, T, C = x.size()

        # Calculate dimensions based on p for TriOD
        if p is not None:
            keep_heads = max(1, math.ceil(self.n_head * p))
            keep_kv_heads = max(1, math.ceil(self.n_kv_head * p))
            cur_dim = keep_heads * self.head_dim
        else:
            keep_heads = self.n_head
            keep_kv_heads = self.n_kv_head
            cur_dim = self.n_embd

        # Apply TriOD LayerNorm
        x_norm = self.ln(x)

        # Project the input to get queries, keys, and values
        q = self.c_q(x_norm, p=p).view(B, T, keep_heads, self.head_dim)
        k = self.c_k(x_norm, p=p).view(B, T, keep_kv_heads, self.head_dim)
        v = self.c_v(x_norm, p=p).view(B, T, keep_kv_heads, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)  # QK rotary embedding
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)  # number of queries in this forward pass
        Tk = k.size(2)  # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = keep_heads != keep_kv_heads  # GQA with TriOD-adjusted heads
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # During inference but with a single query in this forward pass
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, cur_dim)
        y = self.c_proj(y, p=p)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.triangular = config.triangular
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        # TriOD linear layers
        self.c_fc = TriODLinear(config.n_embd, 4 * config.n_embd, bias=False, triangular=self.triangular)
        self.c_proj = TriODLinear(4 * config.n_embd, config.n_embd, bias=False, triangular=self.triangular)
        # TriOD LayerNorm
        self.ln = TriODHeadLayerNorm(config.n_embd, config.n_head, triangular=self.triangular)

    def forward(self, x, p=None):
        x = self.ln(x)
        x = self.c_fc(x, p=p)
        x = F.relu(x).square()
        x = self.c_proj(x, p=p)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache, p=None):
        # TriOD: pass p parameter through attention and MLP
        # Note: norm is now inside attn and mlp for TriOD compatibility
        x = x + self.attn(x, cos_sin, kv_cache, p=p)
        x = x + self.mlp(x, p=p)
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.triangular = config.triangular
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        # Store p_s for submodel training (TriOD)
        self.p_s = np.linspace(config.min_p, 1.0, config.num_models) if config.triangular else None
        # For DDP, we want vocab_size divisible by world_size. Also, there are potential performance benefits, see:
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} to be divisible by {pad_vocab_size_to}")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        # TriOD: lm_head is never triangular (output dim is fixed)
        self.lm_head = TriODLinear(config.n_embd, padded_vocab_size, bias=False, triangular=False)
        # TriOD: Final LayerNorm
        self.final_ln = TriODHeadLayerNorm(config.n_embd, config.n_head, triangular=config.triangular)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        self.rotary_seq_len = config.sequence_len * 10  # 10X over-compute should be enough
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights (TriOD: handle 3D weight shape)
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, TriODLinear)):
            # https://arxiv.org/pdf/2310.17813
            weight = module.weight
            if weight.ndim == 3:  # TriODLinear with blocks
                fan_out = weight.size(0) * weight.size(1)
                fan_in = weight.size(2)
            else:
                fan_out = weight.size(0)
                fan_in = weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', p=None, return_prelast=False, all_models=False):
        """
        Forward pass with TriOD support.
        
        Args:
            idx: Input token indices (B, T)
            targets: Target token indices for loss computation
            kv_cache: Optional KV cache for inference
            loss_reduction: 'mean' or 'none' for loss computation
            p: Optional fraction of model to use (for inference with submodels)
            return_prelast: Return pre-logit representations (for testing prefix invariance)
            all_models: Generate outputs for all submodels (for training with KL loss)
        """
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)

        # TriOD: Apply p to embedding (slice to submodel dimension)
        if p is not None:
            keep_heads = max(1, math.ceil(self.n_head * p))
            keep_dim = keep_heads * (self.n_embd // self.n_head)
            x = x[:, :, :keep_dim]

        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache, p=p)
        
        # TriOD: Use prefix-invariant final LayerNorm
        x = self.final_ln(x)

        # Return pre-logit representations for testing prefix invariance
        if return_prelast:
            return x

        # TriOD: Generate outputs for all submodels
        if all_models and self.p_s is not None:
            x = generate_structured_masked_x(x, self.p_s)

        # Forward the lm_head (compute logits)
        softcap = 15  # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x, p=None)  # No triangular in lm_head
        logits = logits[..., :self.config.vocab_size]  # slice to remove padding
        logits = logits.float()  # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap)  # squash the logits

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits
    
    def forward_with_kl_loss(self, idx, targets, kl_alpha=0.5, loss_reduction='mean'):
        """
        Forward pass with KL distillation loss for training submodels.
        
        Args:
            idx: Input token indices (B, T)
            targets: Target token indices
            kl_alpha: Weight for KL divergence loss
            loss_reduction: 'mean' or 'none' for loss computation
        
        Returns:
            Tuple of (total_loss, ce_loss, kl_loss)
        """
        B, T = idx.size()

        # Get outputs for all submodels: returns (n_models * B, T, vocab_size) logits
        logits = self.forward(idx, p=None, all_models=True)

        # Split into submodels
        n_models = len(self.p_s)
        logits_per_model = logits.view(n_models, B, T, -1)

        # Full model output (last one)
        logits_full = logits_per_model[-1]  # (B, T, vocab_size)

        # CE loss on full model
        ce_loss = F.cross_entropy(
            logits_full.view(-1, logits_full.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )

        # KL loss: submodels learn from full model
        kl_loss = torch.tensor(0.0, device=idx.device)
        if n_models > 1 and kl_alpha > 0.0:
            # Soft targets from full model (no gradient)
            with torch.no_grad():
                soft_targets = F.softmax(logits_full, dim=-1)  # (B, T, vocab_size)

            # KL loss for each submodel (except the full model)
            for i in range(n_models - 1):
                submodel_logits = logits_per_model[i]  # (B, T, vocab_size)
                kl_loss = kl_loss + F.cross_entropy(
                    submodel_logits.view(-1, submodel_logits.size(-1)),
                    soft_targets.view(-1, soft_targets.size(-1)),
                    reduction=loss_reduction
                )
            kl_loss = kl_loss / (n_models - 1)  # Average over submodels

        total_loss = ce_loss + kl_alpha * kl_loss
        return total_loss, ce_loss, kl_loss

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42, p=None):
        """
        Naive autoregressive streaming inference with TriOD submodel support.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        
        Args:
            tokens: List of starting token indices
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            seed: Random seed
            p: Optional fraction of model to use (for submodel inference)
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids, p=p)  # TriOD: pass p for submodel inference
            logits = logits[:, -1, :]  # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
