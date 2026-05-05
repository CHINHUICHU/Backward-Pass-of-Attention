# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu124" }
#
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# explicit = true
# ///
"""
Benchmark manual attention vs Flash Attention at the same sequence lengths
used in flash_cuda_slow.cu forward pass benchmarks.

  D = 64, batch = 1, heads = 1, dtype = float16
"""

import warnings
warnings.filterwarnings("ignore")

import math
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.benchmark import Timer

D = 64
BATCH = 1
HEADS = 1

fwd_sizes = [1024, 2048, 4096, 8192, 16384, 32768]

def make_qkv(n):
    shape = (BATCH, HEADS, n, D)
    Q = torch.full(shape, 0.1, dtype=torch.float16, device="cuda")
    K = torch.full(shape, 0.1, dtype=torch.float16, device="cuda")
    V = torch.full(shape, 0.1, dtype=torch.float16, device="cuda")
    return Q, K, V

def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

def bench(fn, globs, warmup_fn=None):
    # Explicit warmup before handing off to Timer
    warmup = warmup_fn or fn
    exec(warmup, globs)
    torch.cuda.synchronize()

    t = Timer(stmt=fn, globals=globs)
    return t.blocked_autorange().median * 1e3  # ms

def tflops(n, ms):
    flops = 4 * n * n * D * BATCH * HEADS  # QK^T + PV, each 2*N^2*D
    return flops / (ms * 1e-3) / 1e12

sep = "=" * 84
print(sep)
print(f"| {'Seq Len':<8} | {'Manual(ms)':<12} | {'Manual TFLOPS':<14} | {'Flash(ms)':<12} | {'Flash TFLOPS':<13} | {'Speedup':<8} |")
print(sep)

for n in fwd_sizes:
    Q, K, V = make_qkv(n)

    # --- Manual: Q @ K^T, softmax, P @ V ---
    try:
        ms_manual = bench("manual_attn(Q, K, V)", {"manual_attn": manual_attn, "Q": Q, "K": K, "V": V})
        manual_ms_str    = f"{ms_manual:<12.2f}"
        manual_tflops_str = f"{tflops(n, ms_manual):<14.2f}"
    except Exception:
        ms_manual = None
        manual_ms_str    = f"{'OOM':<12}"
        manual_tflops_str = f"{'OOM':<14}"

    # --- Flash Attention ---
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            ms_flash = bench(
                "F.scaled_dot_product_attention(Q, K, V)",
                {"F": F, "Q": Q, "K": K, "V": V},
            )
        flash_ms_str    = f"{ms_flash:<12.2f}"
        flash_tflops_str = f"{tflops(n, ms_flash):<13.2f}"
    except Exception:
        ms_flash = None
        flash_ms_str    = f"{'N/A':<12}"
        flash_tflops_str = f"{'N/A':<13}"

    if ms_manual is not None and ms_flash is not None:
        speedup_str = f"{ms_manual / ms_flash:<8.2f}x"
    else:
        speedup_str = f"{'N/A':<8}"

    print(f"| {n:<8} | {manual_ms_str} | {manual_tflops_str} | {flash_ms_str} | {flash_tflops_str} | {speedup_str} |")

print(sep)
print()
