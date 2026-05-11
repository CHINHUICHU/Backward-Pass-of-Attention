# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""
Benchmark each attention step separately on CPU (single thread) using PyTorch.
Steps: QK^T, softmax, PV.

  D = 64, batch = 1, heads = 1, dtype = float32
"""

import csv
import math
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer

torch.set_num_threads(1)
assert torch.get_num_threads() == 1, "thread count mismatch"
print(f"[info] torch threads : {torch.get_num_threads()}")
print(f"[info] PID           : {os.getpid()}  (check with htop -p {os.getpid()})")

D = 64
BATCH = 1
HEADS = 1
WARMUP = 5

fwd_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

def make_qkv(n):
    shape = (BATCH, HEADS, n, D)
    Q = torch.randn(shape, dtype=torch.float32)
    K = torch.randn(shape, dtype=torch.float32)
    V = torch.randn(shape, dtype=torch.float32)
    return Q, K, V

def bench(fn, stmt, globs):
    for _ in range(WARMUP):
        fn()
    t = Timer(stmt=stmt, globals=globs)
    return t.blocked_autorange().median * 1e3  # ms

sys.stdout.flush()
out_path = "cpu_benchmark_1.csv"
fields = ["seq_len", "qkt_ms", "qkt_pct", "softmax_ms", "softmax_pct", "pv_ms", "pv_pct", "total_ms"]

with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for n in fwd_sizes:
        print(f"n={n}...", file=sys.stderr)
        Q, K, V = make_qkv(n)
        scale = 1.0 / math.sqrt(D)

        ms_qkt = bench(lambda: Q @ K.transpose(-2, -1) * scale,
                       "Q @ K.transpose(-2,-1) * scale", {"Q": Q, "K": K, "scale": scale})
        S = Q @ K.transpose(-2, -1) * scale
        ms_sm  = bench(lambda: F.softmax(S, dim=-1),
                       "F.softmax(S, dim=-1)", {"F": F, "S": S})
        P = F.softmax(S, dim=-1)
        ms_pv  = bench(lambda: P @ V,
                       "P @ V", {"P": P, "V": V})

        total = ms_qkt + ms_sm + ms_pv
        writer.writerow({
            "seq_len":    n,
            "qkt_ms":     round(ms_qkt, 4),
            "qkt_pct":    round(ms_qkt / total * 100, 1),
            "softmax_ms": round(ms_sm, 4),
            "softmax_pct": round(ms_sm / total * 100, 1),
            "pv_ms":      round(ms_pv, 4),
            "pv_pct":     round(ms_pv / total * 100, 1),
            "total_ms":   round(total, 4),
        })

print(f"Saved to {out_path}")
