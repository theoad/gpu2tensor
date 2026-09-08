"""Named vendor baselines evaluated through the ordinary workload contract."""

from gpu2tensor.client import Candidate


MATMUL_SOURCE = '''import torch

def run(a, b):
    return torch.matmul(a, b)
'''

STRICT_CUDA_SOURCE = '''import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

def run(a, b):
    return torch.matmul(a, b)
'''


def matmul(backend, *, precision="default"):
    """Use vendor defaults, or a separately named strict CUDA reduction variant."""
    if backend not in ("cuda", "trainium"):
        raise ValueError("Choose cuda or trainium for a vendor matmul baseline.")
    if precision == "default":
        return Candidate(MATMUL_SOURCE, "torch", name="vendor_matmul")
    if precision == "strict" and backend == "cuda":
        return Candidate(STRICT_CUDA_SOURCE, "torch", name="vendor_matmul_strict")
    raise ValueError("Strict precision is a separate CUDA variant; otherwise use default.")
