"""Evaluate a complete softmax kernel, locally on a worker or through a tunnel."""

import argparse
from pathlib import Path

import numpy as np

from gpu2tensor import Candidate, Evaluator, Workload

REFERENCE = "import torch\ndef reference(x):\n    return torch.softmax(x, dim=-1)\n"

TRITON = '''import torch
import triton
import triton.language as tl

@triton.jit
def softmax(x, y, columns: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    values = tl.load(x + row * columns + column, column < columns, other=-float("inf"))
    values = values - tl.max(values, axis=0)
    numerator = tl.exp(values)
    result = numerator / tl.sum(numerator, axis=0)
    tl.store(y + row * columns + column, result, column < columns)

def run(x):
    y = torch.empty_like(x)
    softmax[(x.shape[0],)](x, y, x.shape[1], triton.next_power_of_2(x.shape[1]), num_warps=4)
    return y
'''

# Kept explicit so a contributor can read the full submitted program.
NKI = '''import nki.language as nl
import nki.isa as nisa

def kernel(x):
    rows, columns = x.shape
    y = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    values = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    negative_max = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    numerator = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    denominator = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    reciprocal = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    result = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=values, src=x)
    nisa.tensor_reduce(dst=negative_max, op=nl.max, data=values, axis=1, negate=True)
    nisa.activation(dst=numerator, op=nl.exp, data=values, bias=negative_max)
    nisa.tensor_reduce(dst=denominator, op=nl.add, data=numerator, axis=1)
    nisa.reciprocal(dst=reciprocal, data=denominator)
    nisa.tensor_scalar(dst=result, data=numerator, op0=nl.multiply, operand0=reciprocal)
    nisa.dma_copy(dst=y, src=result)
    return y
'''


def sources():
    rowwise = TRITON.replace(
        "softmax[(x.shape[0],)](x, y, x.shape[1], triton.next_power_of_2(x.shape[1]), num_warps=4)",
        "for row in range(x.shape[0]):\n        softmax[(1,)](x[row:row+1], y[row:row+1], x.shape[1], triton.next_power_of_2(x.shape[1]), num_warps=4)",
    )
    return [rowwise, TRITON]


def trainium_sources():
    # The simple baseline processes one row per tile. The other program uses
    # all rows in one tile. Both submit complete NKI programs to the same worker.
    start = NKI.index("    values =")
    end = NKI.index("    return y")
    body = NKI[start:end].replace("x.shape", "(1, columns)").replace("(rows, 1)", "(1, 1)")
    body = body.replace("src=x)", "src=x[row:row+1, :])").replace("dst=y,", "dst=y[row:row+1, :],")
    body = "".join("    " + line + "\n" for line in body.splitlines())
    rowwise = NKI[:start] + "    for row in nl.static_range(rows):\n" + body + NKI[end:]
    return [rowwise, NKI]


def workload(rows=32, columns=128, seed=7):
    rng = np.random.default_rng(seed)
    random = rng.normal(size=(rows, columns)).astype(np.float32)
    ramp = np.broadcast_to(np.linspace(-50, 50, columns, dtype=np.float32), (rows, columns)).copy()
    cases = [(random,), (random * 20,), (np.zeros_like(random),), (ramp,)]
    return Workload(REFERENCE, cases)


def candidate(backend):
    if backend == "cuda":
        return Candidate(TRITON, "triton", "softmax")
    if backend == "trainium":
        return Candidate(NKI, "nki", "softmax")
    return Candidate("import torch\ndef run(x):\n    return torch.softmax(x, dim=-1)\n", "python", "softmax")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cuda", "trainium", "cpu"], required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--output", type=Path, default=Path("artifacts/softmax"))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--columns", type=int, default=128)
    args = parser.parse_args()
    evaluator = Evaluator(args.endpoint) if args.endpoint else Evaluator(backend=args.backend)
    result = evaluator.evaluate(candidate(args.backend), workload(args.rows, args.columns), profile=args.profile)
    result.save(args.output)
    print(result.record)
    if not result.correct or (args.profile and result.record["profile_status"] != "ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
