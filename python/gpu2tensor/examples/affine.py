"""An independent pointwise workload: y = 0.5 * x + 0.25."""

import numpy as np
import argparse
from pathlib import Path

from gpu2tensor import Candidate, Evaluator, Workload

REFERENCE = "def reference(x):\n    return 0.5 * x + 0.25\n"

TRITON = '''import torch
import triton
import triton.language as tl

@triton.jit
def affine(x, y, columns: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    value = tl.load(x + row * columns + column, column < columns, other=0.0)
    tl.store(y + row * columns + column, value * 0.5 + 0.25, column < columns)

def run(x):
    y = torch.empty_like(x)
    affine[(x.shape[0],)](x, y, x.shape[1], triton.next_power_of_2(x.shape[1]))
    return y
'''

NKI = '''import nki.language as nl
import nki.isa as nisa

def kernel(x):
    y = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    values = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    result = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=values, src=x)
    nisa.tensor_scalar(dst=result, data=values, op0=nl.multiply, operand0=0.5,
                       op1=nl.add, operand1=0.25)
    nisa.dma_copy(dst=y, src=result)
    return y
'''


def workload(rows=32, columns=128, seed=7):
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=(rows, columns)).astype(np.float32)
    alternating = np.resize(np.array([-8, 0, 8], dtype=np.float32), (rows, columns))
    return Workload(REFERENCE, [(normal,), (normal * 100,), (np.zeros_like(normal),), (alternating,)])


def sources():
    rowwise = TRITON.replace(
        "affine[(x.shape[0],)](x, y, x.shape[1], triton.next_power_of_2(x.shape[1]))",
        "for row in range(x.shape[0]):\n        affine[(1,)](x[row:row+1], y[row:row+1], x.shape[1], triton.next_power_of_2(x.shape[1]))",
    )
    return [rowwise, TRITON]


def trainium_sources():
    start = NKI.index("    values =")
    end = NKI.index("    return y")
    body = NKI[start:end].replace("x.shape", "(1, x.shape[1])")
    body = body.replace("src=x)", "src=x[row:row+1, :])").replace("dst=y,", "dst=y[row:row+1, :],")
    body = "".join("    " + line + "\n" for line in body.splitlines())
    rowwise = NKI[:start] + "    for row in nl.static_range(x.shape[0]):\n" + body + NKI[end:]
    return [rowwise, NKI]


def candidate(backend):
    if backend == "cuda":
        return Candidate(TRITON, "triton", "affine")
    if backend == "trainium":
        return Candidate(NKI, "nki", "affine")
    return Candidate("def run(x):\n    return x * 0.5 + 0.25\n", "python", "affine")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "cuda", "trainium"], required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/affine"))
    args = parser.parse_args()
    worker = Evaluator(args.endpoint) if args.endpoint else Evaluator(backend=args.backend)
    result = worker.evaluate(candidate(args.backend), workload(), profile=args.profile)
    result.save(args.output)
    print(result.record)
    if not result.correct or (args.profile and result.record["profile_status"] != "ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
