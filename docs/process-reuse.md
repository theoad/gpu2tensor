# Reusing evaluation processes

Fresh processes remain the default. Operators can reuse a child process to avoid
reimporting Torch and initializing its runtime for every trusted candidate:

```bash
GPU2TENSOR_CACHE_DIR=/var/tmp/gpu2tensor-cache gpu2tensor-worker --backend cuda --reuse-process --max-requests 32
```

Clients keep using `Evaluator`, `observe`, `Pool`, or `KernelEnv` without changes.
Each endpoint still owns one device and accepts one evaluation at a time. Reuse
has been qualified on CUDA; Trainium reuse still needs hardware qualification.

The parent enforces each job's deadline. It replaces the child after 32 jobs by
default, or after any failed evaluation or profile. A crash or timeout kills the
child's process group, including compiler descendants. Closing the worker also
closes its child. Reused results record the PID and evaluation wall time under
`process`; endpoint health reports `process_mode`.

Each job keeps its own directory, source, inputs, outputs and profiler artifacts.
Input restoration and the unprofiled timing path are unchanged. The child drops
candidate/reference/validator modules, restores its environment and known Torch
matmul precision flags, and releases PTX modules after synchronizing the device.
Allocator cleanup happens outside measured kernel calls.

Arbitrary Python modules, native libraries, threads and global settings can still
retain state. Reuse provides weaker state isolation, even with a bounded lifetime.
Keep fresh processes for workloads that need that separation. Neither mode is a
security sandbox. The job limit bounds lifetime, not a candidate's peak memory.

## PTX compilation cache

`GPU2TENSOR_CACHE_DIR` enables an optional PTX assembly cache in both process modes.
It stores only successful cubins, assembler logs and assembly metadata. The key
includes source, target architecture, assembler path/version/file identity and
options. Numerical results, correctness decisions and timings are never cached.
Every request still runs its checks, warmup, timing and requested profiling.
Triton and Neuron retain their own SDK cache policies; this cache does not replace
them or cache Neuron compilation.

Entries are hash-checked, published atomically and limited to 128 entries and
64 MiB per cache directory. Old entries are removed on writes. Concurrent writers
can duplicate compilation; no cross-worker compilation lock is introduced.
`assembly.cache` reports disabled/miss/hit and the key. On a hit, the original
compiler command remains as provenance, while `assembly_seconds` measures lookup
and materialization. Assembler discovery/version probing is outside that interval.
Use an operator-owned writable directory with trusted writers.

## Measure it

For a sequential comparison only, start two idle workers on the same GPU using
ports 8011 (fresh) and 8010 (reused), then forward them to local ports 18003/18004.
Do not submit concurrent work to these endpoints: they share a device.

```bash
python example/collect/reuse.py --fresh http://127.0.0.1:18003 --reused http://127.0.0.1:18004 --trials 4 --output artifacts/reuse
```

The example checks endpoint process modes, input hashes, device identity and
measurement policy. It warms both workers, alternates request order, saves every
result and reports whole-request medians. See [CUDA evidence](results/cuda/README.md)
for the measured result and its limits. Ordinary multi-worker collection still
requires independently owned devices or partitions.
