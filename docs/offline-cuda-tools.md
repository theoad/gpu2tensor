# Offline PTX assembly and cubin inspection

This is a proposed operator procedure for an existing Linux x86-64 CPU host,
such as `trail-x86`. It has no GPU dependency and runs no candidate kernel. At
the source review, that host had none of `nvdisasm`, `cuobjdump` or
`neuron-explorer` on PATH. Nothing was installed or extracted there for this
change. These CUDA components do not supply Neuron Explorer.

The official [CUDA 12.8.1 redistributable manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json)
pins the following Linux x86-64 archives. The compiler component includes
`ptxas`; its 12.8.93 version matches the previously recorded assembler version.
Record the executable hash as well: matching version labels alone do not prove
identical tool bytes.

| Purpose | Component | Download bytes | SHA256 |
| --- | --- | ---: | --- |
| PTX assembly | [cuda_nvcc 12.8.93](https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz) | 79015464 | `9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3` |
| Cubin disassembly | [cuda_nvdisasm 12.8.90](https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvdisasm/linux-x86_64/cuda_nvdisasm-linux-x86_64-12.8.90-archive.tar.xz) | 5085152 | `8ac63f4079c47707ec958af3a97ceeb18194e16827a2766058f5872f6c3412ae` |
| Optional alternative disassembler | [cuda_cuobjdump 12.8.90](https://developer.download.nvidia.com/compute/cuda/redist/cuda_cuobjdump/linux-x86_64/cuda_cuobjdump-linux-x86_64-12.8.90-archive.tar.xz) | 207644 | `dbf41355a0f7a8aa491c1839a6c6ff8df02cdb6c6eefd66360543616c1b4980f` |

Only the first two are needed for the commands below. No driver, CUDA runtime,
Torch, Triton, NVRTC or system CUDA installation is needed for these offline
operations. Normal host C/C++ runtime libraries and `curl`, `tar` with xz support,
`sha256sum` and `timeout` must already be available. If a host dependency is
missing, stop and report it. Retain the supplied NVIDIA licenses with the tools;
these archives are operator dependencies, not gpu2tensor distribution contents.

Run this proposed Bash block in a task-specific session on the authorized host.
It creates a fresh task directory, checks exact archive hashes before extracting,
and uses absolute binary paths. It neither changes global PATH nor uses sudo.

```bash
set -euo pipefail
CUDA_AUDIT_DIR=$(mktemp -d "${TMPDIR:-/tmp}/gpu2tensor-cuda-12.8.1.XXXXXX")
CUDA_REDIST=https://developer.download.nvidia.com/compute/cuda/redist
mkdir "$CUDA_AUDIT_DIR/compiler" "$CUDA_AUDIT_DIR/disassembler"

curl --fail --location --connect-timeout 10 --max-time 180 \
  "$CUDA_REDIST/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz" \
  --output "$CUDA_AUDIT_DIR/nvcc.tar.xz"
curl --fail --location --connect-timeout 10 --max-time 180 \
  "$CUDA_REDIST/cuda_nvdisasm/linux-x86_64/cuda_nvdisasm-linux-x86_64-12.8.90-archive.tar.xz" \
  --output "$CUDA_AUDIT_DIR/nvdisasm.tar.xz"

printf '%s  %s\n' \
  9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3 "$CUDA_AUDIT_DIR/nvcc.tar.xz" \
  8ac63f4079c47707ec958af3a97ceeb18194e16827a2766058f5872f6c3412ae "$CUDA_AUDIT_DIR/nvdisasm.tar.xz" \
  | sha256sum --check
tar -xJf "$CUDA_AUDIT_DIR/nvcc.tar.xz" --strip-components=1 -C "$CUDA_AUDIT_DIR/compiler"
tar -xJf "$CUDA_AUDIT_DIR/nvdisasm.tar.xz" --strip-components=1 -C "$CUDA_AUDIT_DIR/disassembler"
timeout 10s "$CUDA_AUDIT_DIR/compiler/bin/ptxas" --version
timeout 10s "$CUDA_AUDIT_DIR/disassembler/bin/nvdisasm" --version
sha256sum "$CUDA_AUDIT_DIR/compiler/bin/ptxas" "$CUDA_AUDIT_DIR/disassembler/bin/nvdisasm"
```

For one archived result, set IN to its absolute directory. Use a fresh OUT
directory. The assembly target and options must match that result's
`assembly.json`; `sm_89` below is an RTX 4090 example, not a default for every
GPU. Hash and preserve the archived input files; compile into OUT only.

```bash
OUT=$(mktemp -d "${TMPDIR:-/tmp}/gpu2tensor-cuda-inspect.XXXXXX")
sha256sum "$IN/candidate.ptx" "$IN/candidate.cubin"
timeout 120s "$CUDA_AUDIT_DIR/compiler/bin/ptxas" -arch sm_89 -v \
  "$IN/candidate.ptx" -o "$OUT/reassembled.cubin" > "$OUT/ptxas.log" 2>&1
timeout 60s "$CUDA_AUDIT_DIR/disassembler/bin/nvdisasm" \
  "$IN/candidate.cubin" > "$OUT/original.sass" 2> "$OUT/original.log"
timeout 60s "$CUDA_AUDIT_DIR/disassembler/bin/nvdisasm" \
  "$OUT/reassembled.cubin" > "$OUT/reassembled.sass" 2> "$OUT/reassembled.log"
sha256sum "$OUT/reassembled.cubin" "$OUT/original.sass" "$OUT/reassembled.sass"
```

Apply the [inspection disk/file bounds](instruction-artifacts.md) to this task
directory as well. This command sequence can establish assembly and static
lowering evidence. It cannot establish GPU correctness, dynamic issue order or
performance. Stop on an unsupported instruction/architecture or missing host
library and preserve the diagnostic; no automatic alternate tool installation.
