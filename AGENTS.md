# Working on gpu2tensor

Read [the work board](docs/backlog.md) before changing code. Record evidence and
unfinished work there. Never describe a planned backend as working.

Keep Python clients simple and synchronous. Observation-only collection must
work without Gym, a policy, rewards, or action callbacks. An action submits a
complete kernel source or IR. Clients own workload correctness and rewards.

Use plain English names, small functions, and explicit ownership. Package Python
lives under `python/`, uses absolute imports, and has one import identity. Put
native code under `native/` with one CMake project when a real native component
is needed. Follow the cpu2tensor IDE and lean C++20 conventions for that work.
Do not add empty backends or a common accelerator instruction language.

Keep explanations in `docs/`. Example implementations live under the Python
source root; `example/` contains thin runnable entry points, datasets and guides.
Do not duplicate training loops or change `sys.path` in a launcher.
New code is AGPL-3.0-only. Dependencies and
reference code need their own license review before copying.

Operators own machines, credentials, networking, drivers, and SDK installation.
The user authorized two AWS Spot development runners in us-east-1 using the CLI
default profile (authenticated as root). Do not put credentials in this repo.
Keep inventory and launch IDs in ignored `.local/`. See docs/environments.md.
The active matmul campaign shares a total $100 AWS cap with the experiment thread
and stops by 2026-09-09 06:00 UTC. Read ignored `.local/campaign.json` for its
coordinator and live instance. Coordinate before additional launches; there is no
separate instrumentation budget. Reuse the existing worker and reserve timing
slots so qualification and experiment workloads do not overlap on the same device.

Measure uninstrumented latency separately from profiles. Record hardware and
toolchain identity, units, replay, sampling, and missing/dropped data. Do not
compare timings across different devices as if they were the same experiment.
Correctness checks are empirical, not a proof of equivalence. Keep native SDK
imports lazy so importing the client needs no accelerator or vendor SDK.

For independent agent work, use the ownership recipe in docs/development.md.
One owner edits shared contracts. The runtime has no agent dependency.
