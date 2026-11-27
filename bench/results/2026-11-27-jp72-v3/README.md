# JetPack 7.2.1 native CUDA 13.2 V3 qualification

Jetson AGX Orin, L4T R39.2.1, CUDA 13.2.86, SM87 Release. The native
CLI, server and benchmark link on `feat/jetson-orin-jp72-cuda132`.
Five artifact tests passed, including CUDA materialization. The reader
validated all 1,190 Qwen3.8-27B and all 940 Qwen3.6-35B-A3B V3 objects.

Both groupwise-int artifacts generated 16 tokens from a 22-token chat prompt
with INT8 KV, greedy sampling and optimized MTP-3. The 27B required automatic
context sizing for this setup. Explicit 256 and 512 MiB budgets were rejected
during runtime planning.

| Smoke workload | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| Qwen3.8-27B | 42.0 | 12.7 |
| Qwen3.6-35B-A3B | 84.0 | 62.0 |

The first native performance run used the 35B-A3B V3 artifact,
the first 512 IDs from `bench/fixtures/bench_corpus.ids`, 65 output tokens
(64 decode tokens), INT8 KV, MTP off, CUDA Graphs, no prefix reuse,
a 4,096-token context limit, a 256 MiB context pool and a 1,024-token prefill
chunk. With dynamic clocks, one warmup and three measured repetitions,
median prefill was **1,225.32 tok/s** and median decode was **36.72 tok/s**.
The historic CUDA 12.6 control used a smaller context pool and locked
clocks, so this run establishes native performance for this configuration.
The 2,048-token case used the same settings with 129 output tokens.

| Dynamic-clock workload | TTFT (s) | Prefill (tok/s) | Decode (tok/s) |
| --- | ---: | ---: | ---: |
| 35B-A3B, 512 prompt + 64 decode | 0.419 | 1,225.32 | 36.72 |
| 35B-A3B, 2,048 prompt + 128 decode | 1.470 | 1,394.47 | 36.71 |
| 27B, 512 prompt + 64 decode | 2.041 | 251.01 | 7.31 |
| 27B, 2,048 prompt + 128 decode | 8.155 | 251.17 | 7.29 |

The 27B runs used the same workload settings with automatic context sizing,
which resolved to 6.60 GiB and 9.03 GiB, respectively.

TTFT is the Engine's first-token time. The local JSON reports are under
`build/jp72-v3/`, named by model and workload.

The fixed-clock 35B runs used the historical 130 MiB context pool and verified
the exact saved control token IDs. JetPack 7.2.1 MAXN pinned CPU/GPU/EMC at
2.2016/1.3005/3.199 GHz, the same clocks used for the CUDA 12.6 control.
All repetitions reached their output limits. Peak GPU temperature was 64.7 °C.

| 35B-A3B workload | CUDA 12.6 on JP7 prefill / decode (tok/s) | Native 13.2 prefill / decode (tok/s) | Native 13.2 TTFT (s) |
| --- | ---: | ---: | ---: |
| 512 prompt + 64 decode | 1,221.48 / 36.413 | 1,218.00 / 36.74 | 0.421 |
| 2,048 prompt + 128 decode | 1,435.99 / 36.380 | 1,386.47 / 36.53 | 1.478 |

The native run's short prefill was 0.3% lower and long prefill was 3.4%
lower. Decode was 0.9% and 0.4% higher, respectively. The historic CUDA 12.6
bundle included local Qwen operator edits, so the comparison measures the two
builds. A CUDA 12.6 build from the same published source is in progress.
Local JSON and telemetry are in `build/jp72-v3/fixed/`.

The 27B runs at the same fixed clocks completed all repetitions. The
automatic Context pools resolved to 8.75 GiB and 8.89 GiB. Peak GPU
temperature was 71.9 °C.

| Fixed-clock 27B workload | TTFT (s) | Prefill (tok/s) | Decode (tok/s) |
| --- | ---: | ---: | ---: |
| 512 prompt + 64 decode | 2.041 | 250.99 | 7.31 |
| 2,048 prompt + 128 decode | 8.150 | 251.33 | 7.28 |

All three metrics were within 0.2% of the dynamic-clock runs. Context
sizing also varied across runs. The CPU, GPU and EMC clock policy was
verified after restoring the dynamic settings.

The final native build passed the five artifact tests and the OpenAI and
Anthropic schema tests. Both real V3 artifacts also passed the independent
MTP Main-conditioning oracle with graphs enabled and disabled. Each case
checked 21 KV states. Local logs are in `build/jp72-v3/qualification-tests.log`
and the corresponding `*-mtp-graph*.log` files.

The independent reduction oracle and the expert-cache test also passed.
These exercise the relinked log-probability kernels and the cache's
upload, reuse, and error paths. See `build/jp72-v3/reduction-cache-tests.log`.

Both original models passed the explicit prefix integration test. The 27B
passed the image Vision integration fixture. On the 35B, the same fixture
failed at startup because its default Vision setup exceeded the available
device budget. A public CLI run with a 64×64 PPM, 64 MiB Vision budget,
256 MiB Context pool, INT8 KV and MTP-3 generated all eight requested tokens
from an 85-token expanded prompt. It accepted four of six drafts. This
qualifies the bounded case under the observed memory budget.
