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

| Dynamic-clock 35B-A3B workload | TTFT (s) | Prefill (tok/s) | Decode (tok/s) |
| --- | ---: | ---: | ---: |
| 512 prompt + 64 decode | 0.419 | 1,225.32 | 36.72 |
| 2,048 prompt + 128 decode | 1.470 | 1,394.47 | 36.71 |

TTFT is the Engine's first-token time. The local JSON reports are
`build/jp72-v3/35b-pp512-tg64.json` and `build/jp72-v3/35b-pp2048-tg128.json`.
