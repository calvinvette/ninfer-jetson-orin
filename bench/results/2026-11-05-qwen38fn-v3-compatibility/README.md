# Qwen3.8-Flash-Next NInfer artifact compatibility check

Date: 2026-11-05. Jetson AGX Orin, aarch64, CUDA 12.6, SM87. The attempted
baseline used this checkout's native Orin port binary,
`build/port-cuda126-sm87/bench/ninfer_bench`, against the local
`~/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer` artifact.

## Outcome: artifact unsupported; no performance baseline

The one-cell benchmark request was `pp512+tg64`, 2,048-token context,
BF16 KV, MTP off, one measured repetition, no warmup, and prefix reuse off.
The port exited during artifact open with:

```text
ninfer_bench: artifact magic is not NInfer v1 or v2
```

The artifact begins with `NINFER\0\3`; its conversion manifest identifies
`ninfer-v3` and Qwen3.8-Flash-Next / Qwen4Exp. This Orin port's reader accepts
NInfer v1/v2 framing, and the target is not registered in this checkout. The
reader rejected the header before model loading or inference, so no PP/TG,
latency, quality, or memory-residency measurement was produced. Renaming the
artifact or bypassing its framing check would not make it a valid baseline.

The command ran under the six-GiB physical-memory guard. It exited in 1.08 s,
with 24.57 GiB minimum available memory; the existing cgroup OOM-kill count
remained one. Clock sampling was read-only and dynamic; EMC readback was
unavailable and energy was not measured. The local model files were not copied
or modified.

This result confirms a compatibility boundary, not a performance result. A
usable baseline requires the separate Qwen3.8-Flash-Next/Qwen4 target/runtime
work identified in the [port plan's deferred follow-on](../../../NINFER_JETSON_ORIN_PORT_PLAN.md#deferred-follow-on-work),
including support for its artifact version and registered model semantics.

## Reproduction

Run from the repository root on the Orin host:

```sh
LD_LIBRARY_PATH=./build/jetson-deps/install/lib \
  python3 /home/calvin/workspace/strata-ninfer-expert-e0/tools/integration/run_control.py \
  --output profiles/bench/qwen38fn-baseline/supervisor \
  --sample-clocks --timeout 300 -- \
  ./build/port-cuda126-sm87/bench/ninfer_bench \
  --weights /home/calvin/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer \
  --prompt-gen 512,64 --max-ctx 2048 --no-prefix-reuse \
  -r 1 --warmup 0 --output json \
  --output-file profiles/bench/qwen38fn-baseline/bench.json
```

The supervisor capture is local under `profiles/bench/qwen38fn-baseline/` and
is not committed.
