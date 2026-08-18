# Benchmarks and validation

## Test environment

- Jetson Orin NX 16 GB, Ubuntu 22.04, L4T 36.4.3
- Test audio: `audio/test_vieneu_16k.wav`, 4.64 seconds, mono, 16 kHz
- Browser fixture: `audio/test_vieneu_microphone.webm`, 19,294 bytes
- Model: `nvidia/parakeet-ctc-0.6b-Vietnamese`
- Revision: `b0493142b49458810324e3db8be9e8e07b4ebc17`
- Deployed backend: TensorRT FP16 10.11.0.33
- Shape: mel `[1, 80, 465]`, output `[1, 59, 1025]`

These are measurements from this device and fixture, not universal performance
claims. Audio duration/complexity, clocks, thermals, other GPU work and network
placement affect results.

## PyTorch baselines

| Backend | Inference | RTF | GPU allocated | Peak allocated |
|---|---:|---:|---:|---:|
| NeMo PyTorch FP32 | 155.564 ms | 0.033527 | 2.315 GB | 2.333 GB |
| NeMo PyTorch FP16 | 153.231 ms | 0.033024 | 1.163 GB | 1.172 GB |

Both produced the same transcript for the fixture. The reference contains
product names, so WER 0.6 and CER 0.333 reflect phonetic substitutions such as
“VieNeu” and “Jetson Orin NX”; this is an accuracy characteristic of the model
and test sentence, not a TensorRT mismatch.

Sources: `outputs/baseline_fp32.json`, `outputs/baseline_fp16.json`.

## TensorRT build

| Runtime | Build time | Serialized engine | Result |
|---|---:|---:|---|
| TensorRT 10.3 host | 288.291 s | 1,201.99 MiB | PASSED |
| TensorRT 10.11 container | 245.055 s | 1,195.53 MiB | PASSED |

The deployed 10.11 engine is intentionally not committed. Build plans should be
regenerated on the target GPU/runtime combination.

Sources: `reports/build_tensorrt_fp16.log`,
`reports/build_tensorrt_fp16_trt1011.log`.

## Graph benchmark

The saved `trtexec` benchmark for shape `audio_signal:1x80x465,length:1` used a
1-second warm-up and 5-second duration:

| Metric | Result |
|---|---:|
| Mean latency | 23.5837 ms |
| Median latency | 23.5845 ms |
| p95 latency | 23.6096 ms |
| p99 latency | 23.6246 ms |
| Throughput | 42.2734 qps |

This is graph execution on already-prepared mel input. It excludes audio decode,
mel preprocessing, CTC decoding and HTTP.

Source: `reports/benchmark_tensorrt_t465.log`.

## Numerical validation

TensorRT 10.11 FP16 was compared with the PyTorch FP32 log-probability reference:

| Metric | Result |
|---|---:|
| Reference / TRT shape | `[1, 59, 1025]` |
| Cosine similarity | 0.9999752883 |
| Frame argmax agreement | 1.0 |
| Collapsed CTC tokens equal | true |
| Mean absolute error | 0.0840777 |
| RMSE | 0.1103781 |
| Maximum absolute error | 0.6454334 |

Sources: `test_data/test_vieneu_pytorch_logprobs.npy`,
`test_data/test_vieneu_tensorrt1011_output.json`, and
`workspace/validate_tensorrt_output.py`.

## Persistent runtime

The first complete CLI request included cold library/caching effects:

| Stage | Seconds |
|---|---:|
| Audio load | 0.007240 |
| Mel preprocessing | 0.379792 |
| TensorRT | 0.092177 |
| CTC decode | 0.001469 |
| Total | 0.480687 |

Runtime startup was 16.079625 seconds (model restore 13.471739, engine load
2.381693, warm-up 0.226021). Startup occurs once before readiness and is not
part of steady-state request latency.

Source: `outputs/demo_tensorrt1011.json`.

Five warm WAV HTTP requests measured:

| Metric | Mean |
|---|---:|
| Server end-to-end | 70.077 ms |
| Runtime total | 68.306 ms |
| TensorRT | 57.484 ms |

Sources: `outputs/api/warm-1.json` through `warm-5.json`.

## WebM/Opus acceptance

| Metric | Result |
|---|---:|
| Client end-to-end | 191.587 ms |
| Server end-to-end | 183.475 ms |
| FFmpeg normalization | 113.371 ms |
| Inference pipeline | 68.856 ms |
| TensorRT | 54.833 ms |
| Latency target | 1,500 ms |
| Result | **PASSED** |

The reusable Python client subsequently measured 203.419 ms client end-to-end
and 193.610 ms server end-to-end, also `passed: true`.

Sources: `outputs/api/acceptance-v1.json`,
`outputs/api/server-client-v1.json`.

## Raw evidence index

- `reports/export_onnx.log`: NeMo ONNX export
- `reports/build_tensorrt_fp16.log`: TensorRT 10.3 build
- `reports/build_tensorrt_fp16_trt1011.log`: TensorRT 10.11 build
- `reports/validate_tensorrt_once.log`: one-shot TRT 10.3 output export
- `reports/validate_tensorrt1011_once.log`: one-shot TRT 10.11 output export
- `reports/benchmark_tensorrt_t465.log`: warm graph benchmark
- `outputs/`: complete runtime and API JSON results
- `test_data/`: exact graph inputs and reference/output tensors
