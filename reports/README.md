# Raw reports

This directory keeps the console evidence generated on the Jetson while
exporting ONNX, building TensorRT engines, validating outputs and benchmarking
the graph. See [BENCHMARKS.md](BENCHMARKS.md) for a concise interpretation.

Absolute `/home/jetson4/...` and `/workspace/...` paths in raw logs document the
original execution environment; they are not credentials. Engine and ONNX
artifacts referenced by the logs are excluded from Git because they are large
and target-specific.
