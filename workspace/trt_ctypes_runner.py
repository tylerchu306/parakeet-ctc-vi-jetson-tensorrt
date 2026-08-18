#!/usr/bin/env python3
"""Minimal TensorRT 10 runner using NumPy and CUDA Runtime through ctypes."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Mapping

import numpy as np
import tensorrt as trt


class CudaError(RuntimeError):
    pass


class CudaRuntime:
    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcudart.so")
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int

    def check(self, code: int, operation: str) -> None:
        if code == 0:
            return
        raw = self.lib.cudaGetErrorString(code)
        message = raw.decode("utf-8", "replace") if raw else f"CUDA error {code}"
        raise CudaError(f"{operation}: {message} ({code})")

    def malloc(self, size: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        # Empty tensors still receive a non-null address accepted by TensorRT.
        self.check(self.lib.cudaMalloc(ctypes.byref(pointer), max(int(size), 1)), "cudaMalloc")
        return pointer

    def free(self, pointer: ctypes.c_void_p) -> None:
        if pointer and pointer.value:
            self.check(self.lib.cudaFree(pointer), "cudaFree")

    def create_stream(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        return stream

    def destroy_stream(self, stream: ctypes.c_void_p) -> None:
        if stream and stream.value:
            self.check(self.lib.cudaStreamDestroy(stream), "cudaStreamDestroy")

    def copy_h2d(
        self, destination: ctypes.c_void_p, source: np.ndarray, stream: ctypes.c_void_p
    ) -> None:
        if source.nbytes:
            self.check(
                self.lib.cudaMemcpyAsync(
                    destination,
                    ctypes.c_void_p(source.ctypes.data),
                    source.nbytes,
                    self.HOST_TO_DEVICE,
                    stream,
                ),
                "cudaMemcpyAsync H2D",
            )

    def copy_d2h(
        self, destination: np.ndarray, source: ctypes.c_void_p, stream: ctypes.c_void_p
    ) -> None:
        if destination.nbytes:
            self.check(
                self.lib.cudaMemcpyAsync(
                    ctypes.c_void_p(destination.ctypes.data),
                    source,
                    destination.nbytes,
                    self.DEVICE_TO_HOST,
                    stream,
                ),
                "cudaMemcpyAsync D2H",
            )

    def synchronize(self, stream: ctypes.c_void_p) -> None:
        self.check(self.lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")


class TensorRTRunner:
    """Correctness-oriented runner that allocates buffers for each inference."""

    def __init__(self, engine_path: str | Path, log_level: int = trt.Logger.WARNING) -> None:
        self.engine_path = Path(engine_path)
        self.logger = trt.Logger(log_level)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context")
        self.cuda = CudaRuntime()
        self.stream = self.cuda.create_stream()

        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [
            name
            for name in self.names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name
            for name in self.names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]

    def close(self) -> None:
        stream = getattr(self, "stream", None)
        if stream is not None:
            self.cuda.destroy_stream(stream)
            self.stream = None

    def __enter__(self) -> "TensorRTRunner":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _numpy_dtype(self, name: str) -> np.dtype:
        return np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))

    def infer(self, feeds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        missing = sorted(set(self.input_names) - set(feeds))
        extra = sorted(set(feeds) - set(self.input_names))
        if missing or extra:
            raise ValueError(f"Feed mismatch: missing={missing}, extra={extra}")

        host_inputs: dict[str, np.ndarray] = {}
        for name in self.input_names:
            array = np.ascontiguousarray(feeds[name], dtype=self._numpy_dtype(name))
            host_inputs[name] = array
            declared = tuple(self.engine.get_tensor_shape(name))
            if -1 in declared:
                if not self.context.set_input_shape(name, array.shape):
                    raise ValueError(f"TensorRT rejected {name} shape {array.shape}")
            elif declared != array.shape:
                raise ValueError(f"{name}: expected {declared}, received {array.shape}")

        unresolved = self.context.infer_shapes()
        if unresolved:
            raise RuntimeError(f"Insufficiently specified tensors: {list(unresolved)}")

        host_outputs: dict[str, np.ndarray] = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"Data-dependent output shape is unresolved: {name} {shape}")
            host_outputs[name] = np.empty(shape, dtype=self._numpy_dtype(name))

        device_buffers: dict[str, ctypes.c_void_p] = {}
        try:
            for name in self.names:
                host = host_inputs.get(name, host_outputs.get(name))
                if host is None:
                    raise RuntimeError(f"No host buffer for tensor {name}")
                pointer = self.cuda.malloc(host.nbytes)
                device_buffers[name] = pointer
                if not self.context.set_tensor_address(name, int(pointer.value)):
                    raise RuntimeError(f"Could not bind tensor address for {name}")

            for name, host in host_inputs.items():
                self.cuda.copy_h2d(device_buffers[name], host, self.stream)

            if not self.context.execute_async_v3(stream_handle=int(self.stream.value)):
                raise RuntimeError("TensorRT execute_async_v3 returned false")

            for name, host in host_outputs.items():
                self.cuda.copy_d2h(host, device_buffers[name], self.stream)
            self.cuda.synchronize(self.stream)
            return host_outputs
        finally:
            for pointer in device_buffers.values():
                self.cuda.free(pointer)


__all__ = ["CudaError", "CudaRuntime", "TensorRTRunner"]
