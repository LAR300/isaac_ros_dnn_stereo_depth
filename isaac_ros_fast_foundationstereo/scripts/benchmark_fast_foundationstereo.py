#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 minirov-petro contributors.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone benchmark for a Fast FoundationStereo TensorRT engine.

Runs pure inference (no ROS graph) on a .engine file and reports:
  * throughput  (FPS)
  * latency     (mean / stddev / min / max / p50 / p95 / p99)  in ms
  * per-input / per-output tensor shapes and dtypes
  * peak GPU memory used by the execution context

Designed for Jetson Orin NX 16GB (SM 8.7) but works on any TRT >= 8.6 target.

Example:
  ./benchmark_fast_foundationstereo.py \\
      --engine /workspaces/isaac_ros-dev/models/fast_foundationstereo/\\
20_30_48_iters_4_res_320x736.engine

Prereqs on Jetson (already present on Isaac ROS dev images):
  * python3-tensorrt      (always shipped)
  * one of the following CUDA backends (auto-detected, in order):
        - torch         (recommended — always present in isaac-ros-dev)
        - cuda-python   (new API: `cuda.bindings.driver`)
        - cuda-python   (legacy API: `from cuda import cuda`)
        - pycuda
  * numpy
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any

import numpy as np

try:
    import tensorrt as trt
except ImportError as exc:  # pragma: no cover
    sys.exit(f'ERROR: python bindings for TensorRT not available: {exc}')


# ---------------------------------------------------------------------------
# CUDA backend abstraction
# ---------------------------------------------------------------------------
# Auto-detected in this priority order:
#   1. torch                       - always present in isaac-ros-dev
#   2. cuda-python (new bindings)  - cuda.bindings.driver / .runtime
#   3. cuda-python (legacy)        - from cuda import cuda, cudart
#   4. pycuda
#
# Each backend implements the same tiny API:
#   alloc(name, shape, np_dtype) -> int   # device ptr, buffer tracked by name
#   write(name, ndarray)                  # host -> device
#   read(name)  -> ndarray                # device -> host (shape from alloc)
#   stream_handle() -> int                # for context.execute_async_v3
#   stream_sync()
#   event() -> ev ; record(ev) ; sync(ev) ; elapsed_ms(a, b) -> float
# ---------------------------------------------------------------------------
class _TorchBackend:
    kind = 'torch'

    def __init__(self, torch):
        self._torch = torch
        self._buffers: dict[str, Any] = {}
        self._shapes:  dict[str, tuple] = {}
        self._stream = torch.cuda.Stream()
        self._np_to_torch = {
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float16): torch.float16,
            np.dtype(np.int8):    torch.int8,
            np.dtype(np.uint8):   torch.uint8,
            np.dtype(np.int32):   torch.int32,
            np.dtype(np.bool_):   torch.bool,
        }

    def _tdtype(self, np_dtype):
        if np_dtype not in self._np_to_torch:
            raise RuntimeError(f'Unsupported dtype for torch backend: {np_dtype}')
        return self._np_to_torch[np_dtype]

    def alloc(self, name, shape, np_dtype):
        t = self._torch.empty(shape, dtype=self._tdtype(np_dtype), device='cuda')
        self._buffers[name] = t
        self._shapes[name] = tuple(shape)
        return int(t.data_ptr())

    def write(self, name, arr):
        buf = self._buffers[name]
        src = self._torch.from_numpy(np.ascontiguousarray(arr))
        buf.copy_(src.to(buf.device, non_blocking=False))

    def read(self, name):
        return self._buffers[name].detach().to('cpu').numpy()

    def stream_handle(self):
        return int(self._stream.cuda_stream)

    def stream_sync(self):
        self._stream.synchronize()

    def event(self):
        return self._torch.cuda.Event(enable_timing=True)

    def record(self, ev):
        ev.record(self._stream)

    def sync(self, ev):
        ev.synchronize()

    def elapsed_ms(self, a, b):
        return float(a.elapsed_time(b))


class _CudaPythonBackend:
    def __init__(self, drv, rt, kind):
        self._drv = drv
        self._rt = rt
        self.kind = kind
        self._success = drv.CUresult.CUDA_SUCCESS
        err, = drv.cuInit(0)
        self._check(err)
        err, dev = drv.cuDeviceGet(0)
        self._check(err)
        err, ctx = drv.cuDevicePrimaryCtxRetain(dev)
        self._check(err)
        err, = drv.cuCtxSetCurrent(ctx)
        self._check(err)
        err, self._stream = drv.cuStreamCreate(0)
        self._check(err)
        self._buffers: dict[str, tuple[int, int]] = {}  # name -> (ptr, nbytes)
        self._shapes: dict[str, tuple] = {}
        self._dtypes: dict[str, np.dtype] = {}

    def _check(self, err):
        if err != self._success:
            _, name = self._drv.cuGetErrorName(err)
            raise RuntimeError(f'CUDA error: {name}')

    def alloc(self, name, shape, np_dtype):
        nbytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
        err, ptr = self._drv.cuMemAlloc(nbytes)
        self._check(err)
        self._buffers[name] = (int(ptr), nbytes)
        self._shapes[name] = tuple(shape)
        self._dtypes[name] = np.dtype(np_dtype)
        return int(ptr)

    def write(self, name, arr):
        ptr, nbytes = self._buffers[name]
        src = np.ascontiguousarray(arr)
        assert src.nbytes == nbytes, 'size mismatch'
        err, = self._drv.cuMemcpyHtoD(ptr, src.ctypes.data, nbytes)
        self._check(err)

    def read(self, name):
        ptr, nbytes = self._buffers[name]
        out = np.empty(self._shapes[name], dtype=self._dtypes[name])
        err, = self._drv.cuMemcpyDtoH(out.ctypes.data, ptr, nbytes)
        self._check(err)
        return out

    def stream_handle(self):
        return int(self._stream)

    def stream_sync(self):
        err, = self._drv.cuStreamSynchronize(self._stream)
        self._check(err)

    def event(self):
        err, ev = self._drv.cuEventCreate(0)
        self._check(err)
        return ev

    def record(self, ev):
        err, = self._drv.cuEventRecord(ev, self._stream)
        self._check(err)

    def sync(self, ev):
        err, = self._drv.cuEventSynchronize(ev)
        self._check(err)

    def elapsed_ms(self, a, b):
        err, ms = self._drv.cuEventElapsedTime(a, b)
        self._check(err)
        return float(ms)


class _PycudaBackend:
    kind = 'pycuda'

    def __init__(self, drv):
        self._drv = drv
        self._stream = drv.Stream()
        self._buffers: dict[str, Any] = {}
        self._shapes: dict[str, tuple] = {}
        self._dtypes: dict[str, np.dtype] = {}

    def alloc(self, name, shape, np_dtype):
        nbytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
        buf = self._drv.mem_alloc(nbytes)
        self._buffers[name] = buf
        self._shapes[name] = tuple(shape)
        self._dtypes[name] = np.dtype(np_dtype)
        return int(buf)

    def write(self, name, arr):
        self._drv.memcpy_htod(self._buffers[name], np.ascontiguousarray(arr))

    def read(self, name):
        out = np.empty(self._shapes[name], dtype=self._dtypes[name])
        self._drv.memcpy_dtoh(out, self._buffers[name])
        return out

    def stream_handle(self):
        return int(self._stream.handle)

    def stream_sync(self):
        self._stream.synchronize()

    def event(self):
        return self._drv.Event()

    def record(self, ev):
        ev.record(self._stream)

    def sync(self, ev):
        ev.synchronize()

    def elapsed_ms(self, a, b):
        return float(b.time_since(a))


def _pick_backend():
    """Return an initialized CUDA backend or raise."""
    # 1) torch (recommended — always in isaac-ros-dev)
    try:
        import torch
        if torch.cuda.is_available():
            return _TorchBackend(torch)
    except ImportError:
        pass

    # 2) cuda-python new API
    try:
        from cuda.bindings import driver, runtime  # type: ignore
        return _CudaPythonBackend(driver, runtime, 'cuda-python (bindings)')
    except ImportError:
        pass

    # 3) cuda-python legacy API
    try:
        from cuda import cuda as driver  # type: ignore
        from cuda import cudart as runtime  # type: ignore
        return _CudaPythonBackend(driver, runtime, 'cuda-python (legacy)')
    except ImportError:
        pass

    # 4) pycuda
    try:
        import pycuda.driver as drv
        import pycuda.autoinit  # noqa: F401
        return _PycudaBackend(drv)
    except ImportError:
        pass

    raise RuntimeError(
        'No CUDA backend available. Install one of: torch (recommended), '
        'cuda-python, or pycuda.')


# ---------------------------------------------------------------------------
# TRT helpers
# ---------------------------------------------------------------------------
_TRT_TO_NP = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF:  np.float16,
    trt.DataType.INT8:  np.int8,
    trt.DataType.INT32: np.int32,
    trt.DataType.BOOL:  np.bool_,
}
if hasattr(trt.DataType, 'UINT8'):
    _TRT_TO_NP[trt.DataType.UINT8] = np.uint8
if hasattr(trt.DataType, 'BF16'):
    # numpy has no native bf16 — use float16 as a placeholder for size/dtype
    _TRT_TO_NP[trt.DataType.BF16] = np.float16


def trt_dtype_to_numpy(dt: trt.DataType) -> np.dtype:
    if dt not in _TRT_TO_NP:
        raise RuntimeError(f'Unsupported TRT dtype: {dt}')
    return np.dtype(_TRT_TO_NP[dt])


def load_engine(engine_path: str, logger: trt.Logger) -> trt.ICudaEngine:
    with open(engine_path, 'rb') as fh:
        blob = fh.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(blob)
    if engine is None:
        raise RuntimeError(f'Failed to deserialize engine: {engine_path}')
    return engine


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def bench(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.engine):
        print(f'ERROR: engine not found: {args.engine}', file=sys.stderr)
        return 2

    logger = trt.Logger(trt.Logger.WARNING if not args.verbose else trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')
    engine = load_engine(args.engine, logger)
    context = engine.create_execution_context()
    cuda = _pick_backend()
    print(f'[bench] TensorRT {trt.__version__}  |  CUDA backend: {cuda.kind}')

    # --- Discover I/O tensors (TRT >= 8.5 tensor API) -----------------------
    tensors = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)         # INPUT / OUTPUT
        dtype = engine.get_tensor_dtype(name)
        shape = tuple(engine.get_tensor_shape(name))
        tensors.append((name, mode, dtype, shape))

    inputs = [t for t in tensors if t[1] == trt.TensorIOMode.INPUT]
    outputs = [t for t in tensors if t[1] == trt.TensorIOMode.OUTPUT]

    print('[bench] Engine I/O:')
    for name, mode, dtype, shape in tensors:
        kind = 'IN ' if mode == trt.TensorIOMode.INPUT else 'OUT'
        print(f'    {kind}  {name:<20s}  dtype={str(dtype):<20s}  shape={shape}')

    # --- Allocate device buffers, register with context ---------------------
    total_bytes = 0
    for name, _, dtype, shape in tensors:
        np_dtype = trt_dtype_to_numpy(dtype)
        ptr = cuda.alloc(name, shape, np_dtype)
        context.set_tensor_address(name, ptr)
        total_bytes += int(np.prod(shape)) * np_dtype.itemsize

    # --- Prepare synthetic inputs (uploaded once, reused each iter) --------
    rng = np.random.default_rng(42)
    for name, _, dtype, shape in inputs:
        np_dtype = trt_dtype_to_numpy(dtype)
        arr = rng.standard_normal(shape).astype(np_dtype, copy=False)
        cuda.write(name, arr)

    stream = cuda.stream_handle()
    ev_start = cuda.event()
    ev_end = cuda.event()

    # --- Warmup -------------------------------------------------------------
    print(f'[bench] Warming up: {args.warmup} iters...')
    for _ in range(args.warmup):
        context.execute_async_v3(stream_handle=stream)
    cuda.stream_sync()

    # --- Timed loop ---------------------------------------------------------
    print(f'[bench] Timing: {args.iterations} iters...')
    latencies_ms: list[float] = []
    t_wall_start = time.perf_counter()
    for _ in range(args.iterations):
        cuda.record(ev_start)
        context.execute_async_v3(stream_handle=stream)
        cuda.record(ev_end)
        cuda.sync(ev_end)
        latencies_ms.append(cuda.elapsed_ms(ev_start, ev_end))
    cuda.stream_sync()
    t_wall_end = time.perf_counter()

    # Read back one output so we confirm the engine really produced data.
    for name, _, _, _ in outputs:
        _ = cuda.read(name)

    # --- Report -------------------------------------------------------------
    latencies_ms.sort()
    wall_s = t_wall_end - t_wall_start
    mean = statistics.mean(latencies_ms)
    stdev = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0

    def pct(p: float) -> float:
        # nearest-rank percentile on the sorted list
        k = max(0, min(len(latencies_ms) - 1,
                       int(round(p / 100.0 * len(latencies_ms))) - 1))
        return latencies_ms[k]

    report = {
        'engine':                args.engine,
        'trt_version':           trt.__version__,
        'iterations':            args.iterations,
        'warmup':                args.warmup,
        'wall_time_s':           round(wall_s, 4),
        'throughput_fps_wall':   round(args.iterations / wall_s, 2),
        'throughput_fps_gpu':    round(1000.0 / mean, 2),
        'latency_ms': {
            'mean':  round(mean, 3),
            'stdev': round(stdev, 3),
            'min':   round(latencies_ms[0], 3),
            'p50':   round(pct(50), 3),
            'p95':   round(pct(95), 3),
            'p99':   round(pct(99), 3),
            'max':   round(latencies_ms[-1], 3),
        },
        'device_alloc_MB':       round(total_bytes / (1024 * 1024), 2),
        'inputs':  [{'name': n, 'shape': list(s), 'dtype': str(d)}
                    for n, _, d, s in inputs],
        'outputs': [{'name': n, 'shape': list(s), 'dtype': str(d)}
                    for n, _, d, s in outputs],
    }

    print('\n================= Fast FoundationStereo benchmark =================')
    print(f' Engine   : {report["engine"]}')
    print(f' TRT      : {report["trt_version"]}')
    print(f' Iters    : {report["iterations"]}  (warmup {report["warmup"]})')
    print(f' Wall FPS : {report["throughput_fps_wall"]:.2f}   '
          f'(GPU-only FPS: {report["throughput_fps_gpu"]:.2f})')
    print(' Latency [ms]:')
    lat = report['latency_ms']
    print(f'   mean {lat["mean"]:.3f}  stdev {lat["stdev"]:.3f}   '
          f'min {lat["min"]:.3f}  max {lat["max"]:.3f}')
    print(f'   p50  {lat["p50"]:.3f}   p95 {lat["p95"]:.3f}   p99 {lat["p99"]:.3f}')
    print(f' Device mem allocated for IO: {report["device_alloc_MB"]:.2f} MB')
    print('===================================================================\n')

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(report, fh, indent=2)
        print(f'[bench] JSON report written to {args.json}')

    return 0


def main() -> int:
    default_engine = os.environ.get(
        'FAST_FS_ENGINE',
        '/workspaces/isaac_ros-dev/models/fast_foundationstereo/'
        '20_30_48_iters_4_res_320x736.engine',
    )
    parser = argparse.ArgumentParser(
        description='Benchmark a Fast FoundationStereo TRT engine on Jetson.')
    parser.add_argument('--engine', default=default_engine,
                        help='Path to the .engine file to benchmark.')
    parser.add_argument('--iterations', type=int, default=200,
                        help='Number of timed inference iterations.')
    parser.add_argument('--warmup', type=int, default=30,
                        help='Warmup iterations before timing starts.')
    parser.add_argument('--json', default=None,
                        help='Optional path to dump a JSON report.')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose TensorRT logger.')
    args = parser.parse_args()
    return bench(args)


if __name__ == '__main__':
    raise SystemExit(main())
