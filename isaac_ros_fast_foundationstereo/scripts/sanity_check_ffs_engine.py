#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Sanity check for a Fast FoundationStereo TensorRT engine.

Unlike benchmark_fast_foundationstereo.py (which only measures latency with
random data), this script checks whether the engine is actually producing
valid disparity/depth, and whether its I/O bindings are compatible with the
FastFoundationStereoNode used by petro_localization.

Two independent checks:

  A) Engine correctness (dtype-agnostic): feeds synthetic stereo pairs with
     KNOWN disparities (random noise shifted horizontally by each of `--shifts`
     pixels) using whatever dtype each binding actually declares, and verifies
     the decoded disparity recovers each shift. Sweeping several shifts across
     the model's range checks linearity and sub-pixel accuracy, not just that
     *something* came out. This isolates whether the ONNX->TensorRT/FP16
     conversion itself is healthy.

     NOTE — do not add shift=0 back to this sweep. Identical left/right images
     make the d=0 slice of the cost volume a perfect, saturated match at every
     pixel, so the aggregation has no information gradient to work with and
     falls back on its learned prior. A verified-healthy 320x736 engine that
     recovers shifts of 1..128 px to within 0.06 px returns ~45 px of noise for
     shift=0. d=0 means infinite distance and never occurs in this application
     (0.30-3.0 m maps to 323..32 px of disparity at 736 px wide), so the check
     only ever produced false failures — which is worse than no check, because
     a gate that cries wolf gets ignored. The 1 px case covers the low end.

  B) ROS-node compatibility: FastFoundationStereoNode
     (fast_foundationstereo_node.cpp) hardcodes fp32 for all I/O buffers
     (`output_size_ *= sizeof(float)`), and the NITROS preprocessing graph
     feeding it produces fp32 tensors. If the engine's left_image/right_image
     /disparity bindings are anything other than FLOAT32 (e.g. because the
     build script used --inputIOFormats/--outputIOFormats=fp16:chw), the node
     will read/write garbage across that boundary — this is exactly the "all
     black depth" failure mode seen with build_depth_model_ffs_jetson.sh.
     This check fails loudly (exit 1) regardless of check A, since it is a
     pipeline-integration bug that a standalone engine test would otherwise
     hide.

Input geometry (height, width, max_disp) is read from the engine itself and
from the model's sibling .yaml, so the same invocation works for every model in
the resolution/max-disp sweep. Pass the flags explicitly only to override.

Example:
  ros2 run isaac_ros_fast_foundationstereo sanity_check_ffs_engine.py \\
      --engine /workspaces/isaac_ros-dev/models/fast_foundationstereo/\\
20_30_48_iters_4_res_320x736.engine

Requires: python3-tensorrt, torch (both already present in isaac-ros-dev).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import tensorrt as trt
except ImportError as exc:  # pragma: no cover
    sys.exit(f"ERROR: python bindings for TensorRT not available: {exc}")

try:
    import torch
except ImportError as exc:  # pragma: no cover
    sys.exit(f"ERROR: torch not available (required by this script): {exc}")


# ImageNet-style normalization used by ImageNormalizeNode in the
# petro_localization stereo_depth_ffs launch files (mean/stddev in 0-255
# space, matching the ViT-based FoundationStereo backbone).
_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_NORM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

_TRT_TO_NP = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF: np.float16,
    trt.DataType.INT8: np.int8,
    trt.DataType.INT32: np.int32,
    trt.DataType.BOOL: np.bool_,
}
if hasattr(trt.DataType, "UINT8"):
    _TRT_TO_NP[trt.DataType.UINT8] = np.uint8

_TORCH_DTYPE = {
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.int8): torch.int8,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.bool_): torch.bool,
}


def trt_dtype_to_numpy(dt: trt.DataType) -> np.dtype:
    if dt not in _TRT_TO_NP:
        raise RuntimeError(f"Unsupported TRT dtype: {dt}")
    return np.dtype(_TRT_TO_NP[dt])


def infer_geometry(tensors, engine_path: str):
    """Recover (height, width, max_disp) without the caller having to know them.

    Hard-coded 320x736/192 defaults were fine while there was exactly one model;
    with a resolution/max-disp sweep they become a silent trap — the checks below
    would run at the wrong shape and either crash or, worse, pass vacuously.

    height/width come from the engine's own input binding (1,3,H,W). max_disp is
    not in the engine at all, so it comes from the model's sibling .yaml written
    by make_single_onnx.py; None if there is none.
    """
    height = width = None
    for name, mode, _dtype, shape in tensors:
        if mode == trt.TensorIOMode.INPUT and len(shape) == 4:
            height, width = int(shape[2]), int(shape[3])
            break

    max_disp = None
    yaml_path = os.path.splitext(engine_path)[0] + ".yaml"
    if os.path.isfile(yaml_path):
        try:
            import yaml as _yaml
            with open(yaml_path) as fh:
                cfg = _yaml.safe_load(fh) or {}
            if cfg.get("max_disp") is not None:
                max_disp = int(cfg["max_disp"])
            size = cfg.get("image_size")
            if size and (height, width) != (int(size[0]), int(size[1])):
                print(f"[sanity] WARNING: {os.path.basename(yaml_path)} says "
                      f"image_size={list(size)} but the engine binding is "
                      f"{(height, width)}. Trusting the engine.")
        except Exception as exc:  # noqa: BLE001 - advisory only
            print(f"[sanity] WARNING: could not read {yaml_path}: {exc}")

    return height, width, max_disp


def load_engine(engine_path: str, logger: trt.Logger) -> trt.ICudaEngine:
    with open(engine_path, "rb") as fh:
        blob = fh.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(blob)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
    return engine


def make_stereo_pair(height: int, width: int, shift: int, seed: int):
    """Random-noise stereo pair with a known ground-truth disparity.

    left is uniform random RGB noise; right is left shifted left by `shift`
    px (edge-replicated). Local texture is unique per pixel neighborhood, so
    even simple correlation/stereo matching can recover the shift (same idea
    as a random-dot stereogram). Returns (left_u8, right_u8, valid_mask) where
    valid_mask excludes the replicated border column that has no true match.
    """
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    if shift > 0:
        shifted = np.empty_like(left)
        shifted[:, : width - shift] = left[:, shift:]
        shifted[:, width - shift :] = left[:, -1:]  # replicate edge
        right = shifted
        valid_mask = np.ones((height, width), dtype=bool)
        valid_mask[:, width - shift :] = False
    else:
        right = left.copy()
        valid_mask = np.ones((height, width), dtype=bool)
    return left, right, valid_mask


def preprocess(img_u8: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB -> normalized planar CHW float32, matching
    ImageNormalizeNode + InterleavedToPlanarNode in the launch graph."""
    img = img_u8.astype(np.float32)
    img = (img - _NORM_MEAN) / _NORM_STD
    return np.transpose(img, (2, 0, 1))  # HWC -> CHW


class TorchIO:
    """Minimal CUDA buffer helper, dtype taken from each engine binding."""

    def __init__(self):
        self._buffers = {}
        self._shapes = {}
        self.stream = torch.cuda.Stream()

    def alloc(self, name, shape, np_dtype):
        t = torch.empty(tuple(shape), dtype=_TORCH_DTYPE[np.dtype(np_dtype)], device="cuda")
        self._buffers[name] = t
        self._shapes[name] = tuple(shape)
        return int(t.data_ptr())

    def write(self, name, arr):
        buf = self._buffers[name]
        src = torch.from_numpy(np.ascontiguousarray(arr))
        buf.copy_(src.to(buf.device))

    def read(self, name) -> np.ndarray:
        return self._buffers[name].detach().to("cpu").numpy()

    def sync(self):
        self.stream.synchronize()


def run_engine(engine, context, io: TorchIO, tensors, left_chw: np.ndarray, right_chw: np.ndarray):
    for name, mode, dtype, shape in tensors:
        np_dtype = trt_dtype_to_numpy(dtype)
        ptr = io.alloc(name, shape, np_dtype)
        context.set_tensor_address(name, ptr)

    input_names = [t[0] for t in tensors if t[1] == trt.TensorIOMode.INPUT]
    if len(input_names) != 2:
        raise RuntimeError(f"Expected exactly 2 input tensors, got: {input_names}")
    left_name, right_name = (
        (input_names[0], input_names[1])
        if "left" in input_names[0].lower()
        else (input_names[1], input_names[0])
    )

    for name, _, dtype, _shape in tensors:
        if name == left_name:
            io.write(name, left_chw[None].astype(trt_dtype_to_numpy(dtype), copy=False))
        elif name == right_name:
            io.write(name, right_chw[None].astype(trt_dtype_to_numpy(dtype), copy=False))

    context.execute_async_v3(stream_handle=int(io.stream.cuda_stream))
    io.sync()

    out_names = [t[0] for t in tensors if t[1] == trt.TensorIOMode.OUTPUT]
    if len(out_names) != 1:
        raise RuntimeError(f"Expected exactly 1 output tensor, got: {out_names}")
    return io.read(out_names[0]).astype(np.float32).squeeze()


def check_dtype_compat(tensors) -> bool:
    print("[sanity] Engine I/O bindings:")
    ok = True
    for name, mode, dtype, shape in tensors:
        kind = "IN " if mode == trt.TensorIOMode.INPUT else "OUT"
        flag = ""
        if dtype != trt.DataType.FLOAT:
            ok = False
            flag = "  <-- NOT fp32!"
        print(f"    {kind}  {name:<20s}  dtype={str(dtype):<10s}  shape={tuple(shape)}{flag}")

    if not ok:
        print(
            "\n[sanity] FAIL (check B): one or more I/O bindings are not fp32.\n"
            "  FastFoundationStereoNode hardcodes fp32 for every buffer "
            "(fast_foundationstereo_node.cpp: 'output_size_ *= sizeof(float)'),\n"
            "  and the NITROS preprocessing graph feeding it produces fp32 tensors.\n"
            "  A non-fp32 binding here means the ROS node will read/write garbage\n"
            "  across that boundary -> disparity gets zeroed by FilterDisparity ->\n"
            "  depth renders all black. Rebuild the engine WITHOUT "
            "--inputIOFormats/--outputIOFormats overrides (plain --fp16 only)."
        )
    else:
        print("[sanity] OK (check B): all I/O bindings are fp32, compatible with the ROS node.")
    return ok


def check_one_shift(engine, context, tensors, height, width, shift, seed, max_disp, tol):
    """Run one synthetic shift. Returns (ok, stats_dict)."""
    io = TorchIO()
    left_u8, right_u8, valid_mask = make_stereo_pair(height, width, shift, seed)
    disp = run_engine(engine, context, io, tensors,
                      preprocess(left_u8), preprocess(right_u8))

    if disp.shape[-2:] != (height, width):
        print(f"[sanity] WARNING: output shape {disp.shape} != input ({height},{width}); "
              "check resizing before comparing to valid_mask.")

    sane = np.isfinite(disp) & (disp >= 0) & (disp <= max_disp)
    sane_and_valid = sane & valid_mask[: disp.shape[0], : disp.shape[1]]

    frac_sane = float(sane.mean())
    vals = disp[sane_and_valid]
    if vals.size == 0:
        return False, dict(shift=shift, frac_sane=frac_sane, median=float("nan"),
                           std=float("nan"), err=float("nan"),
                           reason="no finite, in-range pixels at all")

    median = float(np.median(vals))
    std = float(vals.std())
    err = float(np.median(np.abs(vals - shift)))
    stats = dict(shift=shift, frac_sane=frac_sane, median=median, std=std, err=err, reason="")

    # An absolute floor plus a relative term: sub-pixel error is what actually
    # distinguishes a good engine from a degraded one, and a purely relative
    # tolerance is meaningless at small shifts.
    limit = max(1.0, tol * shift)
    if frac_sane < 0.5:
        stats["reason"] = (f"only {frac_sane:.1%} of pixels are finite and in range "
                           "(disparity likely zeroed by FilterDisparity -> black depth)")
        return False, stats
    if err > limit:
        stats["reason"] = f"median |error| {err:.2f}px exceeds the {limit:.2f}px budget"
        return False, stats
    return True, stats


def check_disparity_recovery(engine, context, tensors, height, width, shifts,
                             seed, max_disp, tol) -> bool:
    """Sweep several known shifts and report the response curve.

    Linearity across the range is the real signal: an engine damaged by a bad
    precision or IO-format choice does not degrade gracefully, it collapses.
    """
    print(f"\n[sanity] check A — disparity response over {len(shifts)} shifts:")
    print(f"    {'shift':>6} {'median':>9} {'std':>7} {'|err|':>7} {'sane':>7}   ")
    all_ok = True
    for shift in shifts:
        ok, s = check_one_shift(engine, context, tensors, height, width,
                                shift, seed, max_disp, tol)
        flag = "" if ok else f"  <-- FAIL: {s['reason']}"
        print(f"    {s['shift']:>6} {s['median']:>9.2f} {s['std']:>7.2f} "
              f"{s['err']:>7.2f} {s['frac_sane']:>6.1%}{flag}")
        all_ok = all_ok and ok
    print(f"[sanity] {'OK' if all_ok else 'FAIL'} (check A)")
    return all_ok


def default_shifts(max_disp: int):
    """Powers of two spanning the model's usable disparity range.

    Starts at 1, not 0 — see the note in the module docstring on why shift=0 is
    a degenerate input rather than a valid test.
    """
    out, s = [], 1
    while s < max_disp:
        out.append(s)
        s *= 2
    # Always probe near the top of the range, where the cost volume runs out.
    top = (max_disp * 3) // 4
    if top > 1 and top not in out:
        out.append(top)
    return sorted(out)


def main() -> int:
    default_engine = os.environ.get(
        "FAST_FS_ENGINE",
        "/workspaces/isaac_ros-dev/models/fast_foundationstereo/"
        "20_30_48_iters_4_res_320x736.engine",
    )
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", default=default_engine)
    parser.add_argument("--height", type=int, default=None,
                        help="Override; default is read from the engine binding.")
    parser.add_argument("--width", type=int, default=None,
                        help="Override; default is read from the engine binding.")
    parser.add_argument("--shifts", type=str, default=None,
                        help="Comma-separated synthetic disparities in px. "
                             "Default: powers of two spanning max_disp. "
                             "0 is rejected — see the note in the module docstring.")
    parser.add_argument("--max-disp", type=int, default=None,
                        help="Override; default is read from the model's sibling .yaml.")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Relative error budget per shift; the effective "
                             "limit is max(1.0 px, tolerance * shift).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isfile(args.engine):
        print(f"ERROR: engine not found: {args.engine}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False", file=sys.stderr)
        return 2

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    engine = load_engine(args.engine, logger)
    context = engine.create_execution_context()

    tensors = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        tensors.append((name, engine.get_tensor_mode(name),
                         engine.get_tensor_dtype(name), engine.get_tensor_shape(name)))

    det_h, det_w, det_max_disp = infer_geometry(tensors, args.engine)
    height = args.height if args.height is not None else det_h
    width = args.width if args.width is not None else det_w
    max_disp = args.max_disp if args.max_disp is not None else det_max_disp

    if height is None or width is None:
        print("ERROR: could not determine input height/width from the engine; "
              "pass --height/--width explicitly.", file=sys.stderr)
        return 2
    if max_disp is None:
        # Only used as an upper bound when counting sane pixels, so a permissive
        # fallback is safe — but say so, since a wrong bound weakens check A.
        max_disp = width
        print(f"[sanity] NOTE: no max_disp found (expected a .yaml next to the "
              f"engine); using the image width ({width}) as the sanity bound.")

    print(f"[sanity] Geometry: {height}x{width}, max_disp={max_disp}")

    if args.shifts:
        try:
            shifts = [int(s) for s in args.shifts.split(",") if s.strip()]
        except ValueError:
            print(f"ERROR: could not parse --shifts '{args.shifts}'", file=sys.stderr)
            return 2
    else:
        shifts = default_shifts(max_disp)

    if any(s <= 0 for s in shifts):
        print("ERROR: shifts must be >= 1. Zero disparity is a degenerate input for "
              "this model, not a valid test — see the module docstring.", file=sys.stderr)
        return 2
    if any(s >= max_disp for s in shifts):
        bad = [s for s in shifts if s >= max_disp]
        print(f"ERROR: shifts {bad} are >= max_disp ({max_disp}); the model cannot "
              "represent those disparities.", file=sys.stderr)
        return 2

    dtype_ok = check_dtype_compat(tensors)
    shift_ok = check_disparity_recovery(
        engine, context, tensors, height, width,
        shifts=shifts, seed=args.seed, max_disp=max_disp, tol=args.tolerance)

    print("\n================= Fast FoundationStereo sanity check =================")
    print(f" Engine               : {args.engine}")
    print(f" Geometry             : {height}x{width}, max_disp={max_disp}")
    print(f" B) fp32 I/O bindings : {'PASS' if dtype_ok else 'FAIL'}")
    print(f" A) disparity sweep   : {'PASS' if shift_ok else 'FAIL'}  "
          f"({len(shifts)} shifts: {', '.join(map(str, shifts))})")
    overall = dtype_ok and shift_ok
    print(f" OVERALL              : {'PASS' if overall else 'FAIL'}")
    print("========================================================================\n")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
