#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 minirov-petro contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Build a TensorRT engine for a Fast FoundationStereo ONNX model.
#
# Run this INSIDE the container, on the machine that will run the engine — a
# .engine is specific to the GPU architecture AND the TensorRT version that
# built it. Never copy one between machines.
#
#   ./build_ffs_engine.sh --onnx models/fast_foundationstereo/foo.onnx
#   ./build_ffs_engine.sh --onnx foo.onnx --engine bar.engine
#   ./build_ffs_engine.sh --onnx foo.onnx --dry-run
#
# With only --onnx, the engine, timing cache and log are named after it:
#   foo.onnx -> foo.engine, foo.timing.cache, foo.trtexec.log
#
# The first build of a given shape is SLOW because --builderOptimizationLevel=5
# asks the builder for its most expensive tactic search: measured at 70 min for
# 320x736 on an Orin NX 16 GB (TRT 10.3). Budget accordingly when sweeping
# several shapes. The timing cache makes later rebuilds of similar graphs much
# faster, so keep it next to the engine.
#
# ─── Two flags that must NOT be added here ────────────────────────────────
#
# --inputIOFormats / --outputIOFormats fp16:chw
#     These force the engine's I/O bindings (left_image / right_image /
#     disparity) to fp16. FastFoundationStereoNode hardcodes fp32 for every
#     buffer (fast_foundationstereo_node.cpp: `output_size_ *= sizeof(float)`)
#     and the NITROS preprocessing graph feeding it produces fp32 tensors. The
#     mismatch makes FilterDisparity zero everything (NaN / out of range) and
#     depth renders ALL BLACK, with no error anywhere. Plain --fp16 already
#     uses fp16 inside the kernels while keeping the bindings fp32 — which is
#     what NVIDIA's own install_fast_foundationstereo_models.sh does.
#     sanity_check_ffs_engine.py exists to catch exactly this regression.
#
# --skipInference=false
#     In this trtexec the mere PRESENCE of the flag skips the build's
#     sanity-check inference, regardless of the "=false". Leave it out; the
#     default already runs the inference.
#
set -euo pipefail

ONNX=""
ENGINE=""
TIMING_CACHE=""
BUILD_LOG=""
WORKSPACE_MB=4096
DRY_RUN=false
QUIET=false
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: build_ffs_engine.sh --onnx PATH [options]

  --onnx PATH          ONNX model to build from.                    (required)
  --engine PATH        Output engine.       (default: <onnx> with .engine)
  --timing-cache PATH  Builder timing cache.(default: <onnx> with .timing.cache)
  --log PATH           trtexec log.         (default: <onnx> with .trtexec.log)
  --workspace MB       Workspace memory pool in MB.          (default: 4096)
  --quiet              Drop --verbose from trtexec (~23 MB of log per engine).
  --dry-run            Print the trtexec command, build nothing.
  -h, --help           This message.

Anything after `--` is appended verbatim to the trtexec command line.

Read the header of this file before adding precision or IO-format flags:
two of them silently produce all-black depth or skip validation.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx)          ONNX="${2:?--onnx needs a value}"; shift 2 ;;
    --engine)        ENGINE="${2:?--engine needs a value}"; shift 2 ;;
    --timing-cache)  TIMING_CACHE="${2:?--timing-cache needs a value}"; shift 2 ;;
    --log)           BUILD_LOG="${2:?--log needs a value}"; shift 2 ;;
    --workspace)     WORKSPACE_MB="${2:?--workspace needs a value}"; shift 2 ;;
    --quiet)         QUIET=true; shift ;;
    --dry-run)       DRY_RUN=true; shift ;;
    -h|--help)       usage; exit 0 ;;
    --)              shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$ONNX" ]] || { echo "ERROR: --onnx is required." >&2; usage >&2; exit 2; }
[[ -f "$ONNX" ]] || { echo "ERROR: ONNX not found: $ONNX" >&2; exit 1; }

ONNX="$(readlink -f "$ONNX")"
STEM="${ONNX%.onnx}"
ENGINE="${ENGINE:-${STEM}.engine}"
TIMING_CACHE="${TIMING_CACHE:-${STEM}.timing.cache}"
BUILD_LOG="${BUILD_LOG:-${STEM}.trtexec.log}"

# The isaac-ros-dev images ship trtexec but do not put it on PATH, so resolve
# the canonical TensorRT location before giving up.
TRTEXEC="${TRTEXEC:-}"
if [[ -z "$TRTEXEC" ]]; then
  if command -v trtexec >/dev/null 2>&1; then
    TRTEXEC="$(command -v trtexec)"
  elif [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
    TRTEXEC=/usr/src/tensorrt/bin/trtexec
  elif [[ "$DRY_RUN" == true ]]; then
    # --dry-run exists to show the command line, which is useful from a
    # workstation that has no TensorRT at all. Do not make it require one.
    TRTEXEC="trtexec"
  else
    echo "ERROR: trtexec not found on PATH nor at /usr/src/tensorrt/bin/trtexec." >&2
    echo "  Set TRTEXEC=/path/to/trtexec if it lives somewhere else." >&2
    exit 1
  fi
fi

# ── Warn if running as root in a bind-mounted workspace ───────────────
# `docker exec <container>` lands as root, not as the entrypoint's gosu user, so
# the engine / timing cache / log get written root-owned into a directory that
# is bind-mounted from the host — where the host user then cannot delete or
# overwrite them. Enter as the container user instead:
#     docker exec -u admin isaac-ros-orin ...
if [[ "$(id -u)" -eq 0 ]] && [[ -d /workspaces/isaac_ros-dev ]]; then
  echo "WARNING: running as root inside a bind-mounted workspace."
  echo "  Artifacts will be root-owned on the host. Prefer:"
  echo "    docker exec -u admin <container> $(basename "$0") ..."
  echo
fi

# ── Warn if the board is not pinned to max performance ────────────────
# Benchmarks taken while the governor is free to downclock are not comparable
# between runs, which silently poisons any before/after measurement.
if command -v nvpmodel >/dev/null 2>&1; then
  POWER_MODE="$(nvpmodel -q 2>/dev/null | grep -i 'power mode' || true)"
  if [[ -n "$POWER_MODE" && "$POWER_MODE" != *MAXN* ]]; then
    echo "WARNING: ${POWER_MODE}  — not MAXN."
    echo "  Run on the HOST:  sudo nvpmodel -m 0 && sudo jetson_clocks"
    echo "  Timings from a non-pinned board are not comparable across runs."
    echo
  fi
fi

TRTEXEC_ARGS=(
  --onnx="${ONNX}"
  --saveEngine="${ENGINE}"
  --fp16
  --builderOptimizationLevel=5
  --memPoolSize="workspace:${WORKSPACE_MB}"
  --tacticSources=+CUBLAS,+CUBLAS_LT,+CUDNN,+EDGE_MASK_CONVOLUTIONS,+JIT_CONVOLUTIONS
  --useCudaGraph
  --useSpinWait
  --noDataTransfers
  --separateProfileRun
  --avgRuns=50
  --iterations=100
  --warmUp=500
  --timingCacheFile="${TIMING_CACHE}"
)
# --verbose logs every tactic the builder times: ~23 MB and 170k lines for one
# 320x736 engine. Worth it when diagnosing a build, wasteful across a sweep.
[[ "$QUIET" == true ]] || TRTEXEC_ARGS+=(--verbose)
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && TRTEXEC_ARGS+=("${EXTRA_ARGS[@]}")

echo "[build_ffs_engine] trtexec: ${TRTEXEC}"
echo "[build_ffs_engine] ONNX   : ${ONNX}"
echo "[build_ffs_engine] ENGINE : ${ENGINE}"
echo "[build_ffs_engine] CACHE  : ${TIMING_CACHE}"
echo "[build_ffs_engine] LOG    : ${BUILD_LOG}"

if [[ "$DRY_RUN" == true ]]; then
  echo
  echo "${TRTEXEC} ${TRTEXEC_ARGS[*]}"
  exit 0
fi

mkdir -p "$(dirname "${ENGINE}")"
echo "[build_ffs_engine] Building (~70 min for a new shape at 320x736)..."
"${TRTEXEC}" "${TRTEXEC_ARGS[@]}" 2>&1 | tee "${BUILD_LOG}"

[[ -f "${ENGINE}" ]] || {
  echo "ERROR: trtexec finished but no engine at ${ENGINE}. See ${BUILD_LOG}." >&2
  exit 1
}

echo
echo "[build_ffs_engine] Done: ${ENGINE} ($(du -h "${ENGINE}" | cut -f1))"
echo "[build_ffs_engine] Validate it before using it in the pipeline:"
echo "  ros2 run isaac_ros_fast_foundationstereo sanity_check_ffs_engine.py --engine ${ENGINE}"
echo "  ros2 run isaac_ros_fast_foundationstereo benchmark_fast_foundationstereo.py --engine ${ENGINE}"
