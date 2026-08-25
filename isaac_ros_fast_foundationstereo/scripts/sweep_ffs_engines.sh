#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 minirov-petro contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Build and profile a directory of Fast FoundationStereo ONNX models.
#
# For each *.onnx: build a TensorRT engine, sanity-check it, benchmark it, and
# collect the numbers into one comparison table. Run this INSIDE the container
# on the target board — engines are specific to the GPU and TensorRT version.
#
#   ./sweep_ffs_engines.sh --onnx-dir /workspaces/isaac_ros-dev/models/fast_foundationstereo/sweep
#   ./sweep_ffs_engines.sh --onnx-dir DIR --skip-existing      # resume
#   ./sweep_ffs_engines.sh --onnx-dir DIR --dry-run
#
# BUDGET: builds use --builderOptimizationLevel=5, measured at ~70 min for one
# 320x736 shape on an Orin NX 16 GB. Five shapes is most of a working day. The
# run is resumable (--skip-existing) and each engine is validated as soon as it
# is built, so a failure in shape 4 does not cost you shapes 1-3.
#
# Timings are only comparable if the board is pinned: run
#   sudo nvpmodel -m 0 && sudo jetson_clocks
# on the HOST first. build_ffs_engine.sh warns when it is not.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_DIR=""
OUT_DIR=""
SKIP_EXISTING=false
DRY_RUN=false
ITERATIONS=200

usage() {
  cat <<'EOF'
Usage: sweep_ffs_engines.sh --onnx-dir DIR [options]

  --onnx-dir DIR     Directory of *.onnx to build and profile.     (required)
  --out DIR          Where reports go.              (default: <onnx-dir>/report)
  --skip-existing    Skip a config whose .engine already exists (resume).
  --iterations N     Benchmark iterations.                     (default: 200)
  --dry-run          Print what would run, build nothing.
  -h, --help         This message.

Each engine is built next to its ONNX. Per-config artifacts:
  <name>.engine  <name>.timing.cache  <name>.trtexec.log
  <out>/<name>.bench.json  <out>/<name>.sanity.txt
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx-dir)      ONNX_DIR="${2:?--onnx-dir needs a value}"; shift 2 ;;
    --out)           OUT_DIR="${2:?--out needs a value}"; shift 2 ;;
    --skip-existing) SKIP_EXISTING=true; shift ;;
    --iterations)    ITERATIONS="${2:?--iterations needs a value}"; shift 2 ;;
    --dry-run)       DRY_RUN=true; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$ONNX_DIR" ]] || { echo "ERROR: --onnx-dir is required." >&2; usage >&2; exit 2; }
[[ -d "$ONNX_DIR" ]] || { echo "ERROR: not a directory: $ONNX_DIR" >&2; exit 1; }
OUT_DIR="${OUT_DIR:-${ONNX_DIR}/report}"

mapfile -t ONNX_FILES < <(find "$ONNX_DIR" -maxdepth 1 -name '*.onnx' | sort)
[[ ${#ONNX_FILES[@]} -gt 0 ]] || { echo "ERROR: no *.onnx in $ONNX_DIR" >&2; exit 1; }

BUILD="${SCRIPT_DIR}/build_ffs_engine.sh"
SANITY="${SCRIPT_DIR}/sanity_check_ffs_engine.py"
BENCH="${SCRIPT_DIR}/benchmark_fast_foundationstereo.py"
for f in "$BUILD" "$SANITY" "$BENCH"; do
  [[ -e "$f" ]] || { echo "ERROR: missing companion script: $f" >&2; exit 1; }
done

echo "[sweep] ${#ONNX_FILES[@]} models in ${ONNX_DIR}"
echo "[sweep] reports -> ${OUT_DIR}"
if [[ "$DRY_RUN" == true ]]; then
  for onnx in "${ONNX_FILES[@]}"; do echo "  would build: $(basename "$onnx")"; done
  exit 0
fi
mkdir -p "$OUT_DIR"

STARTED_ALL=$(date +%s)
declare -a ROWS=()

for onnx in "${ONNX_FILES[@]}"; do
  name="$(basename "${onnx%.onnx}")"
  engine="${onnx%.onnx}.engine"
  echo
  echo "=============================================================="
  echo "[sweep] ${name}"
  echo "=============================================================="

  build_s=0
  if [[ "$SKIP_EXISTING" == true && -f "$engine" ]]; then
    echo "[sweep] engine exists, skipping build"
  else
    t0=$(date +%s)
    # --quiet drops trtexec's --verbose: 23 MB and 170k lines per engine, which
    # across a sweep is a lot of disk for tactic spam nobody reads. The layer
    # profile below is the part that is actually wanted, and it is unaffected.
    if ! "$BUILD" --onnx "$onnx" --quiet -- --dumpProfile --dumpLayerInfo; then
      echo "[sweep] BUILD FAILED: ${name}"
      ROWS+=("${name}|BUILD-FAIL|-|-|-|-")
      continue
    fi
    build_s=$(( $(date +%s) - t0 ))
  fi

  sanity_out="${OUT_DIR}/${name}.sanity.txt"
  sanity="PASS"
  if ! python3 "$SANITY" --engine "$engine" > "$sanity_out" 2>&1; then
    sanity="FAIL"
    echo "[sweep] sanity check FAILED — see ${sanity_out}"
  fi

  bench_json="${OUT_DIR}/${name}.bench.json"
  if ! python3 "$BENCH" --engine "$engine" --iterations "$ITERATIONS" \
        --json "$bench_json" > "${OUT_DIR}/${name}.bench.txt" 2>&1; then
    echo "[sweep] benchmark FAILED — see ${OUT_DIR}/${name}.bench.txt"
    ROWS+=("${name}|BENCH-FAIL|${sanity}|-|-|${build_s}")
    continue
  fi

  read -r p50 p95 fps mem < <(python3 - "$bench_json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(r['latency_ms']['p50'], r['latency_ms']['p95'],
      r['throughput_fps_wall'], r['device_alloc_MB'])
PY
)
  ROWS+=("${name}|ok|${sanity}|${p50}|${p95}|${fps}|${mem}|${build_s}")
  echo "[sweep] ${name}: p50=${p50} ms  fps=${fps}  sanity=${sanity}  build=${build_s}s"
done

echo
echo "=============================================================="
printf '%-38s %-6s %-6s %9s %9s %8s %9s %8s\n' \
  model status sanity p50_ms p95_ms fps io_MB build_s
printf '%s\n' "--------------------------------------------------------------------------------------------------"
for row in "${ROWS[@]}"; do
  IFS='|' read -r n st sa p50 p95 fps mem bs <<< "$row"
  printf '%-38s %-6s %-6s %9s %9s %8s %9s %8s\n' \
    "$n" "$st" "${sa:--}" "${p50:--}" "${p95:--}" "${fps:--}" "${mem:--}" "${bs:--}"
done
echo
echo "[sweep] total wall time: $(( ($(date +%s) - STARTED_ALL) / 60 )) min"
echo "[sweep] reports in ${OUT_DIR}"
echo "[sweep] layer profiles are in each <name>.trtexec.log (--dumpProfile)"
