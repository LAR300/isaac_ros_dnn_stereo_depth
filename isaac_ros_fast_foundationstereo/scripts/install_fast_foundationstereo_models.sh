#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Script to convert Fast FoundationStereo ONNX models to TensorRT engines.
# Pre-exported ONNX models are used from the fast-foundationstereo weights directory.
#
# Usage:
#   bash install_fast_foundationstereo_models.sh [--checkpoint CHECKPOINT] [--resolution RESOLUTION] [--iters ITERS]
#
# Examples:
#   # Default: fastest model (20_30_48, 320x736, 4 iters) — recommended for Jetson
#   bash install_fast_foundationstereo_models.sh
#
#   # Higher quality, slower
#   bash install_fast_foundationstereo_models.sh --checkpoint 23_36_37 --resolution 576x960 --iters 8
#
# Available configurations:
#   Checkpoints: 20_30_48 (fastest), 20_26_39, 23_36_37 (most accurate)
#   Resolutions: 320x736 (recommended for Jetson), 576x960
#   Iterations:  4 (faster), 8 (more accurate)

set -e

# Defaults — optimized for Jetson Orin NX
CHECKPOINT="${CHECKPOINT:-20_30_48}"
RESOLUTION="${RESOLUTION:-320x736}"
ITERS="${ITERS:-4}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --resolution) RESOLUTION="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--checkpoint CHECKPOINT] [--resolution RESOLUTION] [--iters ITERS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint  Model checkpoint (20_30_48|20_26_39|23_36_37) [default: 20_30_48]"
            echo "  --resolution  Input resolution (320x736|576x960) [default: 320x736]"
            echo "  --iters       Refinement iterations (4|8) [default: 4]"
            echo ""
            echo "Recommended for Jetson Orin NX: defaults (20_30_48, 320x736, 4 iters)"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

ISAAC_ROS_WS="${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}"
ONNX_DIR="${ISAAC_ROS_WS}/src/fast-foundationstereo/weights/onnx/${CHECKPOINT}/${RESOLUTION}"
MODEL_NAME="${CHECKPOINT}_iters_${ITERS}_res_${RESOLUTION}"
ONNX_FILE="${ONNX_DIR}/${MODEL_NAME}.onnx"

OUTPUT_DIR="${ISAAC_ROS_WS}/isaac_ros_assets/models/fast_foundationstereo"
ENGINE_FILE="${OUTPUT_DIR}/${MODEL_NAME}.engine"

# Validate ONNX file exists
if [[ ! -f "${ONNX_FILE}" ]]; then
    echo "ERROR: ONNX file not found: ${ONNX_FILE}" >&2
    echo "Available models:"
    find "${ISAAC_ROS_WS}/src/fast-foundationstereo/weights/onnx" -name "*.onnx" 2>/dev/null | sort
    exit 1
fi

# Skip if engine already exists
if [[ -f "${ENGINE_FILE}" ]]; then
    echo "Engine file already exists: ${ENGINE_FILE}"
    echo "Delete it and re-run to rebuild, or use --force_engine_update in the launch file."
    exit 0
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "============================================="
echo "Fast FoundationStereo TensorRT Engine Build"
echo "============================================="
echo "Checkpoint:  ${CHECKPOINT}"
echo "Resolution:  ${RESOLUTION}"
echo "Iterations:  ${ITERS}"
echo "ONNX file:   ${ONNX_FILE}"
echo "Engine file:  ${ENGINE_FILE}"
echo "============================================="

TRTEXEC="${TENSORRT_COMMAND:-/usr/src/tensorrt/bin/trtexec}"

${TRTEXEC} \
    --onnx="${ONNX_FILE}" \
    --saveEngine="${ENGINE_FILE}" \
    --fp16

echo ""
echo "Engine successfully created: ${ENGINE_FILE}"
echo ""
echo "To use with ROS, launch with:"
echo "  ros2 launch isaac_ros_fast_foundationstereo isaac_ros_fast_foundationstereo.launch.py \\"
echo "    engine_file_path:=${ENGINE_FILE} \\"
echo "    model_file_path:=${ONNX_FILE} \\"
echo "    model_input_width:=$(echo ${RESOLUTION} | cut -dx -f2) \\"
echo "    model_input_height:=$(echo ${RESOLUTION} | cut -dx -f1)"
