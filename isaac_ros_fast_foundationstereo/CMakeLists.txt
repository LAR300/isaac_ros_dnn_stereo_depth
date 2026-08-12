# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_fast_foundationstereo LANGUAGES C CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(CUDAToolkit REQUIRED)
ament_auto_find_build_dependencies()

# NVTX (optional)
option(USE_NVTX "Enable NVTX markers for improved profiling (if available)" ON)
if(USE_NVTX)
    if(TARGET CUDA::nvtx3)
        add_definitions(-DUSE_NVTX)
        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
        link_libraries("CUDA::nvtx3")
    elseif(TARGET CUDA::nvToolsExt)
        add_definitions(-DUSE_NVTX)
        link_libraries("CUDA::nvToolsExt")
    else()
        message(STATUS "NVTX not found, profiling markers disabled")
    endif()
endif()

# FastFoundationStereoDecoderNode
ament_auto_add_library(fast_foundationstereo_decoder_node SHARED
  src/fast_foundationstereo_decoder_node.cpp
  src/filter_disparity.cu.cpp)
rclcpp_components_register_nodes(fast_foundationstereo_decoder_node "nvidia::isaac_ros::dnn_stereo_depth::FastFoundationStereoDecoderNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::dnn_stereo_depth::FastFoundationStereoDecoderNode;$<TARGET_FILE:fast_foundationstereo_decoder_node>\n")
target_include_directories(fast_foundationstereo_decoder_node PUBLIC include)

# TensorPairSyncNode
ament_auto_add_library(tensor_pair_sync_node SHARED
  src/tensor_pair_sync_node.cpp)
rclcpp_components_register_nodes(tensor_pair_sync_node "nvidia::isaac_ros::dnn_stereo_depth::TensorPairSyncNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::dnn_stereo_depth::TensorPairSyncNode;$<TARGET_FILE:tensor_pair_sync_node>\n")
target_include_directories(tensor_pair_sync_node PUBLIC include)

# FastFoundationStereoNode (combined sync + TRT inference + decode)
find_library(NVINFER_LIB nvinfer)
find_library(NVONNXPARSER_LIB nvonnxparser)
ament_auto_add_library(fast_foundationstereo_node SHARED
  src/fast_foundationstereo_node.cpp
  src/filter_disparity.cu.cpp)
rclcpp_components_register_nodes(fast_foundationstereo_node "nvidia::isaac_ros::dnn_stereo_depth::FastFoundationStereoNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::dnn_stereo_depth::FastFoundationStereoNode;$<TARGET_FILE:fast_foundationstereo_node>\n")
target_include_directories(fast_foundationstereo_node PUBLIC include)
target_link_libraries(fast_foundationstereo_node ${NVINFER_LIB} ${NVONNXPARSER_LIB})

# Mark as CUDA files with non-standard extensions
set_source_files_properties(
  src/filter_disparity.cu.cpp
  include/isaac_ros_fast_foundationstereo/filter_disparity.cu.hpp
  PROPERTIES LANGUAGE CUDA
)

# Install visualizer python scripts
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/isaac_ros_fast_foundationstereo_visualizer.py
  scripts/benchmark_fast_foundationstereo.py
  DESTINATION lib/${PROJECT_NAME})

# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE launch)
