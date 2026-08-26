// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef ISAAC_ROS_FAST_FOUNDATIONSTEREO__FAST_FOUNDATIONSTEREO_NODE_HPP_
#define ISAAC_ROS_FAST_FOUNDATIONSTEREO__FAST_FOUNDATIONSTEREO_NODE_HPP_

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "stereo_msgs/msg/disparity_image.hpp"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include "isaac_ros_managed_nitros/managed_nitros_message_filters_subscriber.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"

#include "isaac_ros_fast_foundationstereo/filter_disparity.cu.hpp"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime.h"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_depth
{

/// TensorRT logger
class TrtLogger : public nvinfer1::ILogger
{
public:
  explicit TrtLogger(rclcpp::Logger ros_logger)
  : ros_logger_(ros_logger) {}

  void log(Severity severity, const char * msg) noexcept override
  {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        RCLCPP_ERROR(ros_logger_, "[TRT] %s", msg);
        break;
      case Severity::kWARNING:
        RCLCPP_WARN(ros_logger_, "[TRT] %s", msg);
        break;
      case Severity::kINFO:
        RCLCPP_INFO(ros_logger_, "[TRT] %s", msg);
        break;
      default:
        RCLCPP_DEBUG(ros_logger_, "[TRT] %s", msg);
        break;
    }
  }

private:
  rclcpp::Logger ros_logger_;
};

/**
 * @brief Combined node: stereo tensor sync + TensorRT inference + disparity decoding.
 *
 * Subscribes to left/right preprocessed tensors via NITROS message_filters,
 * runs TensorRT inference using the C++ API, post-processes the disparity map,
 * and publishes stereo_msgs::DisparityImage.
 *
 * This approach avoids NITROS negotiation issues between non-NitrosNode
 * publishers and NitrosNode subscribers by doing inference directly.
 */
class FastFoundationStereoNode : public rclcpp::Node
{
public:
  explicit FastFoundationStereoNode(const rclcpp::NodeOptions & options);
  ~FastFoundationStereoNode();

private:
  using NitrosTensorList = nvidia::isaac_ros::nitros::NitrosTensorList;
  using NitrosTensorListView = nvidia::isaac_ros::nitros::NitrosTensorListView;
  using SyncPolicy = ::message_filters::sync_policies::ApproximateTime<
      NitrosTensorList, NitrosTensorList>;

  /// Load or build TensorRT engine
  bool loadEngine();

  /// Stereo pair sync callback
  void stereoCallback(
    const NitrosTensorList::ConstSharedPtr & left_msg,
    const NitrosTensorList::ConstSharedPtr & right_msg);

  /// Camera info callback (cached)
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

  // Parameters
  std::string model_file_path_;
  std::string engine_file_path_;
  std::string left_tensor_name_;
  std::string right_tensor_name_;
  std::string left_tensor_topic_;
  std::string right_tensor_topic_;
  std::string camera_info_topic_;
  double min_disparity_;
  double max_disparity_;
  bool force_engine_update_;
  int model_input_height_;
  int model_input_width_;

  // Phase timing (enable_timing parameter).
  bool enable_timing_{false};
  int timing_report_every_{50};
  int timing_n_{0};
  double timing_enqueue_{0.0};
  double timing_filter_{0.0};
  double timing_copy_{0.0};
  double timing_wait_{0.0};
  double timing_publish_{0.0};

  // NITROS subscribers for stereo tensors
  nvidia::isaac_ros::nitros::message_filters::Subscriber<NitrosTensorListView> left_sub_;
  nvidia::isaac_ros::nitros::message_filters::Subscriber<NitrosTensorListView> right_sub_;
  std::unique_ptr<::message_filters::Synchronizer<SyncPolicy>> sync_;

  // Camera info
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  sensor_msgs::msg::CameraInfo::SharedPtr cached_camera_info_;
  std::mutex camera_info_mutex_;

  // Publisher
  rclcpp::Publisher<stereo_msgs::msg::DisparityImage>::SharedPtr disparity_pub_;

  // TensorRT
  TrtLogger trt_logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  bool engine_loaded_{false};

  // CUDA
  cudaStream_t stream_{nullptr};
  void * d_output_{nullptr};
  size_t output_size_{0};
};

}  // namespace dnn_stereo_depth
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_FAST_FOUNDATIONSTEREO__FAST_FOUNDATIONSTEREO_NODE_HPP_
