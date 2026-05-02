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

#ifndef ISAAC_ROS_FAST_FOUNDATIONSTEREO__FAST_FOUNDATIONSTEREO_DECODER_NODE_HPP_
#define ISAAC_ROS_FAST_FOUNDATIONSTEREO__FAST_FOUNDATIONSTEREO_DECODER_NODE_HPP_

#include <memory>
#include <string>
#include <vector>
#include <limits>
#include <stdexcept>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "stereo_msgs/msg/disparity_image.hpp"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_message_filters_subscriber.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"

#include "isaac_ros_fast_foundationstereo/filter_disparity.cu.hpp"

#include "cuda.h"      // NOLINT
#include "cuda_runtime.h"  // NOLINT

#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(call, msg) \
  do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
      throw std::runtime_error( \
        std::string(msg) + ": " + cudaGetErrorString(_err)); \
    } \
  } while (0)
#endif

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_depth
{

class FastFoundationStereoDecoderNode : public rclcpp::Node
{
public:
  explicit FastFoundationStereoDecoderNode(
    const rclcpp::NodeOptions options = rclcpp::NodeOptions());

  ~FastFoundationStereoDecoderNode();

private:
  void SynchronizedCallback(
    const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg,
    const nvidia::isaac_ros::nitros::NitrosCameraInfo::ConstSharedPtr & camera_info_msg);

  void ProcessTensorAndCameraInfo(
    const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg,
    const nvidia::isaac_ros::nitros::NitrosCameraInfo::ConstSharedPtr & camera_info_msg);

  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;

  nvidia::isaac_ros::nitros::message_filters::Subscriber<
    nvidia::isaac_ros::nitros::NitrosTensorListView> tensor_nitros_sub_;
  ::message_filters::Subscriber<nvidia::isaac_ros::nitros::NitrosCameraInfo> camera_info_sub_;

  using ApproxPolicy = ::message_filters::sync_policies::ApproximateTime<
    nvidia::isaac_ros::nitros::NitrosTensorList,
    nvidia::isaac_ros::nitros::NitrosCameraInfo>;
  ::message_filters::Synchronizer<ApproxPolicy> sync_;

  rclcpp::Publisher<stereo_msgs::msg::DisparityImage>::SharedPtr disparity_pub_;

  std::string disparity_tensor_name_{};
  std::string tensor_input_topic_{};
  std::string camera_info_topic_{};
  double min_disparity_{};
  double max_disparity_{};

  static constexpr int height_dim_{2};
  static constexpr int width_dim_{3};

  cudaStream_t stream_;
};

}  // namespace dnn_stereo_depth
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_FAST_FOUNDATIONSTEREO__FAST_FOUNDATIONSTEREO_DECODER_NODE_HPP_
