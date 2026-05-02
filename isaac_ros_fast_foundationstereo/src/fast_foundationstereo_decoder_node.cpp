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

#include "isaac_ros_fast_foundationstereo/fast_foundationstereo_decoder_node.hpp"
#include "isaac_ros_nitros/types/type_adapter_nitros_context.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_depth
{

FastFoundationStereoDecoderNode::FastFoundationStereoDecoderNode(
  const rclcpp::NodeOptions options)
: rclcpp::Node("fast_foundationstereo_decoder_node", options),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")},
  tensor_nitros_sub_{},
  camera_info_sub_{},
  sync_{ApproxPolicy{10}, tensor_nitros_sub_, camera_info_sub_},
  disparity_pub_{this->create_publisher<stereo_msgs::msg::DisparityImage>(
    "disparity", output_qos_)},
  disparity_tensor_name_{declare_parameter<std::string>(
      "disparity_tensor_name", "disparity")},
  tensor_input_topic_{declare_parameter<std::string>(
      "tensor_input_topic", "tensor_pub")},
  camera_info_topic_{declare_parameter<std::string>(
      "camera_info_topic", "right/camera_info")},
  min_disparity_{declare_parameter<double>("min_disparity", 0.0)},
  max_disparity_{declare_parameter<double>("max_disparity", 10000.0)}
{
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_), "Failed to create CUDA stream");

  RCLCPP_INFO(this->get_logger(), "Subscribing to tensor topic: %s", tensor_input_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "Subscribing to camera_info topic: %s", camera_info_topic_.c_str());

  tensor_nitros_sub_.subscribe(this, tensor_input_topic_);
  camera_info_sub_.subscribe(this, camera_info_topic_);

  sync_.registerCallback(
    std::bind(
      &FastFoundationStereoDecoderNode::SynchronizedCallback, this,
      std::placeholders::_1, std::placeholders::_2));
}

void FastFoundationStereoDecoderNode::SynchronizedCallback(
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg,
  const nvidia::isaac_ros::nitros::NitrosCameraInfo::ConstSharedPtr & camera_info_msg)
{
  RCLCPP_DEBUG(this->get_logger(), "Processing synchronized tensor and camera info pair!");
  ProcessTensorAndCameraInfo(tensor_msg, camera_info_msg);
}

void FastFoundationStereoDecoderNode::ProcessTensorAndCameraInfo(
  const nvidia::isaac_ros::nitros::NitrosTensorList::ConstSharedPtr & tensor_msg,
  const nvidia::isaac_ros::nitros::NitrosCameraInfo::ConstSharedPtr & camera_info_msg)
{
  auto tensor_view = nvidia::isaac_ros::nitros::NitrosTensorListView(*tensor_msg);

  auto tensor = tensor_view.GetNamedTensor(disparity_tensor_name_);
  auto shape = tensor.GetShape().shape();
  int height = shape.dimension(height_dim_);
  int width = shape.dimension(width_dim_);
  size_t tensor_size = tensor.GetTensorSize();

  // Allocate GPU buffer and copy tensor data
  void * gpu_data;
  CHECK_CUDA_ERROR(
    cudaMallocAsync(&gpu_data, tensor_size, stream_),
    "Failed to allocate GPU buffer for disparity tensor");
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      gpu_data, tensor.GetBuffer(),
      tensor_size, cudaMemcpyDefault, stream_),
    "Failed to copy disparity tensor to GPU");

  // Filter disparity map in-place on GPU
  nvidia::isaac_ros::fast_foundationstereo::FilterDisparity(
    static_cast<float *>(gpu_data),
    static_cast<uint32_t>(width), static_cast<uint32_t>(height),
    static_cast<float>(min_disparity_), static_cast<float>(max_disparity_),
    stream_);

  CHECK_CUDA_ERROR(cudaGetLastError(), "CUDA error after FilterDisparity kernel");

  // Copy filtered disparity data back to host
  std::vector<uint8_t> host_data(tensor_size);
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      host_data.data(), gpu_data,
      tensor_size, cudaMemcpyDeviceToHost, stream_),
    "Failed to copy disparity data to host");
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_), "Failed to synchronize CUDA stream");
  CHECK_CUDA_ERROR(cudaFreeAsync(gpu_data, stream_), "Failed to free GPU buffer");

  // Convert NitrosCameraInfo to standard ROS CameraInfo
  sensor_msgs::msg::CameraInfo ros_camera_info;
  try {
    rclcpp::TypeAdapter<nvidia::isaac_ros::nitros::NitrosCameraInfo, sensor_msgs::msg::CameraInfo>
    ::convert_to_ros_message(*camera_info_msg, ros_camera_info);
  } catch (const std::runtime_error & e) {
    RCLCPP_ERROR(this->get_logger(),
    "Failed to convert NitrosCameraInfo to ROS CameraInfo: %s", e.what());
    return;
  }

  // Build DisparityImage message
  stereo_msgs::msg::DisparityImage disp_msg;
  disp_msg.header.stamp.sec = tensor_view.GetTimestampSeconds();
  disp_msg.header.stamp.nanosec = tensor_view.GetTimestampNanoseconds();
  disp_msg.header.frame_id = tensor_view.GetFrameId();

  // Fill the image field
  disp_msg.image.header = disp_msg.header;
  disp_msg.image.height = static_cast<uint32_t>(height);
  disp_msg.image.width = static_cast<uint32_t>(width);
  disp_msg.image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  disp_msg.image.is_bigendian = 0;
  disp_msg.image.step = static_cast<uint32_t>(width) * sizeof(float);
  disp_msg.image.data = std::move(host_data);

  // Fill disparity parameters from camera projection matrix
  disp_msg.f = ros_camera_info.p[0];  // focal_length from projection matrix
  disp_msg.t = -ros_camera_info.p[3] / ros_camera_info.p[0];  // baseline
  disp_msg.min_disparity = static_cast<float>(min_disparity_);
  disp_msg.max_disparity = static_cast<float>(max_disparity_);
  disp_msg.delta_d = 1.0f / 16.0f;  // sub-pixel precision

  // Publish
  disparity_pub_->publish(disp_msg);
}

FastFoundationStereoDecoderNode::~FastFoundationStereoDecoderNode()
{
  cudaError_t err = cudaStreamDestroy(stream_);
  if (err != cudaSuccess) {
    RCLCPP_ERROR(this->get_logger(), "Failed to destroy CUDA stream: %s",
      cudaGetErrorString(err));
  }
}

}  // namespace dnn_stereo_depth
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::dnn_stereo_depth::FastFoundationStereoDecoderNode)
