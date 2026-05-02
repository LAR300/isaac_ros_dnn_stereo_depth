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

#include "isaac_ros_fast_foundationstereo/tensor_pair_sync_node.hpp"

#include <functional>

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_depth
{

TensorPairSyncNode::TensorPairSyncNode(const rclcpp::NodeOptions & options)
: Node("tensor_pair_sync", options)
{
  // Declare parameters
  const auto tensor1_topic = declare_parameter<std::string>("tensor1_topic", "tensor1");
  const auto tensor2_topic = declare_parameter<std::string>("tensor2_topic", "tensor2");
  const auto output_topic = declare_parameter<std::string>("output_topic", "tensor_pub");
  const auto queue_size = declare_parameter<int>("queue_size", 10);
  const auto max_interval = declare_parameter<double>("max_interval", 0.1);

  // Standard ROS publisher — TensorRT in separate container receives via DDS
  pub_ = create_publisher<TensorList>(output_topic, rclcpp::QoS(10));

  // NITROS-aware message_filters subscribers
  sub1_.subscribe(this, tensor1_topic);
  sub2_.subscribe(this, tensor2_topic);

  // Synchronizer
  sync_ = std::make_unique<::message_filters::Synchronizer<SyncPolicy>>(
    SyncPolicy(queue_size), sub1_, sub2_);
  sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(max_interval));
  sync_->registerCallback(
    std::bind(
      &TensorPairSyncNode::syncCallback, this,
      std::placeholders::_1, std::placeholders::_2));

  RCLCPP_INFO(
    get_logger(),
    "TensorPairSyncNode: subscribing [%s] + [%s] -> publishing [%s]",
    tensor1_topic.c_str(), tensor2_topic.c_str(), output_topic.c_str());
}

void TensorPairSyncNode::syncCallback(
  const NitrosTensorList::ConstSharedPtr & msg1,
  const NitrosTensorList::ConstSharedPtr & msg2)
{
  // Convert NITROS tensor lists to ROS messages
  TensorList ros_msg1, ros_msg2;
  rclcpp::TypeAdapter<NitrosTensorList, TensorList>::convert_to_ros_message(
    *msg1, ros_msg1);
  rclcpp::TypeAdapter<NitrosTensorList, TensorList>::convert_to_ros_message(
    *msg2, ros_msg2);

  // Merge all tensors into a single ROS TensorList
  auto merged = std::make_unique<TensorList>();
  merged->header = ros_msg1.header;
  merged->tensors.reserve(ros_msg1.tensors.size() + ros_msg2.tensors.size());
  merged->tensors.insert(
    merged->tensors.end(), ros_msg1.tensors.begin(), ros_msg1.tensors.end());
  merged->tensors.insert(
    merged->tensors.end(), ros_msg2.tensors.begin(), ros_msg2.tensors.end());

  RCLCPP_DEBUG(
    get_logger(), "Merged %zu + %zu tensors -> %zu",
    ros_msg1.tensors.size(), ros_msg2.tensors.size(), merged->tensors.size());

  pub_->publish(std::move(merged));
}

}  // namespace dnn_stereo_depth
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_stereo_depth::TensorPairSyncNode)
