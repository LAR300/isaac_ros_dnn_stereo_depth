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

#ifndef ISAAC_ROS_FAST_FOUNDATIONSTEREO__TENSOR_PAIR_SYNC_NODE_HPP_
#define ISAAC_ROS_FAST_FOUNDATIONSTEREO__TENSOR_PAIR_SYNC_NODE_HPP_

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include "isaac_ros_managed_nitros/managed_nitros_message_filters_subscriber.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_depth
{

/**
 * @brief Composable node that synchronises two NitrosTensorList streams and
 *        merges them into a single TensorList for downstream consumers
 *        (e.g. TensorRT in a separate container via DDS).
 *
 * Uses NITROS-aware message_filters subscribers so it can receive from NITROS
 * composable nodes within the same container (GXF intra-process path).
 * Publishes standard ROS TensorList (DDS) so NITROS nodes in other containers
 * can receive it via the type-adapter fallback.
 */
class TensorPairSyncNode : public rclcpp::Node
{
public:
  explicit TensorPairSyncNode(const rclcpp::NodeOptions & options);

private:
  using NitrosTensorList = nvidia::isaac_ros::nitros::NitrosTensorList;
  using NitrosTensorListView = nvidia::isaac_ros::nitros::NitrosTensorListView;
  using TensorList = isaac_ros_tensor_list_interfaces::msg::TensorList;

  using SyncPolicy = ::message_filters::sync_policies::ApproximateTime<
      NitrosTensorList, NitrosTensorList>;

  void syncCallback(
    const NitrosTensorList::ConstSharedPtr & msg1,
    const NitrosTensorList::ConstSharedPtr & msg2);

  nvidia::isaac_ros::nitros::message_filters::Subscriber<NitrosTensorListView> sub1_;
  nvidia::isaac_ros::nitros::message_filters::Subscriber<NitrosTensorListView> sub2_;
  std::unique_ptr<::message_filters::Synchronizer<SyncPolicy>> sync_;
  rclcpp::Publisher<TensorList>::SharedPtr pub_;
};

}  // namespace dnn_stereo_depth
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_FAST_FOUNDATIONSTEREO__TENSOR_PAIR_SYNC_NODE_HPP_
