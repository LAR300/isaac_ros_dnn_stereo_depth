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

#include "isaac_ros_fast_foundationstereo/fast_foundationstereo_node.hpp"

#include <fstream>
#include <functional>

#include "sensor_msgs/image_encodings.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_depth
{

FastFoundationStereoNode::FastFoundationStereoNode(const rclcpp::NodeOptions & options)
: Node("fast_foundationstereo_node", options),
  trt_logger_(get_logger())
{
  // Declare parameters
  model_file_path_ = declare_parameter<std::string>("model_file_path", "");
  engine_file_path_ = declare_parameter<std::string>("engine_file_path", "");
  left_tensor_name_ = declare_parameter<std::string>("left_tensor_name", "left_image");
  right_tensor_name_ = declare_parameter<std::string>("right_tensor_name", "right_image");
  left_tensor_topic_ = declare_parameter<std::string>("left_tensor_topic", "left/tensor_reshape");
  right_tensor_topic_ = declare_parameter<std::string>(
    "right_tensor_topic", "right/tensor_reshape");
  camera_info_topic_ = declare_parameter<std::string>(
    "camera_info_topic", "right/camera_info_resize");
  min_disparity_ = declare_parameter<double>("min_disparity", 0.0);
  max_disparity_ = declare_parameter<double>("max_disparity", 10000.0);
  force_engine_update_ = declare_parameter<bool>("force_engine_update", false);
  model_input_height_ = declare_parameter<int>("model_input_height", 320);
  model_input_width_ = declare_parameter<int>("model_input_width", 736);

  const auto queue_size = declare_parameter<int>("queue_size", 10);

  // CUDA stream
  auto err = cudaStreamCreate(&stream_);
  if (err != cudaSuccess) {
    throw std::runtime_error(
            std::string("Failed to create CUDA stream: ") + cudaGetErrorString(err));
  }

  // Publisher
  disparity_pub_ = create_publisher<stereo_msgs::msg::DisparityImage>("disparity", 10);

  // Camera info subscriber (standard ROS - cached)
  camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    camera_info_topic_, rclcpp::QoS(10),
    std::bind(&FastFoundationStereoNode::cameraInfoCallback, this, std::placeholders::_1));

  // NITROS tensor subscribers
  left_sub_.subscribe(this, left_tensor_topic_);
  right_sub_.subscribe(this, right_tensor_topic_);

  // Synchronizer
  sync_ = std::make_unique<::message_filters::Synchronizer<SyncPolicy>>(
    SyncPolicy(queue_size), left_sub_, right_sub_);
  sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.1));
  sync_->registerCallback(
    std::bind(
      &FastFoundationStereoNode::stereoCallback, this,
      std::placeholders::_1, std::placeholders::_2));

  RCLCPP_INFO(get_logger(), "FastFoundationStereoNode initialized");
  RCLCPP_INFO(get_logger(), "  Left tensor:  %s", left_tensor_topic_.c_str());
  RCLCPP_INFO(get_logger(), "  Right tensor: %s", right_tensor_topic_.c_str());
  RCLCPP_INFO(get_logger(), "  Camera info:  %s", camera_info_topic_.c_str());
  RCLCPP_INFO(get_logger(), "  Engine:       %s", engine_file_path_.c_str());
  RCLCPP_INFO(get_logger(), "  Model:        %s", model_file_path_.c_str());
}

void FastFoundationStereoNode::cameraInfoCallback(
  const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(camera_info_mutex_);
  cached_camera_info_ = msg;
}

bool FastFoundationStereoNode::loadEngine()
{
  if (engine_loaded_) {
    return true;
  }

  // Check if engine file exists and we don't need to rebuild
  std::ifstream engine_file(engine_file_path_, std::ios::binary);
  bool build_from_onnx = false;

  if (!engine_file.good() || force_engine_update_) {
    if (model_file_path_.empty()) {
      RCLCPP_ERROR(get_logger(), "No engine file and no ONNX model path provided");
      return false;
    }
    build_from_onnx = true;
    RCLCPP_INFO(get_logger(), "Building TensorRT engine from ONNX: %s", model_file_path_.c_str());
  }

  runtime_.reset(nvinfer1::createInferRuntime(trt_logger_));
  if (!runtime_) {
    RCLCPP_ERROR(get_logger(), "Failed to create TensorRT runtime");
    return false;
  }

  if (build_from_onnx) {
    // Build engine from ONNX
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(trt_logger_));
    if (!builder) {
      RCLCPP_ERROR(get_logger(), "Failed to create TensorRT builder");
      return false;
    }

    const auto explicit_batch = 1U << static_cast<uint32_t>(
      nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(explicit_batch));
    if (!network) {
      RCLCPP_ERROR(get_logger(), "Failed to create network definition");
      return false;
    }

    auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, trt_logger_));
    if (!parser->parseFromFile(
        model_file_path_.c_str(),
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
      RCLCPP_ERROR(get_logger(), "Failed to parse ONNX model: %s", model_file_path_.c_str());
      return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);  // 1 GB
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    RCLCPP_INFO(get_logger(), "Building TensorRT engine (this may take several minutes)...");
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
      RCLCPP_ERROR(get_logger(), "Failed to build TensorRT engine");
      return false;
    }

    // Save engine
    std::ofstream out_file(engine_file_path_, std::ios::binary);
    if (out_file.good()) {
      out_file.write(static_cast<const char *>(serialized->data()), serialized->size());
      RCLCPP_INFO(get_logger(), "Saved TensorRT engine to: %s", engine_file_path_.c_str());
    }

    engine_.reset(runtime_->deserializeCudaEngine(serialized->data(), serialized->size()));
  } else {
    // Load existing engine
    engine_file.seekg(0, std::ios::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    engine_file.read(buffer.data(), size);

    RCLCPP_INFO(get_logger(), "Loading TensorRT engine: %s (%zu bytes)",
      engine_file_path_.c_str(), size);
    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
  }

  if (!engine_) {
    RCLCPP_ERROR(get_logger(), "Failed to create TensorRT engine");
    return false;
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    RCLCPP_ERROR(get_logger(), "Failed to create TensorRT execution context");
    return false;
  }

  // Log bindings
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char * name = engine_->getIOTensorName(i);
    auto dims = engine_->getTensorShape(name);
    auto mode = engine_->getTensorIOMode(name);
    std::string shape_str;
    for (int d = 0; d < dims.nbDims; ++d) {
      shape_str += std::to_string(dims.d[d]);
      if (d < dims.nbDims - 1) {shape_str += "x";}
    }
    RCLCPP_INFO(get_logger(), "  Tensor '%s': %s [%s]",
      name, shape_str.c_str(),
      mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT");
  }

  // Allocate output buffer
  auto out_dims = engine_->getTensorShape("disparity");
  output_size_ = 1;
  for (int i = 0; i < out_dims.nbDims; ++i) {
    output_size_ *= out_dims.d[i];
  }
  output_size_ *= sizeof(float);

  auto cuda_err = cudaMalloc(&d_output_, output_size_);
  if (cuda_err != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "Failed to allocate output buffer: %s",
      cudaGetErrorString(cuda_err));
    return false;
  }

  engine_loaded_ = true;
  RCLCPP_INFO(get_logger(), "TensorRT engine loaded successfully");
  return true;
}

void FastFoundationStereoNode::stereoCallback(
  const NitrosTensorList::ConstSharedPtr & left_msg,
  const NitrosTensorList::ConstSharedPtr & right_msg)
{
  // Load engine on first callback
  if (!engine_loaded_ && !loadEngine()) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
      "TensorRT engine not loaded - skipping frame");
    return;
  }

  // Get cached camera info
  sensor_msgs::msg::CameraInfo camera_info;
  {
    std::lock_guard<std::mutex> lock(camera_info_mutex_);
    if (!cached_camera_info_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "No camera_info received yet - skipping frame");
      return;
    }
    camera_info = *cached_camera_info_;
  }

  // Extract tensor views
  auto left_view = nvidia::isaac_ros::nitros::NitrosTensorListView(*left_msg);
  auto right_view = nvidia::isaac_ros::nitros::NitrosTensorListView(*right_msg);

  auto left_tensor = left_view.GetNamedTensor(left_tensor_name_);
  auto right_tensor = right_view.GetNamedTensor(right_tensor_name_);

  // Get GPU pointers from NITROS tensors
  const void * d_left = left_tensor.GetBuffer();
  const void * d_right = right_tensor.GetBuffer();

  // Set TensorRT input/output tensors
  if (!context_->setTensorAddress("left_image", const_cast<void *>(d_left))) {
    RCLCPP_ERROR(get_logger(), "Failed to set left_image tensor address");
    return;
  }
  if (!context_->setTensorAddress("right_image", const_cast<void *>(d_right))) {
    RCLCPP_ERROR(get_logger(), "Failed to set right_image tensor address");
    return;
  }
  if (!context_->setTensorAddress("disparity", d_output_)) {
    RCLCPP_ERROR(get_logger(), "Failed to set disparity tensor address");
    return;
  }

  // Run inference
  if (!context_->enqueueV3(stream_)) {
    RCLCPP_ERROR(get_logger(), "TensorRT inference failed");
    return;
  }

  // Filter disparity on GPU
  int height = model_input_height_;
  int width = model_input_width_;
  nvidia::isaac_ros::fast_foundationstereo::FilterDisparity(
    static_cast<float *>(d_output_),
    static_cast<uint32_t>(width), static_cast<uint32_t>(height),
    static_cast<float>(min_disparity_), static_cast<float>(max_disparity_),
    stream_);

  // Copy result to host
  std::vector<uint8_t> host_data(output_size_);
  auto err = cudaMemcpyAsync(
    host_data.data(), d_output_, output_size_, cudaMemcpyDeviceToHost, stream_);
  if (err != cudaSuccess) {
    RCLCPP_ERROR(get_logger(), "Failed to copy disparity to host: %s", cudaGetErrorString(err));
    return;
  }
  cudaStreamSynchronize(stream_);

  // Build DisparityImage
  stereo_msgs::msg::DisparityImage disp_msg;

  // Use timestamps from the tensor view
  disp_msg.header.stamp.sec = left_view.GetTimestampSeconds();
  disp_msg.header.stamp.nanosec = left_view.GetTimestampNanoseconds();
  disp_msg.header.frame_id = left_view.GetFrameId();

  disp_msg.image.header = disp_msg.header;
  disp_msg.image.height = static_cast<uint32_t>(height);
  disp_msg.image.width = static_cast<uint32_t>(width);
  disp_msg.image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  disp_msg.image.is_bigendian = 0;
  disp_msg.image.step = static_cast<uint32_t>(width) * sizeof(float);
  disp_msg.image.data = std::move(host_data);

  // Camera parameters
  disp_msg.f = camera_info.p[0];                          // focal length
  disp_msg.t = -camera_info.p[3] / camera_info.p[0];     // baseline
  disp_msg.min_disparity = static_cast<float>(min_disparity_);
  disp_msg.max_disparity = static_cast<float>(max_disparity_);
  disp_msg.delta_d = 1.0f / 16.0f;

  disparity_pub_->publish(disp_msg);

  RCLCPP_DEBUG(get_logger(), "Published disparity %dx%d", width, height);
}

FastFoundationStereoNode::~FastFoundationStereoNode()
{
  // Destroy TRT resources before CUDA stream
  context_.reset();
  engine_.reset();
  runtime_.reset();

  if (d_output_) {
    cudaFree(d_output_);
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

}  // namespace dnn_stereo_depth
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::dnn_stereo_depth::FastFoundationStereoNode)
