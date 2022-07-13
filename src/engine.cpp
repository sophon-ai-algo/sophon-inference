/* Copyright 2016-2022 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

==============================================================================*/

#include <numeric>
#include <cmath>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
#include "engine.h"
#include "internal.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#ifdef PYTHON
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eval.h>

using namespace pybind11::literals;
#endif



namespace sail {

class Engine::Engine_CC{
public:
  Engine_CC(IOMode mode, int tpu_id);

  Engine_CC(IOMode mode, const Handle& handle);

  Engine_CC(const Engine_CC& other);    

  void copyAssignment(const Engine_CC& other);

  ~Engine_CC(){};

  bool load(const std::string& bmodel_path);

  bool load(const void* bmodel_ptr, size_t bmodel_size);

  std::vector<std::string> get_graph_names();

  std::vector<std::string> get_input_names(const std::string& graph_name);

  std::vector<std::string> get_output_names(const std::string& graph_name);

  std::map<std::string, std::vector<int>> get_max_input_shapes(const std::string& graph_name);

  std::map<std::string, std::vector<int>> get_max_output_shapes(const std::string& graph_name);

  int reshape(const std::string& graph_name, std::map<std::string, std::vector<int>>& input_shapes);

  void* get_input_tensor_sys_mem(const std::string& graph_name, const std::string& tensor_name);
 
  bm_data_type_t get_input_tensor_dtype(const std::string& graph_name, const std::string& tensor_name);

  void* get_output_tensor_sys_mem(const std::string& graph_name, const std::string& tensor_name);

  bm_data_type_t get_output_tensor_dtype(const std::string& graph_name, const std::string& tensor_name);

//only input is numpy arrays
  bool had_alloc_flag_;

//only input is numpy arrays
  void set_io_mode(const std::string& graph_name, IOMode io_mode);

//only input is numpy arrays
  IOMode get_io_mode(std::string& graph_name);

  IOMode get_io_mode2(std::string& graph_name);

// input in system memory with IOMode
  bool input_mode_sys(std::string& graph_name);

// output in system memory with IOMode
  bool output_mode_sys(std::string& graph_name);

  bool get_alloc_flag(const std::string& graph_name);

  void set_alloc_flag(std::string& graph_name, bool alloc_flag);

private:
  friend class Engine;


    /**
   * @brief Initialize bmruntime.
   *
   * @return Program state
   *     @retval O      Success
   *     @retval others Failure
   */
  int init_bmrt();

  /**
   * @brief Judge if the shape of input tensor is valid.
   *
   * @param graph_name   The specified graph name
   * @param input_shapes Specified shapes of all input tensors of the graph
   * @return Program state
   *     @retval O      Success
   *     @retval others Failure
   */
  int is_input_shape_valid(
      const std::string&                             graph_name,
      const std::map<std::string, std::vector<int>>& input_shapes);

  /**
   * @brief Allocate system and device memroy for input and output tensors.
   *
   * @return Program state
   *     @retval O      Success
   *     @retval others Failure
   */
  int alloc_tensors();

  int alloc_tensors_gragp(std::string& graph_name);

  /**
   * @brief Update information of newly loaded models.
   *
   *
   * @param graph_names Graph names of newly loaded models
   * @return Program state
   *     @retval O      Success
   *     @retval others Failure
   */
  int update_status(std::vector<std::string> graph_names);

  /**
   * @brief Free all resources.
   */
  void free();

  /// Indicator of where to store input and output tensors.
  /// SYSI: Input tensors are in system memory while output tensors are
  ///       in device memory.
  /// SYSO: Input tensors are in device memory while output tensors are
  ///       in system memory.
  /// SYSIO: Both input and output tensors are in system memory.
  /// DEVIO: Both input and output tensors are in device memory.
  IOMode io_mode_;

  /// Handle instance.
  Handle handle_;

  /// Pointer to bmruntime instance.
  void* p_bmrt_;

  /// Graph instance for each model.
  std::map<std::string, std::shared_ptr<Graph>> graphs_;

  /// Data types of all input tensors for int8 models.
  std::map<std::string, std::map<std::string, bm_data_type_t>> input_dtypes_;

  /// Data types of all output tensors for int8 models.
  std::map<std::string, std::map<std::string, bm_data_type_t>> output_dtypes_;

  /// Scales of all input tensors for int8 models.
  std::map<std::string, std::map<std::string, float>> input_scales_;

  /// Scales of all output tensors for int8 models.
  std::map<std::string, std::map<std::string, float>> output_scales_;

  /// Shapes of all input tensors.
  std::map<std::string, std::map<std::string, std::vector<int>>> input_shapes_;

  /// Shapes of all output tensors.
  std::map<std::string, std::map<std::string, std::vector<int>>> output_shapes_;

  /// Max shapes of input tensors.
  std::map<std::string, std::map<std::string, std::vector<int>>>
      max_input_shapes_;

  /// Max shapes of output tensors.
  std::map<std::string, std::map<std::string, std::vector<int>>>
      max_output_shapes_;

  std::map<std::string, IOMode> graphs_mode_;

  std::map<std::string, bool> graphs_alloc_;
};

Engine::Engine_CC::Engine_CC(IOMode mode, int tpu_id)
    : io_mode_(mode), handle_(tpu_id), p_bmrt_(nullptr) ,had_alloc_flag_(false) {
}

Engine::Engine_CC::Engine_CC(IOMode mode, const Handle& handle)
    : io_mode_(mode), handle_(handle), p_bmrt_(nullptr) ,had_alloc_flag_(false) {
}

Engine::Engine_CC::Engine_CC(const Engine_CC& other):
    io_mode_          (other.io_mode_),
    handle_           (other.handle_),
    p_bmrt_           (other.p_bmrt_),
    graphs_           (other.graphs_),
    input_dtypes_     (other.input_dtypes_),
    output_dtypes_    (other.output_dtypes_),
    input_scales_     (other.input_scales_),
    output_scales_    (other.output_scales_),
    input_shapes_     (other.input_shapes_),
    output_shapes_    (other.output_shapes_),
    max_input_shapes_ (other.max_input_shapes_),
    max_output_shapes_(other.max_output_shapes_), 
    had_alloc_flag_   (other.had_alloc_flag_){
}  

void Engine::Engine_CC::copyAssignment(const Engine_CC& other){
  if (this != &other) {
    io_mode_           = other.io_mode_;
    handle_            = other.handle_;
    p_bmrt_            = other.p_bmrt_;
    graphs_            = other.graphs_;
    input_dtypes_      = other.input_dtypes_;
    output_dtypes_     = other.output_dtypes_;
    input_scales_      = other.input_scales_;
    output_scales_     = other.output_scales_;
    input_shapes_      = other.input_shapes_;
    output_shapes_     = other.output_shapes_;
    max_input_shapes_  = other.max_input_shapes_;
    max_output_shapes_ = other.max_output_shapes_;
    had_alloc_flag_    = other.had_alloc_flag_;
  }
}

int Engine::Engine_CC::init_bmrt() {
  p_bmrt_ = bmrt_create(handle_.data());
  if (!p_bmrt_) {
    SPDLOG_ERROR("bmrt_create failed.");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  return 0;
}

int Engine::Engine_CC::is_input_shape_valid(
    const std::string&                             graph_name,
    const std::map<std::string, std::vector<int>>& input_shapes) {
  std::vector<std::string> input_names;
  const bm_net_info_t* p_info = bmrt_get_network_info(p_bmrt_,
                                                      graph_name.c_str());
  std::map<std::string, int> input_map;
  for (int i = 0; i < p_info->input_num; ++i) {
    input_names.push_back(p_info->input_names[i]);
    input_map[p_info->input_names[i]] = i;
  }
  for (auto& item : input_shapes) {
    auto tensor_name = item.first;
    auto it = std::find(input_names.begin(), input_names.end(), tensor_name);
    if (it == input_names.end()) {
      SPDLOG_ERROR("Bad input tensor '{}' for {}.", tensor_name, graph_name);
      exit(SAIL_ERR_ENGINE_INNER);
    }
    std::vector<int>& max_shape = max_input_shapes_[graph_name][tensor_name];
    if (item.second.size() != max_shape.size()) {
      SPDLOG_ERROR("Bad dimension of input tensor {} for {}, {}(N) vs {}(Y)",
                    tensor_name, graph_name, item.second.size(),
                    max_shape.size());
      exit(SAIL_ERR_ENGINE_INNER);
    }
    if (p_info->is_dynamic) {
      std::string msg("Invalid value at dim {} for input tensor '{}': {}");
      for (size_t i = 0; i < max_shape.size(); ++i) {
        if (item.second[i] > max_shape[i] || item.second[i] <= 0) {
          SPDLOG_ERROR(msg.c_str(), i, tensor_name, item.second[i]);
          exit(SAIL_ERR_ENGINE_INNER);
        }
      }
    } else {
      int idx = input_map[tensor_name];
      bool flag = false;
      for (int i = 0; i < p_info->stage_num; ++i) {
        bm_shape_t in_shape = p_info->stages[i].input_shapes[idx];
        bool flag_stage = true;
        for (size_t i = 0; i < max_shape.size(); ++i) {
          if (in_shape.dims[i] != item.second[i]) {
            flag_stage = false;
            break;
          }
        }
        if (flag_stage) {
          flag = true;
          break;
        }
      }
      if (!flag) {
        SPDLOG_ERROR("Invalid shape for input tensor '{}': [{}]",
                      tensor_name, fmt::join(item.second, ", "));
        exit(SAIL_ERR_ENGINE_INNER);
      }
    }
  }
  return 0;
}

int Engine::Engine_CC::alloc_tensors() {
  /** For some models, output shapes are not fixed and are comfirmed after
   *  launching. So system memory and device memory for output tensors are
   *  allocated for the maximum shapes.
   */
  auto graph_names = get_graph_names();
  for (auto graph_name : graph_names) {
    int ret = graphs_[graph_name]->alloc_tensors(io_mode_,
                                                 &input_shapes_[graph_name],
                                                 &output_shapes_[graph_name]);
    if (ret) {
      SPDLOG_ERROR("Allocate tensors failed.");
      exit(SAIL_ERR_ENGINE_INNER);
    }
  }
  had_alloc_flag_ = true;
  return 0;
}

int Engine::Engine_CC::alloc_tensors_gragp(std::string& graph_name){
  IOMode io_mode = get_io_mode(graph_name);
  int ret = graphs_[graph_name]->alloc_tensors(io_mode,
                                               &input_shapes_[graph_name],
                                               &output_shapes_[graph_name]);
  if (ret) {
    SPDLOG_ERROR("Allocate tensors failed.");
    exit(SAIL_ERR_ENGINE_INNER);
  }
  set_alloc_flag(graph_name,true);
  return ret;
}


int Engine::Engine_CC::update_status(std::vector<std::string> graph_names) {
  for (auto graph_name : graph_names) {
    const bm_net_info_t* info = bmrt_get_network_info(p_bmrt_, graph_name.c_str());
    std::shared_ptr<Graph> graph(new Graph(handle_.data(), p_bmrt_, graph_name));
    graphs_[graph_name] = graph;
    auto max_shape = get_max_input_shapes(graph_name);
    max_input_shapes_[graph_name] = max_shape;
    max_shape = get_max_output_shapes(graph_name);
    max_output_shapes_[graph_name] = max_shape;
    input_shapes_[graph_name] = max_input_shapes_[graph_name];
    output_shapes_[graph_name] = max_output_shapes_[graph_name];
    for (int i = 0; i < info->input_num; ++i) {
      input_scales_[graph_name][info->input_names[i]] = info->input_scales[i];
      input_dtypes_[graph_name][info->input_names[i]] = info->input_dtypes[i];
    }
    for (int i = 0; i < info->output_num; ++i) {
      output_scales_[graph_name][info->output_names[i]] = info->output_scales[i];
      output_dtypes_[graph_name][info->output_names[i]] = info->output_dtypes[i];
    }
    graphs_[graph_name]->init_dtypes(&input_dtypes_[graph_name],
                                     &output_dtypes_[graph_name]);
  }
  return 0;
}

void Engine::Engine_CC::free() {
  // free graphs, and device memories will be freed by deconstructor of Graph
  for (auto& g : graphs_) {
    g.second.reset();
  }
  // Destroy bmruntime
  bmrt_destroy(p_bmrt_);
}

bool Engine::Engine_CC::load(const std::string& bmodel_path) {
  std::vector<std::string> previous_graph_names = get_graph_names();
  // check bmodel is exist or not
  struct stat buffer;
  if (stat(bmodel_path.c_str(), &buffer) != 0) {
    spdlog::error("bmodel {} does not exist", bmodel_path);
    exit(SAIL_ERR_ENGINE_INIT);
  }
  if (!bmrt_load_bmodel(p_bmrt_, bmodel_path.c_str())) {
    spdlog::error("Load {} failed", bmodel_path);
    exit(SAIL_ERR_ENGINE_INIT);
  }
  std::vector<std::string> current_graph_names = get_graph_names();
  if(current_graph_names.size() == previous_graph_names.size()) {
    return false;
  }
  std::vector<std::string> update_graph_names(
      current_graph_names.begin() + previous_graph_names.size(),
      current_graph_names.end());
  update_status(update_graph_names);
  return true;
}

bool Engine::Engine_CC::load(const void* bmodel_ptr, size_t bmodel_size) {
  // check bmodel is exist or not
  if (bmodel_size <= 0) {
    SPDLOG_ERROR("bmodel path does not exist");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  std::vector<std::string> previous_graph_names = get_graph_names();
  if (!bmrt_load_bmodel_data(p_bmrt_, bmodel_ptr, bmodel_size)) {
    SPDLOG_ERROR("Load bmodel failed");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  std::vector<std::string> current_graph_names = get_graph_names();
  if(current_graph_names.size() == previous_graph_names.size()) {
    return false;
  }
  std::vector<std::string> update_graph_names(
      current_graph_names.begin() + previous_graph_names.size(),
      current_graph_names.end());
  update_status(update_graph_names);
  return true;
}

std::vector<std::string> Engine::Engine_CC::get_graph_names() {
  std::vector<std::string> graph_names;
  int graph_num = bmrt_get_network_number(p_bmrt_);
  const char** names;
  bmrt_get_network_names(p_bmrt_, &names);
  for (int i = 0; i < graph_num; ++i) {
    graph_names.push_back(names[i]);
  }
  std::free(names);
  return std::move(graph_names);
}

std::vector<std::string> Engine::Engine_CC::get_input_names(
    const std::string& graph_name) {
  std::vector<std::string> tensor_names;
  const bm_net_info_t* p_info = bmrt_get_network_info(p_bmrt_,
                                                      graph_name.c_str());
  for (int i = 0; i < p_info->input_num; ++i) {
    tensor_names.push_back(p_info->input_names[i]);
  }
  return tensor_names;
}

std::vector<std::string> Engine::Engine_CC::get_output_names(
    const std::string &graph_name) {
  std::vector<std::string> tensor_names;
  const bm_net_info_t* p_info = bmrt_get_network_info(p_bmrt_,
                                                      graph_name.c_str());
  for (int i = 0; i < p_info->output_num; ++i) {
    tensor_names.push_back(p_info->output_names[i]);
  }
  return tensor_names;
}

std::map<std::string, std::vector<int>> Engine::Engine_CC::get_max_input_shapes(
    const std::string& graph_name) {
  auto it_find = max_input_shapes_.find(graph_name);
  if (it_find != max_input_shapes_.end()) {
    return max_input_shapes_[graph_name];
  } else {
    std::map<std::string, std::vector<int>> input_shapes;
    const bm_net_info_t* p_info = bmrt_get_network_info(p_bmrt_,
                                                        graph_name.c_str());
    std::map<std::string, int> input_map;
    for (int i = 0; i < p_info->input_num; ++i) {
      input_map[p_info->input_names[i]] = i;
    }
    for (auto item : input_map) {
      auto name = item.first;
      auto idx = item.second;
      bm_shape_t init_shape = p_info->stages[0].input_shapes[idx];
      std::vector<int> shape(init_shape.dims,
                             init_shape.dims + init_shape.num_dims);
      for (int i = 1; i < p_info->stage_num; ++i) {
        bm_shape_t in_shape = p_info->stages[i].input_shapes[idx];
        for (int j = 0; j < in_shape.num_dims; ++j) {
          if (in_shape.dims[j] > shape[j]) {
            shape[j] = in_shape.dims[j];
          }
        }
      }
      input_shapes[name] = shape;
    }
    return std::move(input_shapes);
  }
}

std::map<std::string, std::vector<int>>
Engine::Engine_CC::get_max_output_shapes(const std::string& graph_name) {
  auto it_find = max_output_shapes_.find(graph_name);
  if (it_find != max_output_shapes_.end()) {
    return max_output_shapes_[graph_name];
  } else {
    std::map<std::string, std::vector<int>> output_shapes;
    std::vector<std::string> output_names;
    const bm_net_info_t* p_info = bmrt_get_network_info(p_bmrt_,
                                                        graph_name.c_str());
    for (int i = 0; i < p_info->output_num; ++i) {
      output_names.push_back(p_info->output_names[i]);
    }
    std::map<std::string, int> output_map;
    for (int i = 0; i < p_info->output_num; ++i) {
      output_map[p_info->output_names[i]] = i;
    }
    for (auto name : output_names) {
      int idx = output_map[name];
      bm_shape_t init_shape = p_info->stages[0].output_shapes[idx];
      std::vector<int> shape(init_shape.dims,
                             init_shape.dims + init_shape.num_dims);
      for (int i = 1; i < p_info->stage_num; ++i) {
        bm_shape_t out_shape = p_info->stages[i].output_shapes[idx];
        for (int j = 0; j < out_shape.num_dims; ++j) {
          if (out_shape.dims[j] > shape[j]) {
            shape[j] = out_shape.dims[j];
          }
        }
      }
      output_shapes[name] = shape;
    }
    return std::move(output_shapes);
  }
}

int Engine::Engine_CC::reshape(
    const std::string&                       graph_name,
    std::map<std::string, std::vector<int>>& input_shapes) {
  int ret = is_input_shape_valid(graph_name, input_shapes);
  if (ret) {
    SPDLOG_ERROR("Reshape failed.");
    exit(SAIL_ERR_ENGINE_INNER);
  }
  input_shapes_[graph_name] = input_shapes;
  graphs_[graph_name]->reshape();
  return 0;
}

void* Engine::Engine_CC::get_input_tensor_sys_mem(const std::string& graph_name, const std::string& tensor_name){
  return graphs_[graph_name]->get_input_tensor(tensor_name)->sys_data();
}

bm_data_type_t Engine::Engine_CC::get_input_tensor_dtype(const std::string& graph_name, const std::string& tensor_name){
  return input_dtypes_[graph_name][tensor_name];
}

void* Engine::Engine_CC::get_output_tensor_sys_mem(const std::string& graph_name, const std::string& tensor_name){
  return graphs_[graph_name]->get_output_tensor(tensor_name)->sys_data();
}

bm_data_type_t Engine::Engine_CC::get_output_tensor_dtype(const std::string& graph_name, const std::string& tensor_name){
  return output_dtypes_[graph_name][tensor_name]; 
}

void Engine::Engine_CC::set_io_mode(const std::string& graph_name, IOMode io_mode){
  auto iter_temp = graphs_mode_.find(graph_name);
  if(iter_temp == graphs_mode_.end()){
    graphs_mode_.insert(std::pair<std::string,IOMode>(graph_name, io_mode));
  }else{
    auto iter_temp_a = graphs_alloc_.find(graph_name);
    if(iter_temp_a == graphs_alloc_.end()){
      iter_temp->second = io_mode;
    }else if (iter_temp_a->second == true){
      if(iter_temp->second != io_mode){
        SPDLOG_ERROR("Had Alloc but iomode different!");
        exit(SAIL_ERR_ENGINE_INIT);
      }
    }
  }
}

IOMode Engine::Engine_CC::get_io_mode(std::string& graph_name){
  auto iter_temp = graphs_mode_.find(graph_name);
  if(iter_temp == graphs_mode_.end()){
    SPDLOG_ERROR("Error graph name:{}!",graph_name);
    exit(SAIL_ERR_ENGINE_INNER);
  }
  return iter_temp->second;
}

IOMode Engine::Engine_CC::get_io_mode2(std::string& graph_name){
  auto iter_temp = graphs_mode_.find(graph_name);
  if(iter_temp == graphs_mode_.end()){
    return io_mode_;
  }
  return iter_temp->second;
}
// input in system memory with IOMode
bool Engine::Engine_CC::input_mode_sys(std::string& graph_name)
{
  IOMode io_mode_temp = get_io_mode2(graph_name);
  if(io_mode_temp == SYSI || io_mode_temp == SYSIO){
    return true;
  }
  return false;
}

// output in system memory with IOMode
bool Engine::Engine_CC::output_mode_sys(std::string& graph_name)
{
  IOMode io_mode_temp = get_io_mode2(graph_name);
  if(io_mode_temp == SYSO || io_mode_temp == SYSIO){
    return true;
  }
  return false;
}

bool Engine::Engine_CC::get_alloc_flag(const std::string& graph_name)
{
  if (had_alloc_flag_){
    return true;
  }
  auto iter_temp = graphs_alloc_.find(graph_name);
  if (iter_temp == graphs_alloc_.end())
    return false;
  else
    return iter_temp->second;
}

void Engine::Engine_CC::set_alloc_flag(std::string& graph_name, bool alloc_flag)
{
  auto iter_temp = graphs_alloc_.find(graph_name);
  if (iter_temp == graphs_alloc_.end())
    graphs_alloc_.insert(std::pair<std::string,bool>(graph_name,alloc_flag));
  else
    iter_temp->second = alloc_flag;
}


Engine::Engine(int tpu_id)
    :_impl(new Engine_CC(DEVIO, tpu_id)){
  _impl->init_bmrt();
}

Engine::Engine(
    const std::string& bmodel_path,
    int                tpu_id,
    IOMode             mode)
    :_impl(new Engine_CC(mode, tpu_id)){
  if (_impl->init_bmrt()) {
    return;
  }
  if (!_impl->load(bmodel_path)) {
    spdlog::error("Load {} failed!", bmodel_path);
    exit(SAIL_ERR_ENGINE_INIT);
    return;
  }
  if (_impl->alloc_tensors()) {
    return;
  }
}

Engine::Engine(
    const void* bmodel_ptr,
    size_t      bmodel_size,
    int         tpu_id,
    IOMode      mode)
    :_impl(new Engine_CC(mode, tpu_id)){
  if (_impl->init_bmrt()) {
    return;
  }
  if (!_impl->load(bmodel_ptr, bmodel_size)) {
    spdlog::error("Load bmodel failed!");
    exit(SAIL_ERR_ENGINE_INIT);
    return;
  }
  if (_impl->alloc_tensors()) {
    return;
  }
}

Engine::Engine(const Handle& handle)
    :_impl(new Engine_CC(DEVIO, handle)){
  _impl->init_bmrt();
}

Engine::Engine(
    const std::string& bmodel_path,
    const Handle&      handle,
    IOMode             mode)
    :_impl(new Engine_CC(mode, handle)){
  if (_impl->init_bmrt()) {
    return;
  }
  if (!_impl->load(bmodel_path)) {
    spdlog::error("Load {} failed!", bmodel_path);
    exit(SAIL_ERR_ENGINE_INIT);
    return;
  }
  if (_impl->alloc_tensors()) {
    return;
  }
}

Engine::Engine(
    const void*        bmodel_ptr,
    size_t             bmodel_size,
    const Handle&      handle,
    IOMode             mode)
    :_impl(new Engine_CC(mode, handle)){
  if (_impl->init_bmrt()) {
    return;
  }
  if (!_impl->load(bmodel_ptr, bmodel_size)) {
    spdlog::error("Load bmodel failed!");
    exit(SAIL_ERR_ENGINE_INIT);
    return;
  }
  if (_impl->alloc_tensors()) {
    return;
  }
}

// Engine::Engine(const Engine& other)
//     :_impl(new Engine_CC(*other._impl)){
// }

// Engine& Engine::operator=(const Engine& other) {
//   if (this != &other) {
//     _impl->free();
//     _impl->copyAssignment(*other._impl);
//   }
//   return *this;
// }

Engine::~Engine() {
  _impl->free();
  delete _impl;
}

Handle& Engine::get_handle() {
  return _impl->handle_;
}

int Engine::get_device_id() {
  return _impl->handle_.get_device_id();
}

bool Engine::load(const std::string& bmodel_path) {
  return _impl->load(bmodel_path);
}

bool Engine::load(const void* bmodel_ptr, size_t bmodel_size) {
  return _impl->load(bmodel_ptr, bmodel_size);
}

std::vector<std::string> Engine::get_graph_names() {
  return _impl->get_graph_names();
}

void Engine::set_io_mode(const std::string& graph_name, IOMode mode) {
  _impl->graphs_[graph_name]->set_io_mode(mode);
  _impl->set_io_mode(graph_name, mode);
}

std::vector<std::string> Engine::get_input_names(
    const std::string& graph_name) {
  return std::move(_impl->get_input_names(graph_name));
}

std::vector<std::string> Engine::get_output_names(
    const std::string &graph_name) {
  return std::move(_impl->get_output_names(graph_name));
}

std::map<std::string, std::vector<int>> Engine::get_max_input_shapes(
    const std::string& graph_name) {
  return _impl->get_max_input_shapes(graph_name);
}

std::vector<int> Engine::get_input_shape(
    const std::string& graph_name,
    const std::string& tensor_name) {
  return _impl->input_shapes_[graph_name][tensor_name];
}

std::map<std::string, std::vector<int>>
Engine::get_max_output_shapes(const std::string& graph_name) {
  return _impl->get_max_output_shapes(graph_name);
}

std::vector<int> Engine::get_output_shape(
    const std::string& graph_name,
    const std::string& tensor_name) {
  return _impl->output_shapes_[graph_name][tensor_name];
}

bm_data_type_t Engine::get_input_dtype(
    const std::string& graph_name,
    const std::string& tensor_name) {
  return _impl->input_dtypes_[graph_name][tensor_name];
}

bm_data_type_t Engine::get_output_dtype(
    const std::string& graph_name,
    const std::string& tensor_name) {
  return _impl->output_dtypes_[graph_name][tensor_name];
}

float Engine::get_input_scale(
    const std::string& graph_name,
    const std::string& tensor_name) {
  return _impl->input_scales_[graph_name][tensor_name];
}

float Engine::get_output_scale(
    const std::string& graph_name,
    const std::string& tensor_name) {
  return _impl->output_scales_[graph_name][tensor_name];
}

int Engine::reshape(
    const std::string&                       graph_name,
    std::map<std::string, std::vector<int>>& input_shapes) {
  return _impl->reshape(graph_name, input_shapes);
}

Tensor* Engine::get_input_tensor(
    const std::string& graph_name,
    const std::string& tensor_name) {
  return _impl->graphs_[graph_name]->get_input_tensor(tensor_name);
}

Tensor* Engine::get_output_tensor(
    const std::string& graph_name,
    const std::string& tensor_name) {
  return _impl->graphs_[graph_name]->get_output_tensor(tensor_name);
}

void Engine::scale_input_tensor(
    const std::string& graph_name,
    const std::string& tensor_name,
    float*             data) {
  void* tensor_sys_mem = _impl->get_input_tensor_sys_mem(graph_name, tensor_name);
  std::vector<int> shape = get_input_shape(graph_name, tensor_name);
  int size = shape_size(shape);
  bm_data_type_t dtype = _impl->get_input_tensor_dtype(graph_name, tensor_name);
  if (dtype == BM_FLOAT32) {
    float* target = reinterpret_cast<float*>(tensor_sys_mem);
    memcpy(target, data, size * sizeof(float));
  } else if (dtype == BM_INT8) {
    float scale = get_input_scale(graph_name, tensor_name);
    int8_t* target = reinterpret_cast<int8_t*>(tensor_sys_mem);
    scale_fp32_to_int8(data, target, scale, size);
  } else if (dtype == BM_UINT8) {
    float scale = get_input_scale(graph_name, tensor_name);
    uint8_t* target = reinterpret_cast<uint8_t*>(tensor_sys_mem);
    scale_fp32_to_uint8(data, target, scale, size);
  } else if (dtype == BM_INT32) {
    float scale = get_input_scale(graph_name, tensor_name);
    int32_t* target = reinterpret_cast<int32_t*>(tensor_sys_mem);
    scale_fp32_to_int32(data, target, scale, size);
  }else{
      SPDLOG_ERROR("scale_input_tensor() not support!");
  }
}

void Engine::scale_output_tensor(
    const std::string& graph_name,
    const std::string& tensor_name,
    float*             data) {
    if (NULL == data) {
        SPDLOG_ERROR("param err, input parameter:data=null");
        return;
    }

  void* tensor = _impl->get_output_tensor_sys_mem(graph_name,tensor_name);
    if (NULL == tensor) {
        SPDLOG_ERROR("sys_data() is null, must set own_sys_data is True");
        return;
    }

  std::vector<int> shape = get_output_shape(graph_name, tensor_name);
  int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  bm_data_type_t dtype = _impl->get_output_tensor_dtype(graph_name, tensor_name);
  if (dtype == BM_FLOAT32) {
    float* src = reinterpret_cast<float*>(tensor);
    memcpy(data, src, size * sizeof(float));
  } else if (dtype == BM_INT8) {
    float scale = get_output_scale(graph_name, tensor_name);
    int8_t* src = reinterpret_cast<int8_t*>(tensor);
    scale_int8_to_fp32(src, data, scale, size);
  } else if (dtype == BM_UINT8) {
    float scale = get_output_scale(graph_name, tensor_name);
    uint8_t* src = reinterpret_cast<uint8_t*>(tensor);
    scale_uint8_to_fp32(src, data, scale, size);
  } else if (dtype == BM_INT32) {
    float scale = get_output_scale(graph_name, tensor_name);
    int32_t* src = reinterpret_cast<int32_t*>(tensor);
    scale_int32_to_fp32(src, data, scale, size);
  }else{
      SPDLOG_ERROR("scale_output_tensor() not support!");
  }
}

void Engine::scale_fp32_to_int8(float* src, int8_t* dst, float scale, int size) {
    if (NULL == src || NULL == dst) {
        SPDLOG_ERROR("param err, src=0x{}, dst={}", src, dst);
        return;
    }
    double process_start_time_scale = get_current_time_us();
#if USE_ASM_SSE
    AnyScale_SSE(src, BM_FLOAT32, dst, BM_INT8, scale, size);
    PRINT_TIME_MS("AnyScale_SSE", process_start_time_scale)
#else
    AnyScale(src, BM_FLOAT32, dst, BM_INT8, scale, size);
    PRINT_TIME_MS("AnyScale", process_start_time_scale)
#endif
}

void Engine::scale_fp32_to_uint8(float* src, uint8_t* dst, float scale, int size) {
    if (NULL == src || NULL == dst) {
        SPDLOG_ERROR("param err, src={}, dst={}", src, dst);
        return;
    }
    double process_start_time_scale = get_current_time_us();
#if USE_ASM_SSE
    AnyScale_SSE(src, BM_FLOAT32, dst, BM_UINT8, scale, size);
    PRINT_TIME_MS("AnyScale_SSE", process_start_time_scale)
#else
    AnyScale(src, BM_FLOAT32, dst, BM_UINT8, scale, size);
    PRINT_TIME_MS("AnyScale", process_start_time_scale)
#endif
}

void Engine::scale_fp32_to_int32(float* src, int32_t* dst, float scale, int size){
    if (NULL == src || NULL == dst) {
        SPDLOG_ERROR("param err, src={}, dst={}", src, dst);
        return;
    }
    double process_start_time_scale = get_current_time_us();
#if USE_ASM_SSE
    AnyScale_SSE(src, BM_FLOAT32, dst, BM_INT32, scale, size);
    PRINT_TIME_MS("AnyScale_SSE", process_start_time_scale)
#else
    AnyScale(src, BM_FLOAT32, dst, BM_INT32, scale, size);
    PRINT_TIME_MS("AnyScale", process_start_time_scale)
#endif
}

void Engine::scale_int8_to_fp32(int8_t* src, float* dst, float scale, int size) {
    if (NULL == src || NULL == dst) {
        SPDLOG_ERROR("param err, src={}, dst={}", src, dst);
        return;
    }
    double process_start_time_scale = get_current_time_us();
#if USE_ASM_SSE
    AnyScale_SSE(src, BM_INT8, dst, BM_FLOAT32, scale, size);
    PRINT_TIME_MS("AnyScale_SSE", process_start_time_scale)
#else
    AnyScale(src, BM_INT8, dst, BM_FLOAT32, scale, size);
    PRINT_TIME_MS("AnyScale", process_start_time_scale)
#endif
}

void Engine::scale_uint8_to_fp32(uint8_t* src, float* dst, float scale, int size) {
    if (NULL == src || NULL == dst) {
        SPDLOG_ERROR("param err, src={}, dst={}", src, dst);
        return;
    }
    double process_start_time_scale = get_current_time_us();
#if USE_ASM_SSE
    AnyScale_SSE(src, BM_UINT8, dst, BM_FLOAT32, scale, size);
    PRINT_TIME_MS("AnyScale_SSE", process_start_time_scale)
#else
    AnyScale(src, BM_UINT8, dst, BM_FLOAT32, scale, size);
    PRINT_TIME_MS("AnyScale", process_start_time_scale)
#endif
}

void Engine::scale_int32_to_fp32(int32_t* src, float* dst, float scale, int size) {
    if (NULL == src || NULL == dst) {
        SPDLOG_ERROR("param err, src={}, dst={}", src, dst);
        return;
    }
    double process_start_time_scale = get_current_time_us();
#if USE_ASM_SSE
    AnyScale_SSE(src, BM_INT32, dst, BM_FLOAT32, scale, size);
    PRINT_TIME_MS("AnyScale_SSE", process_start_time_scale)
#else
    AnyScale(src, BM_INT32, dst, BM_FLOAT32, scale, size);
    PRINT_TIME_MS("AnyScale", process_start_time_scale)
#endif
}

std::map<std::string, Tensor*> Engine::create_input_tensors_map(const std::string& graph_name, int create_mode)
{
  std::vector<std::string> input_tensor_names = get_input_names(graph_name);
  std::map<std::string,Tensor*> result;
  for(int i=0;i<input_tensor_names.size();++i){
    std::vector<int> input_shape = get_input_shape(graph_name, input_tensor_names[i]);
    bm_data_type_t input_dtype = get_input_dtype(graph_name, input_tensor_names[i]);
    switch (create_mode){
      case 0:
      {
        Tensor* temp_tensor = new Tensor(input_shape,input_dtype);
        result.insert(std::pair<std::string,Tensor*>(input_tensor_names[i],temp_tensor));
      }
      break;
      case 1:
      {
        Tensor* temp_tensor = new Tensor(_impl->handle_,input_shape,input_dtype,false,true);
        result.insert(std::pair<std::string,Tensor*>(input_tensor_names[i],temp_tensor));
      }
      break;
      default:
      {
        bool sys_in = false;
        if (_impl->io_mode_ == SYSI || _impl->io_mode_ == SYSIO){
          sys_in = true;
        }
        Tensor* temp_tensor = new Tensor(_impl->handle_,input_shape,input_dtype,sys_in,true);
        result.insert(std::pair<std::string,Tensor*>(input_tensor_names[i],temp_tensor));
      }
      break;
    }
  }
  return std::move(result);
}

std::map<std::string, Tensor*> Engine::create_output_tensors_map(const std::string& graph_name, int create_mode)
{
  std::vector<std::string> output_tensor_names = get_output_names(graph_name);
  std::map<std::string,Tensor*> result;
  for(int i=0;i<output_tensor_names.size();++i){
    std::vector<int> output_shape = get_output_shape(graph_name, output_tensor_names[i]);
    bm_data_type_t output_dtype = get_output_dtype(graph_name, output_tensor_names[i]);
    switch(create_mode){
      case 0: 
      {
        Tensor* temp_tensor = new Tensor(output_shape,output_dtype);
        result.insert(std::pair<std::string,Tensor*>(output_tensor_names[i],temp_tensor));
      }
      break;
      case 1: 
      {
        Tensor* temp_tensor = new Tensor(_impl->handle_,output_shape,output_dtype,false,true);
        result.insert(std::pair<std::string,Tensor*>(output_tensor_names[i],temp_tensor));
      }
      break;
      default: 
      {
        bool sys_out = false;
        if (_impl->io_mode_ == SYSO || _impl->io_mode_ == SYSIO){
          sys_out = true;
        }
        Tensor* temp_tensor = new Tensor(_impl->handle_,output_shape,output_dtype,sys_out,true);
        result.insert(std::pair<std::string,Tensor*>(output_tensor_names[i],temp_tensor));
      }
      break;
    }
  }
  return std::move(result);
}

void Engine::process(const std::string& graph_name) {
    TRACE_POINT;
  _impl->graphs_[graph_name]->inference();
}

void Engine::process(
    const std::string&                       graph_name,
    std::map<std::string, std::vector<int>>& input_shapes,
    std::map<std::string, void*>&            input_tensors) {
    TRACE_POINT;
  reshape(graph_name, input_shapes);
  for (auto& item : input_tensors) {
    if (_impl->input_dtypes_[graph_name][item.first] == BM_FLOAT32) {
      _impl->graphs_[graph_name]->reset_input_tensor(item.first, item.second);
    }else{
        SPDLOG_ERROR("input_dtype {} is {}, not supported", item.first, _impl->input_dtypes_[graph_name][item.first]);
        exit(EXIT_FAILURE);
    }
  }
  _impl->graphs_[graph_name]->inference();
}

void Engine::process(
    const std::string&                       graph_name,
    std::map<std::string, Tensor*>&          input_tensors,
    std::map<std::string, std::vector<int>>& input_shapes,
    std::map<std::string, Tensor*>&          output_tensors) {
    TRACE_POINT;
  std::string graph_name_temp(graph_name);
  bool input_mode_sys = _impl->input_mode_sys(graph_name_temp);
  for (auto& item : input_tensors) {
    if (item.second->sys_data() == NULL && input_mode_sys){
      SPDLOG_ERROR("IOMode Error!, Not found system memory in input tensor: {}",item.first);
      exit(SAIL_ERR_ENGINE_INNER);
    }
    if (item.second->dev_data().u.device.device_addr == 0 || item.second->dev_data().size <= 0){
      SPDLOG_ERROR("Input Tensor Error!, Not found device memory in input tensor: {}",item.first);
      exit(SAIL_ERR_ENGINE_INNER);
    }
  }
  bool output_mode_sys = _impl->output_mode_sys(graph_name_temp);
  for (auto& item : output_tensors) {
    if (item.second->sys_data() == NULL && output_mode_sys){
      SPDLOG_ERROR("IOMode Error!, Not found system memory in output tensor: {}",item.first);
      exit(SAIL_ERR_ENGINE_INNER);
    }
    if (item.second->dev_data().u.device.device_addr == 0 || item.second->dev_data().size <= 0){
      SPDLOG_ERROR("Output Tensor Error!, Not found device memory in input tensor: {}",item.first);
      exit(SAIL_ERR_ENGINE_INNER);
    }
  }
  _impl->output_shapes_[graph_name] = _impl->graphs_[graph_name]->inference(
      input_tensors, input_shapes, output_tensors);
}

void Engine::process(
    const std::string&              graph_name,
    std::map<std::string, Tensor*>& input_tensors,
    std::map<std::string, Tensor*>& output_tensors) {
    TRACE_POINT;
  std::string graph_name_temp(graph_name);
  bool input_mode_sys = _impl->input_mode_sys(graph_name_temp);
  for (auto& item : input_tensors) {
    if (item.second->sys_data() == NULL && input_mode_sys){
      SPDLOG_ERROR("IOMode Error!, Not found system memory in input tensor: {}",item.first);
      exit(SAIL_ERR_ENGINE_INNER);
    }
    if (item.second->dev_data().u.device.device_addr == 0 || item.second->dev_data().size <= 0){
      SPDLOG_ERROR("Input Tensor Error!, Not found device memory in input tensor: {}",item.first);
      exit(SAIL_ERR_ENGINE_INNER);
    }
  }
  bool output_mode_sys = _impl->output_mode_sys(graph_name_temp);
  for (auto& item : output_tensors) {
    if (item.second->sys_data() == NULL && output_mode_sys){
      SPDLOG_ERROR("IOMode Error!, Not found system memory in output tensor: {}",item.first);
      exit(SAIL_ERR_ENGINE_INNER);
    }
    if (item.second->dev_data().u.device.device_addr == 0 || item.second->dev_data().size <= 0){
      SPDLOG_ERROR("Output Tensor Error!, Not found device memory in output tensor: {}",item.first);
      exit(SAIL_ERR_ENGINE_INNER);
    }
  }
  _impl->output_shapes_[graph_name] = _impl->graphs_[graph_name]->inference(
      input_tensors, output_tensors);
}


#ifdef PYTHON
Engine::Engine(
    pybind11::bytes& bmodel,
    int              bmodel_size,
    int              tpu_id,
    IOMode           mode)
    :_impl(new Engine_CC(mode, tpu_id)){
  char* bmodel_ptr = nullptr;
  ssize_t size;
  if (PYBIND11_BYTES_AS_STRING_AND_SIZE(bmodel.ptr(), &bmodel_ptr, &size)) {
    SPDLOG_ERROR("Unable to extract bytes contents!");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  if (bmodel_size != static_cast<int>(size)) {
    SPDLOG_ERROR("Wrong bmodel_size.");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  if (_impl->init_bmrt()) {
    return;
  }
  if (!_impl->load(bmodel_ptr, bmodel_size)) {
    spdlog::error("Load bmodel failed!");
    exit(SAIL_ERR_ENGINE_INIT);
    return;
  }
  if (_impl->alloc_tensors()) {
    return;
  }
}

Engine::Engine(
    pybind11::bytes& bmodel,
    int              bmodel_size,
    const Handle&    handle,
    IOMode           mode)
    :_impl(new Engine_CC(mode, handle)){
  char* bmodel_ptr = nullptr;
  ssize_t size;
  if (PYBIND11_BYTES_AS_STRING_AND_SIZE(bmodel.ptr(), &bmodel_ptr, &size)) {
    SPDLOG_ERROR("Unable to extract bytes contents!");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  if (bmodel_size != static_cast<int>(size)) {
    SPDLOG_ERROR("Wrong bmodel_size.");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  if (_impl->init_bmrt()) {
    return;
  }
  if (!_impl->load(bmodel_ptr, bmodel_size)) {
    spdlog::error("Load bmodel failed!");
    exit(SAIL_ERR_ENGINE_INIT);
    return;
  }
  if (_impl->alloc_tensors()) {
    return;
  }
}

bool Engine::load(
    pybind11::bytes& bmodel,
    int              bmodel_size) {
  char* bmodel_ptr = nullptr;
  ssize_t size;
  if (PYBIND11_BYTES_AS_STRING_AND_SIZE(bmodel.ptr(), &bmodel_ptr, &size)) {
    SPDLOG_ERROR("Unable to extract bytes contents!");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  if (bmodel_size != static_cast<int>(size)) {
    SPDLOG_ERROR("Wrong bmodel_size.");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  if (!load(bmodel_ptr, bmodel_size)) {
    spdlog::error("Load bmodel failed!");
    exit(SAIL_ERR_ENGINE_INIT);
    return false;
  }
  return true;
}

using namespace pybind11::detail;
std::map<std::string, pybind11::array_t<float>> Engine::process(
    const std::string&                               graph_name,
    std::map<std::string, pybind11::array_t<float>>& input_tensors) {
  std::string graph_name_temp(graph_name);
  bool input_mode_sys = _impl->input_mode_sys(graph_name_temp);
  if (input_mode_sys == false){
    SPDLOG_ERROR("Error IOMode! If input type is numpy.array, IOMode must be SYSIO or SYSI!");
    exit(SAIL_ERR_ENGINE_INNER);
  }

  std::map<std::string, std::vector<int>> shapes;
  std::map<std::string, float*> tensors;
  TRACE_POINT;
  if (_impl->get_alloc_flag(graph_name) == false) {
    std::string g_name_temp(graph_name);
    _impl->alloc_tensors_gragp(g_name_temp);
  }

  pybind11::module np = pybind11::module::import("numpy");  // like 'import numpy as np'

  for (auto& item : input_tensors) {
    std::string name = std::string(pybind11::str(item.first));
    auto buf = item.second.request();
    std::vector<int> shape;
    for (auto it : buf.shape) {
      shape.push_back(static_cast<int>(it));
    }
    shapes[name] = shape;
    tensors[name] = reinterpret_cast<float*>(buf.ptr);
   
    //check if need to move data, support 4-D tensor
    if (!pybind11::detail::check_flags(item.second.ptr(), pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_) && buf.shape.size() == 4) {
      int length=1;
      
      for (auto it: buf.shape){
        length *= int(it);
      }
      
    double process_start_time_asc = get_current_time_us();
#if 0
      std::unique_ptr<float []> tmp(new float[length]);
      for (int i =0; i < buf.shape[0]; i++){
          for (int j=0; j < buf.shape[1]; j++){
            for (int k=0; k< buf.shape[2]; k++){
              for (int m=0; m < buf.shape[3]; m++) {
               
                *(float *)(tmp.get() + i*buf.shape[1]*buf.shape[2]*buf.shape[3] + j*buf.shape[2]*buf.shape[3] + k * buf.shape[3] + m) = \
                  *((float *)buf.ptr + i* buf.strides[0]/sizeof(float) + j*buf.strides[1]/sizeof(float) + k*buf.strides[2]/sizeof(float) + m*buf.strides[3]/sizeof(float));
                  
              }
            }
          }
      }

      memcpy((void *)buf.ptr, (void *)tmp.get(), length*sizeof(float)); 
#else
      pybind11::array_t<float> arr_c = np.attr("ascontiguousarray")(item.second, "dtype"_a="float32");
      memcpy((void *)buf.ptr, (void *)arr_c.request().ptr, length*sizeof(float)); 
#endif
      PRINT_TIME_MS("ascontiguousarray", process_start_time_asc)
      //don't change the strides
      pybind11::detail::array_proxy(item.second.ptr())->flags |=  pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_;
    }

  }
  
  auto start = std::chrono::system_clock::now();
  reshape(graph_name, shapes);
  for (auto& item : tensors) {
    if (_impl->input_dtypes_[graph_name][item.first] == BM_FLOAT32) {
      _impl->graphs_[graph_name]->reset_input_tensor(item.first, item.second);
    } else {
      scale_input_tensor(graph_name, item.first, item.second);
    }
  }
  auto end = std::chrono::system_clock::now();
  auto duration_det =
              std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  _impl->graphs_[graph_name]->inference();
  end = std::chrono::system_clock::now();
  auto duration_rec =
              std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::map<std::string, pybind11::array_t<float>> output;
  shapes = _impl->output_shapes_[graph_name];
  for (auto& s : shapes) {
    if (_impl->output_dtypes_[graph_name][s.first] == BM_INT8
        || _impl->output_dtypes_[graph_name][s.first] == BM_UINT8
        || _impl->output_dtypes_[graph_name][s.first] == BM_INT32) {
      std::vector<ssize_t> shape;
      for (auto v : s.second) {
        shape.push_back(static_cast<ssize_t>(v));
      }
      auto ndarray = pybind11::array_t<float>(shape);
      float* data = ndarray.mutable_data();
      scale_output_tensor(graph_name, s.first, data);
      output[pybind11::str(s.first.c_str())] = ndarray;
    } else {
      pybind11::ssize_t item_size = sizeof(float);
      std::string format = pybind11::format_descriptor<float>::format();
      pybind11::ssize_t ndim = s.second.size();
      std::vector<pybind11::ssize_t> shape;
      for (auto it : s.second) {
          shape.push_back(it);
      }
      std::vector<pybind11::ssize_t> stride;
      for (size_t i = 1; i < s.second.size(); ++i) {
        pybind11::ssize_t inner_stride = std::accumulate(shape.begin() + i,
            shape.end(), sizeof(float), std::multiplies<pybind11::ssize_t>());
        stride.push_back(inner_stride);
      }
      stride.push_back(sizeof(float));
      void* ptr = _impl->get_output_tensor_sys_mem(graph_name, s.first);
      pybind11::buffer_info buf(ptr, item_size, format, ndim, shape, stride);
      output[pybind11::str(s.first.c_str())] = pybind11::array_t<float>(buf);
    }
  }

  end = std::chrono::system_clock::now();
  auto duration_total =
              std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  return std::move(output);
}

void Engine::process(
    const std::string&              graph_name,
    std::map<std::string, Tensor&>& input_tensors,
    std::map<std::string, Tensor&>& output_tensors) {
  std::map<std::string, Tensor*> inputs;
  std::map<std::string, Tensor*> outputs;
  TRACE_POINT;
  for (auto& item : input_tensors) {
    inputs[item.first] = &item.second;
  }
  for (auto& item : output_tensors) {
    outputs[item.first] = &item.second;
  }
  process(graph_name, inputs, outputs);

}

void Engine::process(
    const std::string&                       graph_name,
    std::map<std::string, Tensor&>&          input_tensors,
    std::map<std::string, std::vector<int>>& input_shapes,
    std::map<std::string, Tensor&>&          output_tensors) {
  std::map<std::string, Tensor*> inputs;
  std::map<std::string, Tensor*> outputs;
  TRACE_POINT;
  for (auto& item : input_tensors) {
    inputs[item.first] = &item.second;
  }
  for (auto& item : output_tensors) {
    outputs[item.first] = &item.second;
  }
  process(graph_name, inputs, input_shapes, outputs);
}
#endif

}  // namespace sail
