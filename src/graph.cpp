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
#include <type_traits>
#include "spdlog/spdlog.h"
#include "graph.h"

namespace sail {

Graph::Graph(
    bm_handle_t        handle,
    void*              p_bmrt,
    const std::string& name)
    : handle_(handle), p_bmrt_(p_bmrt), input_dtypes_(nullptr),
      output_dtypes_(nullptr), name_(name), io_mode_(DEVIO),
      inputs_(nullptr), outputs_(nullptr),
      input_shapes_(nullptr), output_shapes_(nullptr) {
  if (!p_bmrt) {
    spdlog::error("Error while constructing Graph: bmruntime is null");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  init_graph_info();
}

void Graph::map_tensors(
    std::map<std::string, Tensor*>&          input,
    std::map<std::string, Tensor*>&          output,
    bm_tensor_t*                             bm_in,
    bm_tensor_t*                             bm_out) {
  for (auto& item : input) {
    auto idx = input_map_[item.first];
    bm_in[idx].dtype = (*input_dtypes_)[item.first];
    bm_in[idx].st_mode = BM_STORE_1N;
    bm_in[idx].device_mem = item.second->dev_data();
  }
  for (auto& item : output) {
    auto idx = output_map_[item.first];
    bm_out[idx].dtype = (*output_dtypes_)[item.first];
    bm_out[idx].st_mode = BM_STORE_1N;
    bm_out[idx].device_mem = item.second->dev_data();
  }
}

void Graph::map_input_shapes(
    std::map<std::string, std::vector<int>>& input_shapes,
    bm_tensor_t*                             bm_inputs) {
  for (auto& item : input_shapes) {
    int idx = input_map_[item.first];
    bm_inputs[idx].shape.num_dims = item.second.size();
    for (int i = 0; i < bm_inputs[idx].shape.num_dims; ++i) {
      bm_inputs[idx].shape.dims[i] = (item.second)[i];
    }
  }
}

void Graph::set_io_mode(IOMode io_mode) {
  io_mode_ = io_mode;
}

int Graph::alloc_tensors(
    IOMode                                   io_mode,
    std::map<std::string, std::vector<int>>* input_shapes,
    std::map<std::string, std::vector<int>>* output_shapes) {
  io_mode_ = io_mode;
  Handle handle(handle_);

  inputs_ = new bm_tensor_t[input_map_.size()];
  bool own_sys_data = (io_mode_ == SYSI || io_mode_ == SYSIO);
  for (auto& item : *input_shapes) {
    int idx = input_map_[item.first];
    Tensor* tensor = new Tensor(handle, item.second,
                                (*input_dtypes_)[item.first],
                                own_sys_data, true);
    input_tensors_[item.first] = tensor;
  }

  outputs_ = new bm_tensor_t[output_map_.size()];
  own_sys_data = (io_mode_ == SYSO || io_mode_ == SYSIO);
  for (auto& item : *output_shapes) {
    int idx = input_map_[item.first];
    Tensor* tensor = new Tensor(handle, item.second,
                                (*output_dtypes_)[item.first],
                                own_sys_data, true);
    output_tensors_[item.first] = tensor;
  }

  map_tensors(input_tensors_, output_tensors_, inputs_, outputs_);
  input_shapes_ = input_shapes;
  for (auto& item : input_tensors_) {
    int idx = input_map_[item.first];
    inputs_[idx].shape.num_dims = (*input_shapes)[item.first].size();
    for (int i = 0; i < inputs_[idx].shape.num_dims; ++i) {
      inputs_[idx].shape.dims[i] = (*input_shapes)[item.first][i];
    }
  }
  output_shapes_ = output_shapes;
  return 0;
}

void Graph::free() {
  for (auto& item : input_tensors_) {
    delete item.second;
    item.second = nullptr;
  }
  for (auto& item : output_tensors_) {
    delete item.second;
    item.second = nullptr;
  }
  input_tensors_.clear();
  output_tensors_.clear();
  if (inputs_) {
    delete [] inputs_;
    inputs_ = nullptr;
  }
  if (outputs_) {
    delete [] outputs_;
    outputs_ = nullptr;
  }
}

Graph::~Graph() {
  free();
}

Tensor* Graph::get_input_tensor(const std::string& name) {
  auto it = input_map_.find(name);
  if (it != input_map_.end()) {
    return input_tensors_[name];
  } else {
    spdlog::error("Invalid input tensor name: {}", name);
    exit(SAIL_ERR_ENGINE_INPUT);
  }
}

Tensor* Graph::get_output_tensor(const std::string& name) {
  auto it = output_map_.find(name);
  if (it != output_map_.end()) {
    return output_tensors_[name];
  } else {
    spdlog::error("Invalid output tensor name: {}", name);
    exit(SAIL_ERR_ENGINE_OUTPUT);
  }
}

void Graph::reshape() {
  for (auto& item : *input_shapes_) {
    size_t idx = input_map_[item.first];
    inputs_[idx].shape.num_dims = item.second.size();
    for (int i = 0; i < inputs_[idx].shape.num_dims; ++i) {
      inputs_[idx].shape.dims[i] = item.second[i];
    }
  }
}

void Graph::reset_input_tensor(
    const std::string& tensor_name,
    void*              data) {
  input_tensors_[tensor_name]->reset_sys_data(
      data, (*input_shapes_)[tensor_name]);
}

void Graph::init_graph_info() {
  const bm_net_info_t* p_info = bmrt_get_network_info(p_bmrt_, name_.c_str());
  for (int i = 0; i < p_info->input_num; ++i) {
    input_map_[p_info->input_names[i]] = i;
  }
  for (int i = 0; i < p_info->output_num; ++i) {
    output_map_[p_info->output_names[i]] = i;
  }
}

void Graph::init_dtypes(
    std::map<std::string, bm_data_type_t>* input_dtypes,
    std::map<std::string, bm_data_type_t>* output_dtypes) {
  input_dtypes_ = input_dtypes;
  output_dtypes_ = output_dtypes;
}

void Graph::input_s2d(
    std::map<std::string, Tensor*>&          input_tensors,
    bm_tensor_t*                             bm_inputs,
    std::map<std::string, std::vector<int>>& input_shapes) {
  for (auto& item : input_tensors) {
    int type_size = 0;
    if ((*input_dtypes_)[item.first] == BM_FLOAT32) {
      type_size = sizeof(float);
    } else if ((*input_dtypes_)[item.first] == BM_INT8) {
      type_size = sizeof(int8_t);
    } else if ((*input_dtypes_)[item.first] == BM_UINT8) {
      type_size = sizeof(uint8_t);
    }
    int size = std::accumulate(input_shapes[item.first].begin(),
               input_shapes[item.first].end(), 1, std::multiplies<int>())
               * type_size;
    item.second->sync_s2d(size);
  }
}

void Graph::output_d2s(
    std::map<std::string, Tensor*>&          output_tensors,
    bm_tensor_t*                             bm_outputs,
    std::map<std::string, std::vector<int>>& output_shapes) {
  for (auto& item : output_tensors) {
    int idx = output_map_[item.first];
    output_shapes[item.first].assign(bm_outputs[idx].shape.dims,
        bm_outputs[idx].shape.dims + bm_outputs[idx].shape.num_dims);
    int type_size = 0;
    if ((*output_dtypes_)[item.first] == BM_FLOAT32) {
      type_size = sizeof(float);
    } else if ((*output_dtypes_)[item.first] == BM_INT8) {
      type_size = sizeof(int8_t);
    } else if ((*output_dtypes_)[item.first] == BM_UINT8) {
      type_size = sizeof(uint8_t);
    }
    int size = std::accumulate(output_shapes[item.first].begin(),
               output_shapes[item.first].end(), type_size, std::multiplies<int>());
    item.second->sync_d2s(size);
  }
}

// inference with builtin tensors
void Graph::inference() {
  // copy input data from system memory to device memory
  map_input_shapes(*input_shapes_, inputs_);
  if (io_mode_ == SYSI || io_mode_ == SYSIO) {
    input_s2d(input_tensors_, inputs_, *input_shapes_);
  }
  // calculate on tpu
  bmrt_launch_tensor_ex(p_bmrt_,
                        name_.c_str(),
                        inputs_,
                        input_tensors_.size(),
                        outputs_,
                        output_tensors_.size(),
                        true,
                        true);
  // call this func to make sure calculation is done
  bm_thread_sync(handle_);
  // copy output data from device memory to system memory
  if (io_mode_ == SYSO || io_mode_ == SYSIO) {
    output_d2s(output_tensors_, outputs_, *output_shapes_);
  }
}

// inference with provided tensors
std::map<std::string, std::vector<int>> Graph::inference(
    std::map<std::string, Tensor*>&          input,
    std::map<std::string, std::vector<int>>& input_shapes,
    std::map<std::string, Tensor*>&          output) {
  bm_tensor_t bm_in[input.size()];
  bm_tensor_t bm_out[output.size()];
  map_tensors(input, output, bm_in, bm_out);
  map_input_shapes(input_shapes, bm_in);
  // copy input data from system memory to device memory
  if (io_mode_ == SYSI || io_mode_ == SYSIO) {
    input_s2d(input, bm_in, input_shapes);
  }
  // calculate on tpu
  bmrt_launch_tensor_ex(p_bmrt_,
                        name_.c_str(),
                        bm_in,
                        input.size(),
                        bm_out,
                        output.size(),
                        true,
                        true);
  // call this func to make sure calculation is done
  bm_thread_sync(handle_);
  std::map<std::string, std::vector<int>> output_shapes;
  // copy output data from device memory to system memory
  if (io_mode_ == SYSO || io_mode_ == SYSIO) {
    output_d2s(output, bm_out, output_shapes);
  }
  return output_shapes;
}

// inference with provided tensors
std::map<std::string, std::vector<int>> Graph::inference(
    std::map<std::string, Tensor*>& input,
    std::map<std::string, Tensor*>& output) {
  std::map<std::string, std::vector<int>> input_shapes;
  for (auto& item : input) {
    input_shapes[item.first] = item.second->shape();
  }
  return inference(input, input_shapes, output);
}

}  // namespace sail
