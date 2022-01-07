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
#include "internal.h"
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
  map_input_tensors(input, bm_in);
  map_output_tensors(output, bm_out);
}

void Graph::map_input_tensors(std::map<std::string, sail::Tensor *> &input, bm_tensor_t *bm_in) {
    for (auto& item : input) {
        auto idx = input_map_[item.first];
        bm_in[idx].dtype = (*input_dtypes_)[item.first];
        bm_in[idx].st_mode = BM_STORE_1N;
        auto& shape = item.second->shape();
        bm_in[idx].shape.num_dims = shape.size();
        for(int i = 0;i < shape.size(); ++i) {
            bm_in[idx].shape.dims[i] = shape[i];
        }
        bm_in[idx].device_mem = item.second->dev_data();
    }
}

void Graph::map_output_tensors(std::map<std::string, sail::Tensor *> &output, bm_tensor_t *bm_out) {
    for (auto& item : output) {
        auto idx = output_map_[item.first];
        bm_out[idx].dtype = (*output_dtypes_)[item.first];
        bm_out[idx].st_mode = BM_STORE_1N;
        bm_out[idx].device_mem = item.second->dev_data();
        auto& shape = item.second->shape();
        bm_out[idx].shape.num_dims = shape.size();
        for(int i = 0;i < shape.size(); ++i) {
            bm_out[idx].shape.dims[i] = shape[i];
        }
        bm_memset_device(handle_, 0, bm_out[idx].device_mem);
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

  assert(inputs_ == nullptr);
  inputs_ = new bm_tensor_t[input_map_.size()];
  bool own_sys_data = (io_mode_ == SYSI || io_mode_ == SYSIO);
  for (auto& item : *input_shapes) {
    int idx = input_map_[item.first];
    Tensor* tensor = new Tensor(handle, item.second,
                                (*input_dtypes_)[item.first],
                                own_sys_data, true);
    input_tensors_[item.first] = tensor;
  }

  assert(outputs_ == nullptr);
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

  //Remap again, because device address may be reallocated.
  map_input_tensors(input_tensors_, inputs_);
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
    } else if ((*input_dtypes_)[item.first] == BM_INT32){
      type_size = sizeof(int32_t);
    }else{
        SPDLOG_ERROR("unhandled input {}'s dtype={}", item.first, (*input_dtypes_)[item.first]);
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
    } else if ((*output_dtypes_)[item.first] == BM_INT32){
      type_size = sizeof(int32_t);
    }else{
        SPDLOG_ERROR("unhandled output {}'s dtype={}", item.first, (*output_dtypes_)[item.first]);
    }

    // update shape
    item.second->reset(output_shapes[item.first], (*output_dtypes_)[item.first]);

    // update data
    int size = std::accumulate(output_shapes[item.first].begin(),
               output_shapes[item.first].end(), type_size, std::multiplies<int>());
    item.second->sync_d2s(size);
  }
}

void Graph::
dump_io_tensors(const std::string& name, bm_tensor_t *ins, int in_num, bm_tensor_t *outs, int out_num)
{
    char tmp_file[256];
    sprintf(tmp_file, "%s_in.dat", name.c_str());
    // input tensors
    FILE *fp = fopen(tmp_file, "wb");
    for(int i = 0; i < in_num; ++i) {
        //auto shape = ins[i].shape;
        //fwrite(&shape.num_dims, 1, sizeof(int), fp);
        //fwrite(shape.dims, 1, sizeof(int) * shape.num_dims, fp);

        //dtype
        //fwrite(&ins[i].dtype, 1, sizeof(ins[i].dtype), fp);

        //data
        int size = bmrt_tensor_bytesize(&ins[i]);
        int8_t *data = new int8_t[size];
        bm_memcpy_d2s_partial(handle_, data, ins[i].device_mem, size);
        fwrite(data, 1, size, fp);
        delete [] data;
    }
    fclose(fp);

    //output tensors
    sprintf(tmp_file, "%s_out.dat", name.c_str());
    // input tensors
    fp = fopen(tmp_file, "wb");
    for(int i = 0; i < out_num; ++i) {
        //auto shape = outs[i].shape;
        //fwrite(&shape.num_dims, 1, sizeof(int), fp);
        //fwrite(shape.dims, 1, sizeof(int) * shape.num_dims, fp);

        //dtype
        //fwrite(&outs[i].dtype, 1, sizeof(outs[i].dtype), fp);

        //data
        int size = bmrt_tensor_bytesize(&outs[i]);
        int8_t *data = new int8_t[size];
        bm_memcpy_d2s_partial(handle_, data, outs[i].device_mem, size);
        fwrite(data, 1, size, fp);
        delete [] data;
    }
    fclose(fp);
}

// inference with builtin tensors
void Graph::inference() {
    int ret = 0;
  // copy input data from system memory to device memory
  map_input_shapes(*input_shapes_, inputs_);
  if (io_mode_ == SYSI || io_mode_ == SYSIO) {
     input_s2d(input_tensors_, inputs_, *input_shapes_);
  }else{
      SPDLOG_INFO("io_mode is {}, no input_s2d!", io_mode_str(io_mode_));
  }

  // memset output tensors
  for(int i = 0;i < output_tensors_.size(); ++i) {
      bm_memset_device(handle_, 0, outputs_[i].device_mem);
  }

  bool ok = bmrt_launch_tensor_ex(p_bmrt_,
                        name_.c_str(),
                        inputs_,
                        input_tensors_.size(),
                        outputs_,
                        output_tensors_.size(),
                        true,
                        true);
  if (!ok) {
      SPDLOG_ERROR("bmrt_launch_tensor_ex() err");
  }
  // call this func to make sure calculation is done
  bm_thread_sync(handle_);

  const char *dump_flag = getenv("SAIL_SAVE_IO_TENSORS");
  if (dump_flag != nullptr && 0 == strcmp(dump_flag, "1")){
        dump_io_tensors(name_, inputs_, input_tensors_.size(), outputs_, output_tensors_.size());
  }

  // copy output data from device memory to system memory
  if (io_mode_ == SYSO || io_mode_ == SYSIO) {
       output_d2s(output_tensors_, outputs_, *output_shapes_);
  }else{
      SPDLOG_INFO("io_mode is {}, no output_d2s!", io_mode_str(io_mode_));
  }
}


// inference with provided tensors
std::map<std::string, std::vector<int>> Graph::inference(
    std::map<std::string, Tensor*>&          input,
    std::map<std::string, std::vector<int>>& input_shapes,
    std::map<std::string, Tensor*>&          output) {
    TRACE_POINT;
  std::unique_ptr<bm_tensor_t[]> bm_in_ptr(new bm_tensor_t[input.size()]);
  std::unique_ptr<bm_tensor_t[]> bm_out_ptr(new bm_tensor_t[output.size()]);
  bm_tensor_t *bm_in = bm_in_ptr.get();
  bm_tensor_t *bm_out = bm_out_ptr.get();
  map_tensors(input, output, bm_in, bm_out);

  // copy input data from system memory to device memory
  if (io_mode_ == SYSI || io_mode_ == SYSIO) {
    input_s2d(input, bm_in, input_shapes);
  }else{
      SPDLOG_INFO("io_mode is {}, no input_s2d!", io_mode_str(io_mode_));
  }


  TRACE_POINT;
  // calculate on tpu
  if (!bmrt_launch_tensor_ex(p_bmrt_,
                        name_.c_str(),
                        bm_in,
                        input.size(),
                        bm_out,
                        output.size(),
                        true,
                        true)) {
      SPDLOG_ERROR("bmrt_launch_tensor_ex() failed");
  }
  // call this func to make sure calculation is done
  bm_thread_sync(handle_);
  TRACE_POINT;
  const char *dump_flag = getenv("SAIL_SAVE_IO_TENSORS");
  if (dump_flag != nullptr && 0 == strcmp(dump_flag, "1")){
      dump_io_tensors(name_, bm_in, input.size(), bm_out, output.size());
  }
  TRACE_POINT;
  std::map<std::string, std::vector<int>> output_shapes;
  // copy output data from device memory to system memory
  if (io_mode_ == SYSO || io_mode_ == SYSIO) {
    output_d2s(output, bm_out, output_shapes);
  }else{
      SPDLOG_INFO("io_mode is {}, no output_d2s!", io_mode_str(io_mode_));
  }
  TRACE_POINT;
  return output_shapes;
}

// inference with provided tensors
std::map<std::string, std::vector<int>> Graph::inference(
    std::map<std::string, Tensor*>& input,
    std::map<std::string, Tensor*>& output) {
    TRACE_POINT;
  std::map<std::string, std::vector<int>> input_shapes;
  for (auto& item : input) {
    input_shapes[item.first] = item.second->shape();
  }
  return inference(input, input_shapes, output);
}

}  // namespace sail
