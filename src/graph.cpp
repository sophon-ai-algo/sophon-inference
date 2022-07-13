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

class Graph::Graph_CC{
public:
  explicit Graph_CC(
      bm_handle_t        handle,
      void*              p_bmrt,
      const std::string& name);

  virtual ~Graph_CC() {};

  void free();

  void reshape();

  Tensor* get_input_tensor(const std::string& name);

  Tensor* get_output_tensor(const std::string& name);

  void reset_input_tensor(const std::string& tensor_name, void* data);

  void init_dtypes(
    std::map<std::string, bm_data_type_t>* input_dtypes,
    std::map<std::string, bm_data_type_t>* output_dtypes);

  int alloc_tensors(
    IOMode                                   io_mode,
    std::map<std::string, std::vector<int>>* input_shapes,
    std::map<std::string, std::vector<int>>* output_shapes);

  void set_io_mode(IOMode io_mode);

  void inference();

  std::map<std::string, std::vector<int>> inference(
    std::map<std::string, Tensor*>&          input,
    std::map<std::string, std::vector<int>>& input_shapes,
    std::map<std::string, Tensor*>&          output);
private:
    /**
   * @brief Forbidden copy constructor.
   */
  Graph_CC(const Graph_CC &other);

  /**
   * @brief Forbidden assignment function.
   */
  Graph_CC& operator=(const Graph_CC &other);

  friend class Graph;

  /**
   * @brief Initialize graph information, such as input/output name and number
   */
  void init_graph_info();

  /**
   * @brief Copy all input tensors from system memory to device memory.
   *
   * @param input_tensors Input tensors
   * @param bm_inputs     A pointer to bm_tensor_t array for input tensors
   * @param input_shapes  Real shapes of input tensors
   */
  void input_s2d(
    std::map<std::string, Tensor*>&          input_tensors,
    bm_tensor_t*                             bm_inputs,
    std::map<std::string, std::vector<int>>& input_shapes);

  /**
   * @brief Copy all output tensors from device memory to system memory.
   *
   * @param output_tensors output tensors
   * @param bm_outputs     A pointer to bm_tensor_t array for output tensors
   * @param output_shapes  Real shapes of output tensors
   */
  void output_d2s(
    std::map<std::string, Tensor*>&          output_tensors,
    bm_tensor_t*                             bm_outputs,
    std::map<std::string, std::vector<int>>& output_shapes);

  /**
   * @brief Map tensors to bm_tensors.
   *
   * @param input        A map contains Tensor instances of input
   * @param output       A map contains Tensor instances of output
   * @param bm_in        A pointer to bm_tensor_t array for input tensors
   * @param bm_out       A pointer to bm_tensor_t array for output tensors
   */
  void map_tensors(
      std::map<std::string, Tensor*>&          input,
      std::map<std::string, Tensor*>&          output,
      bm_tensor_t*                             bm_in,
      bm_tensor_t*                             bm_out);

  void map_input_tensors(
            std::map<std::string, Tensor*>&          input,
            bm_tensor_t*                             bm_in);

  void map_output_tensors(
            std::map<std::string, Tensor*>&          output,
            bm_tensor_t*                             bm_out);

  void dump_io_tensors(const std::string& name, bm_tensor_t *ins, int in_num, bm_tensor_t *outs, int out_num);
  
  void dump_uint8_data(const char* name, int8_t *ptensor, int data_len, int data_w, int data_h);

  void dump_float32_data(const char* name, float *ptensor, int data_len, int data_w, int data_h);
  
  void dump_int32_data(const char* name, int *ptensor, int data_len, int data_w, int data_h);

  /**
   * @brief Map real input shapes to bm_tensors.
   *
   * @param input_shapes Real shapes of input tensors
   * @param bm_in        A pointer to bm_tensor_t array for input tensors
   */
  void map_input_shapes(
      std::map<std::string, std::vector<int>>& input_shapes,
      bm_tensor_t*                             bm_inputs);

  /// Graph name
  std::string name_;

  /// Indicator of where to store input and output tensors.
  /// SYSI: Input tensors are in system memory while output tensors are
  ///       in device memory.
  /// SYSO: Input tensors are in device memory while output tensors are
  ///       in system memory.
  /// SYSIO: Both input and output tensors are in system memory.
  /// DEVIO: Both input and output tensors are in device memory.
  IOMode io_mode_;

  /// bm_handle.
  bm_handle_t handle_;

  /// Pointer to bmruntime.
  void* p_bmrt_;

  /// Data type of input tensors
  std::map<std::string, bm_data_type_t>* input_dtypes_;

  /// Data type of output tensors
  std::map<std::string, bm_data_type_t>* output_dtypes_;

  /// Map of input tensor names and indexes
  std::map<std::string, size_t> input_map_;

  /// Map of output tensor names and indexes
  std::map<std::string, size_t> output_map_;

  /// Pointers to all input tensors needed by bmrt_launch
  bm_tensor_t* inputs_;

  /// Pointers to all output tensors needed by bmrt_launch
  bm_tensor_t* outputs_;

  /// Pointers to all input tensors
  std::map<std::string, Tensor*> input_tensors_;

  /// Pointers to all output tensors
  std::map<std::string, Tensor*> output_tensors_;

  /// Shapes of all input tensors
  std::map<std::string, std::vector<int>>* input_shapes_;

  /// Shapes of all output tensors
  std::map<std::string, std::vector<int>>* output_shapes_;

};


Graph::Graph_CC::Graph_CC(
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

void Graph::Graph_CC::init_graph_info() {
  const bm_net_info_t* p_info = bmrt_get_network_info(p_bmrt_, name_.c_str());
  for (int i = 0; i < p_info->input_num; ++i) {
    input_map_[p_info->input_names[i]] = i;
  }
  for (int i = 0; i < p_info->output_num; ++i) {
    output_map_[p_info->output_names[i]] = i;
  }
}

void Graph::Graph_CC::free() {
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

void Graph::Graph_CC::map_input_tensors(std::map<std::string, sail::Tensor *> &input, bm_tensor_t *bm_in) {
    for (auto& item : input) {
        auto idx = input_map_[item.first];
        bm_in[idx].dtype = (*input_dtypes_)[item.first];
        bm_in[idx].st_mode = BM_STORE_1N;
        bm_in[idx].device_mem = item.second->dev_data();
    }
}

void Graph::Graph_CC::map_output_tensors(std::map<std::string, sail::Tensor *> &output, bm_tensor_t *bm_out) {
    for (auto& item : output) {
        auto idx = output_map_[item.first];
        bm_out[idx].dtype = (*output_dtypes_)[item.first];
        bm_out[idx].st_mode = BM_STORE_1N;
        bm_out[idx].device_mem = item.second->dev_data();
        // double time_temp = get_current_time_us();
        // bm_memset_device(handle_, 0, bm_out[idx].device_mem);
        // PRINT_TIME_MS("bm_memset_device .....",time_temp)
    }
}

void Graph::Graph_CC::map_tensors(
    std::map<std::string, Tensor*>&          input,
    std::map<std::string, Tensor*>&          output,
    bm_tensor_t*                             bm_in,
    bm_tensor_t*                             bm_out) {
  map_input_tensors(input, bm_in);
  map_output_tensors(output, bm_out);
}

void Graph::Graph_CC::map_input_shapes(
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

void Graph::Graph_CC::input_s2d(
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

void Graph::Graph_CC::output_d2s(
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
    int size = std::accumulate(output_shapes[item.first].begin(),
               output_shapes[item.first].end(), type_size, std::multiplies<int>());
    item.second->sync_d2s(size);
  }
}

void Graph::Graph_CC::dump_uint8_data(const char* name, int8_t *ptensor, int data_len, int data_w, int data_h)
{
  FILE *fp_temp = fopen(name, "w+");
  int line_num = 0;
  for(int mm=0;mm<data_len;++mm){
    char txt_temp[32]={0};
    if(data_w >= 1){
      if (mm%data_w == data_w-1){
        sprintf(txt_temp,"%03d\n",ptensor[mm]);
        line_num++;
        if(line_num%data_h == 0){
          sprintf(txt_temp,"%03d\n\n",ptensor[mm]);
        }
      }else{
        sprintf(txt_temp,"%03d ",ptensor[mm]);
      }
    }else{
        sprintf(txt_temp,"%03d ",ptensor[mm]);
    }
    fputs(txt_temp,fp_temp);
  }
  fclose(fp_temp);
}

void Graph::Graph_CC::dump_float32_data(const char* name, float *ptensor, int data_len, int data_w, int data_h)
{
  FILE *fp_temp = fopen(name, "w+");
  int line_num = 0;
  for(int mm=0;mm<data_len;++mm){
    char txt_temp[32]={0};
    if(data_w >= 1){
      if (mm%data_w == data_w-1){
        sprintf(txt_temp,"%.3f\n",ptensor[mm]);
        line_num++;
        if(line_num%data_h == 0){
          sprintf(txt_temp,"%.3f\n\n",ptensor[mm]);
        }
      }else{
        sprintf(txt_temp,"%.3f ",ptensor[mm]);
      }
    }else{
        sprintf(txt_temp,"%.3f ",ptensor[mm]);
    }
    fputs(txt_temp,fp_temp);
  }
  fclose(fp_temp);
}

void Graph::Graph_CC::dump_int32_data(const char* name, int *ptensor, int data_len, int data_w, int data_h)
{
  FILE *fp_temp = fopen(name, "w+");
  int line_num = 0;
  for(int mm=0;mm<data_len;++mm){
    char txt_temp[32]={0};
    if(data_w >= 1){
      if (mm%data_w == data_w-1){
        sprintf(txt_temp,"%d\n",ptensor[mm]);
        line_num++;
        if(line_num%data_h == 0){
          sprintf(txt_temp,"%d\n\n",ptensor[mm]);
        }
      }else{
        sprintf(txt_temp,"%d ",ptensor[mm]);
      }
    }else{
        sprintf(txt_temp,"%d ",ptensor[mm]);
    }
    fputs(txt_temp,fp_temp);
  }
  fclose(fp_temp);
}

void Graph::Graph_CC::dump_io_tensors(const std::string& name, bm_tensor_t *ins, int in_num, bm_tensor_t *outs, int out_num)
{
    // input tensors
    for(int i = 0; i < in_num; ++i) {
        //auto shape = ins[i].shape;
        //fwrite(&shape.num_dims, 1, sizeof(int), fp);
        //fwrite(shape.dims, 1, sizeof(int) * shape.num_dims, fp);

        //dtype
        //fwrite(&ins[i].dtype, 1, sizeof(ins[i].dtype), fp);

        //data
        int size = bmrt_tensor_bytesize(&ins[i]);
        int8_t *data = new int8_t[size];
        double process_start_time_d2s = get_current_time_us();
        bm_memcpy_d2s_partial(handle_, data, ins[i].device_mem, size);
        PRINT_TIME_MS("bm_memcpy_d2s_partial", process_start_time_d2s)

        process_start_time_d2s = get_current_time_us();
        int tensor_w = 0;
        int tensor_h = 0;
        if(ins[i].shape.num_dims>=3){
          tensor_h = ins[i].shape.dims[ins[i].shape.num_dims-1];
          tensor_w = ins[i].shape.dims[ins[i].shape.num_dims-2];
        }
        char tmp_file_in[256];
        sprintf(tmp_file_in, "%s_in_%d.dat", name.c_str(), i);
        if(ins[i].dtype == BM_UINT8 || ins[i].dtype == BM_INT8){
          dump_uint8_data(tmp_file_in,data,size,tensor_w,tensor_h);
        }else if(ins[i].dtype == BM_FLOAT32){
          dump_float32_data(tmp_file_in,(float*)data,size/4,tensor_w,tensor_h);
        }else if(ins[i].dtype == BM_INT32){
          dump_int32_data(tmp_file_in,(int*)data,size/4,tensor_w,tensor_h);
        }
        else{
          FILE *fp = fopen(tmp_file_in, "wb");
          fwrite(data, 1, size, fp);
          fclose(fp);
        }
        PRINT_TIME_MS("dump_io_data_in", process_start_time_d2s)
        delete [] data;
    }

    //output tensors
    for(int i = 0; i < out_num; ++i) {
        //auto shape = outs[i].shape;
        //fwrite(&shape.num_dims, 1, sizeof(int), fp);
        //fwrite(shape.dims, 1, sizeof(int) * shape.num_dims, fp);

        //dtype
        //fwrite(&outs[i].dtype, 1, sizeof(outs[i].dtype), fp);

        //data
        int size = bmrt_tensor_bytesize(&outs[i]);
        int8_t *data = new int8_t[size];
        double process_start_time_d2s = get_current_time_us();
        bm_memcpy_d2s_partial(handle_, data, outs[i].device_mem, size);
        PRINT_TIME_MS("bm_memcpy_d2s_partial", process_start_time_d2s)
        
        process_start_time_d2s = get_current_time_us();
        int tensor_w = 0;
        int tensor_h = 0;
        if(outs[i].shape.num_dims>=3){
          tensor_h = outs[i].shape.dims[outs[i].shape.num_dims-1];
          tensor_w = outs[i].shape.dims[outs[i].shape.num_dims-2];
        }
        char tmp_file_in[256];
        sprintf(tmp_file_in, "%s_out_%d.dat", name.c_str(), i);
        if(outs[i].dtype == BM_UINT8 || outs[i].dtype == BM_INT8){
          dump_uint8_data(tmp_file_in,data,size,tensor_w,tensor_h);
        }else if(outs[i].dtype == BM_FLOAT32){
          dump_float32_data(tmp_file_in,(float*)data,size/4,tensor_w,tensor_h);
        }else if(outs[i].dtype == BM_INT32){
          dump_int32_data(tmp_file_in,(int*)data,size/4,tensor_w,tensor_h);
        }else{
          FILE* fp = fopen(tmp_file_in, "wb");
          fwrite(data, 1, size, fp);
          fclose(fp);
        }
        PRINT_TIME_MS("dump_io_data_out", process_start_time_d2s)
        delete [] data;
    }
}

Tensor* Graph::Graph_CC::get_input_tensor(const std::string& name) {
  auto it = input_map_.find(name);
  if (it != input_map_.end()) {
    return input_tensors_[name];
  } else {
    spdlog::error("Invalid input tensor name: {}", name);
    exit(SAIL_ERR_ENGINE_INPUT);
  }
}

Tensor* Graph::Graph_CC::get_output_tensor(const std::string& name) {
  auto it = output_map_.find(name);
  if (it != output_map_.end()) {
    return output_tensors_[name];
  } else {
    spdlog::error("Invalid output tensor name: {}", name);
    exit(SAIL_ERR_ENGINE_OUTPUT);
  }
}

void Graph::Graph_CC::reshape() {
  for (auto& item : *input_shapes_) {
    size_t idx = input_map_[item.first];
    inputs_[idx].shape.num_dims = item.second.size();
    for (int i = 0; i < inputs_[idx].shape.num_dims; ++i) {
      inputs_[idx].shape.dims[i] = item.second[i];
    }
  }
}

void Graph::Graph_CC::reset_input_tensor(
    const std::string& tensor_name,
    void*              data) {
  input_tensors_[tensor_name]->reset_sys_data(data, (*input_shapes_)[tensor_name]);
  //Remap again, because device address may be reallocated.
  map_input_tensors(input_tensors_, inputs_);
}

void Graph::Graph_CC::init_dtypes(
    std::map<std::string, bm_data_type_t>* input_dtypes,
    std::map<std::string, bm_data_type_t>* output_dtypes) {
  input_dtypes_ = input_dtypes;
  output_dtypes_ = output_dtypes;
}

void Graph::Graph_CC::set_io_mode(IOMode io_mode) {
  io_mode_ = io_mode;
}

int Graph::Graph_CC::alloc_tensors(
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

void Graph::Graph_CC::inference() {
    int ret = 0;
  // copy input data from system memory to device memory
  map_input_shapes(*input_shapes_, inputs_);
  if (io_mode_ == SYSI || io_mode_ == SYSIO) {
    input_s2d(input_tensors_, inputs_, *input_shapes_);
  }

  // memset output tensors
  // for(int i = 0;i < output_tensors_.size(); ++i) {
  //     bm_memset_device(handle_, 0, outputs_[i].device_mem);
  // }

  double process_start_time = get_current_time_us();
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
  PRINT_TIME_MS("bmrt_launch_tensor_ex",process_start_time);

  const char *dump_flag = getenv("SAIL_SAVE_IO_TENSORS");
  if (dump_flag != nullptr && 0 == strcmp(dump_flag, "1")){
    double id_save = get_current_time_us();
    char save_name_temp[256]={0};
    sprintf(save_name_temp,"%s_%.0f",name_.c_str(),id_save);
    dump_io_tensors(std::string(save_name_temp), inputs_, input_tensors_.size(), outputs_, output_tensors_.size());
  }

  // copy output data from device memory to system memory
  if (io_mode_ == SYSO || io_mode_ == SYSIO) {
    output_d2s(output_tensors_, outputs_, *output_shapes_);
  }
}

std::map<std::string, std::vector<int>> Graph::Graph_CC::inference(
    std::map<std::string, Tensor*>&          input,
    std::map<std::string, std::vector<int>>& input_shapes,
    std::map<std::string, Tensor*>&          output) {
    TRACE_POINT;
  std::unique_ptr<bm_tensor_t[]> bm_in_ptr(new bm_tensor_t[input.size()]);
  std::unique_ptr<bm_tensor_t[]> bm_out_ptr(new bm_tensor_t[output.size()]);
  bm_tensor_t *bm_in = bm_in_ptr.get();
  bm_tensor_t *bm_out = bm_out_ptr.get();
  map_tensors(input, output, bm_in, bm_out);
  map_input_shapes(input_shapes, bm_in);
  // copy input data from system memory to device memory
  if (io_mode_ == SYSI || io_mode_ == SYSIO) {
    input_s2d(input, bm_in, input_shapes);
  }
  TRACE_POINT;
  // calculate on tpu
  double process_start_time = get_current_time_us();
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
  PRINT_TIME_MS("bmrt_launch_tensor_ex",process_start_time);

  TRACE_POINT;
  const char *dump_flag = getenv("SAIL_SAVE_IO_TENSORS");
  if (dump_flag != nullptr && 0 == strcmp(dump_flag, "1")){
    char save_name_temp[256]={0};
    double id_save = get_current_time_us();
    sprintf(save_name_temp,"%.0f_%s",name_.c_str(),id_save);
    dump_io_tensors(std::string(save_name_temp), bm_in, input.size(), bm_out, output.size());
  }
  TRACE_POINT;
  std::map<std::string, std::vector<int>> output_shapes;
  // copy output data from device memory to system memory
  if (io_mode_ == SYSO || io_mode_ == SYSIO) {
    output_d2s(output, bm_out, output_shapes);
  }
  TRACE_POINT;
  return output_shapes;
}

Graph::Graph(
    bm_handle_t        handle,
    void*              p_bmrt,
    const std::string& name)
    : _impl(new Graph_CC(handle, p_bmrt, name)) {
}

void Graph::set_io_mode(IOMode io_mode) {
  return _impl->set_io_mode(io_mode);
}

int Graph::alloc_tensors(
    IOMode                                   io_mode,
    std::map<std::string, std::vector<int>>* input_shapes,
    std::map<std::string, std::vector<int>>* output_shapes) {
  return _impl->alloc_tensors(io_mode, input_shapes, output_shapes);
}

void Graph::free() {
  return _impl->free();
}

Graph::~Graph() {
  free();
  delete _impl;
}

Tensor* Graph::get_input_tensor(const std::string& name) {
  return _impl->get_input_tensor(name);
}

Tensor* Graph::get_output_tensor(const std::string& name) {
  return _impl->get_output_tensor(name);
}

void Graph::reshape() {
  return _impl->reshape();
}

void Graph::reset_input_tensor(
    const std::string& tensor_name,
    void*              data) {
  return _impl->reset_input_tensor(tensor_name, data);
}

void Graph::init_dtypes(
    std::map<std::string, bm_data_type_t>* input_dtypes,
    std::map<std::string, bm_data_type_t>* output_dtypes) {
  return _impl->init_dtypes(input_dtypes, output_dtypes);
}

// inference with builtin tensors
void Graph::inference() {
  return _impl->inference();
}

// inference with provided tensors
std::map<std::string, std::vector<int>> Graph::inference(
    std::map<std::string, Tensor*>&          input,
    std::map<std::string, std::vector<int>>& input_shapes,
    std::map<std::string, Tensor*>&          output) {
  return _impl->inference(input, input_shapes, output);
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
