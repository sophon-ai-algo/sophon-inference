#include "tools.h"
#include <chrono>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <sstream>
#include <thread>
#include <vector>
#include "bmruntime_interface.h"
#include "tabulate.hpp"
using std::chrono::microseconds;

#define TENSOR_VAR(name, type, num) std::unique_ptr<type[]> \
          ptr##name(new type[num]); \
          type* name = ptr##name.get() \

void model_dryrun(std::string path) {
  int loopnum = 10;
  if (getenv("LOOPNUM")) {
    loopnum = std::stoi(getenv("LOOPNUM"));
  }
  const std::string dtype_names[] = {"float32", "float16", "int8",  "uint8",
                                     "int16",   "uint16",  "int32", "uint32"};
  // create bmruntime
  bm_handle_t bm_handle;
  bm_dev_request(&bm_handle, 0);
  void* p_bmrt = bmrt_create(bm_handle);
  int ret = bmrt_load_bmodel(p_bmrt, path.c_str());
  if (ret == 0) {
    printf("load bmodel file, check it available");
    exit(-1);
  }
  const char** net_names = nullptr;
  bmrt_get_network_names(p_bmrt, &net_names);
  int net_num = bmrt_get_network_number(p_bmrt);
  // init Table
  Table table;
  std::vector<std::string> head = {"net_name",  "in_node",   "in_shape",
                                   "out_node",  "out_shape", "npu_time",
                                   "total_time"};
  table.setHead(head);
  std::vector<std::vector<std::string>> body;

  for (int net_idx = 0; net_idx < net_num; ++net_idx) {
    auto net_info = bmrt_get_network_info(p_bmrt, net_names[net_idx]);
    //** clear std out
    // printf("\e[1;1H\e[2J");
    for (int stage_idx = 0; stage_idx < net_info->stage_num; stage_idx++) {
      std::string input_node, input_shape, output_node, output_shape;
      auto& stage = net_info->stages[stage_idx];
      // prepare input tensor
      TENSOR_VAR(input_tensors, bm_tensor_t, net_info->input_num);
      for (int input_idx = 0; input_idx < net_info->input_num; input_idx++) {
        input_node += std::string(net_info->input_names[input_idx]) + "(" +
                      dtype_names[net_info->input_dtypes[input_idx]] + ")";
        input_shape += "( ";
        auto shape = stage.input_shapes;
        for (int k = 0; k < shape->num_dims; ++k) {
          input_shape += (std::to_string(shape->dims[k]) + " ");
        }
        input_shape += ")";
        auto& input_tensor = input_tensors[input_idx];
        bmrt_tensor(&input_tensor, p_bmrt, net_info->input_dtypes[input_idx],
                    stage.input_shapes[input_idx]);
        bmrt_tensor_bytesize(&input_tensor);
      }
      // prepare output tensor
      TENSOR_VAR(output_tensors, bm_tensor_t, net_info->output_num);
      for (int output_idx = 0; output_idx < net_info->output_num;
           output_idx++) {
        output_node += std::string(net_info->output_names[output_idx]) + "(" +
                       dtype_names[net_info->output_dtypes[output_idx]] + ")";
        output_shape += "( ";
        auto shape = stage.output_shapes;
        for (int k = 0; k < shape->num_dims; ++k) {
          output_shape += (std::to_string(shape->dims[k]) + " ");
        }
        output_shape += ")";
        auto& output_tensor = output_tensors[output_idx];
        bmrt_tensor(&output_tensor, p_bmrt, net_info->output_dtypes[output_idx],
                    stage.input_shapes[output_idx]);
        bmrt_tensor_bytesize(&output_tensor);
      }

      bm_profile_t tic, toc;
      auto start = std::chrono::system_clock::now();
      for (int t = 0; t < loopnum; t++) {
        bm_get_profile(bm_handle, &tic);
        bool ret = bmrt_launch_tensor_ex(
            p_bmrt, net_names[net_idx], input_tensors, net_info->input_num,
            output_tensors, net_info->output_num, true, true);
        // sync, wait for finishing inference
        bm_thread_sync(bm_handle);
        bm_get_profile(bm_handle, &toc);
      }
      auto end = std::chrono::system_clock::now();
      auto duration = std::chrono::duration_cast<microseconds>(end - start);
      std::stringstream ss_npu, ss_total;
      ss_npu << std::fixed << std::setprecision(3)
             << (toc.tpu_process_time - tic.tpu_process_time) / 1000.0;
      ss_total << std::fixed << std::setprecision(3)
               << double(duration.count()) / loopnum * 1000.0 *
                      microseconds::period::num / microseconds::period::den;
      body.push_back({net_names[net_idx], input_node, input_shape, output_node,
                      output_shape, ss_npu.str(), ss_total.str()});

      // free memory
      for (int i = 0; i < net_info->input_num; ++i) {
        bm_free_device(bm_handle, input_tensors[i].device_mem);
      }
      for (int i = 0; i < net_info->output_num; ++i) {
        bm_free_device(bm_handle, output_tensors[i].device_mem);
      }
    }
  }
  table.setBody(body);
  table.print();
  free(net_names);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

float perf_thread(std::string path,
                  int tpu_id,
                  int net_id,
                  int stage_id,
                  int loopnum) {
  // create bmruntime
  bm_handle_t bm_handle;
  bm_dev_request(&bm_handle, tpu_id);
  void* p_bmrt = bmrt_create(bm_handle);
  bmrt_load_bmodel(p_bmrt, path.c_str());
  const char** net_names = nullptr;
  bmrt_get_network_names(p_bmrt, &net_names);
  int net_num = bmrt_get_network_number(p_bmrt);
  auto net_info = bmrt_get_network_info(p_bmrt, net_names[net_id]);
  auto& stage = net_info->stages[stage_id];
  // prepare input tensor
  TENSOR_VAR(input_tensors, bm_tensor_t, net_info->input_num);
  for (int input_idx = 0; input_idx < net_info->input_num; input_idx++) {
    auto& input_tensor = input_tensors[input_idx];
    bmrt_tensor(&input_tensor, p_bmrt, net_info->input_dtypes[input_idx],
                stage.input_shapes[input_idx]);
    bmrt_tensor_bytesize(&input_tensor);
  }
  // prepare output tensor
  TENSOR_VAR(output_tensors, bm_tensor_t, net_info->output_num);
  for (int output_idx = 0; output_idx < net_info->output_num; output_idx++) {
    auto& output_tensor = output_tensors[output_idx];
    bmrt_tensor(&output_tensor, p_bmrt, net_info->output_dtypes[output_idx],
                stage.input_shapes[output_idx]);
    bmrt_tensor_bytesize(&output_tensor);
  }
  auto start = std::chrono::system_clock::now();
  for (int t = 0; t < loopnum; t++) {
    bmrt_launch_tensor_ex(p_bmrt, net_names[net_id], input_tensors,
                          net_info->input_num, output_tensors,
                          net_info->output_num, true, true);
    // sync, wait for finishing inference
    bm_thread_sync(bm_handle);
  }
  auto end = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<microseconds>(end - start);
  float duration_value = float(duration.count()) / loopnum * 1000.0 *
                         microseconds::period::num / microseconds::period::den;
  // free memory
  for (int i = 0; i < net_info->input_num; ++i) {
    bm_free_device(bm_handle, input_tensors[i].device_mem);
  }
  for (int i = 0; i < net_info->output_num; ++i) {
    bm_free_device(bm_handle, output_tensors[i].device_mem);
  }
  free(net_names);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
  return duration_value;
}

void multi_tpu_perf(std::string path, std::vector<int> tpu_id_list) {
  // init some global variable
  const std::string dtype_names[] = {"float32", "float16", "int8",  "uint8",
                                     "int16",   "uint16",  "int32", "uint32"};
  int loopnum = 10;
  if (getenv("LOOPNUM")) {
    loopnum = std::stoi(getenv("LOOPNUM"));
  }
  int num_tpu = tpu_id_list.size();
  std::string tpu_info = "[ ";
  for (int tid : tpu_id_list)
    tpu_info += std::to_string(tid) + " ";
  tpu_info += "]";
  Table table;
  std::vector<std::string> head = {
      "net_name", "tpu_id",         "batch",          "in_node",
      "out_node", "total_time(ms)", "throughput(ips)"};
  table.setHead(head);
  std::vector<std::vector<std::string>> body;
  // load bmodel to extract info
  bm_handle_t bm_handle;
  bm_dev_request(&bm_handle, 0);
  void* p_bmrt = bmrt_create(bm_handle);
  int ret = bmrt_load_bmodel(p_bmrt, path.c_str());
  if (ret == 0) {
    printf("load bmodel file, check it available");
    exit(-1);
  }
  const char** net_names = nullptr;
  bmrt_get_network_names(p_bmrt, &net_names);
  int net_num = bmrt_get_network_number(p_bmrt);
  for (int net_id = 0; net_id < net_num; ++net_id) {
    auto net_info = bmrt_get_network_info(p_bmrt, net_names[net_id]);
    int stage_num = net_info->stage_num;
    for (int stage_id = 0; stage_id < stage_num; stage_id++) {
      int batch;  // extract batch size to calculate throughput
      std::string input_node, output_node;
      auto& stage = net_info->stages[stage_id];
      for (int input_idx = 0; input_idx < net_info->input_num; input_idx++) {
        input_node += std::string(net_info->input_names[input_idx]) + "(" +
                      dtype_names[net_info->input_dtypes[input_idx]] + ")";
        auto shape = stage.input_shapes;
        batch = shape->dims[0];
      }
      for (int output_idx = 0; output_idx < net_info->output_num;
           output_idx++) {
        output_node += std::string(net_info->output_names[output_idx]) + "(" +
                       dtype_names[net_info->output_dtypes[output_idx]] + ")";
      }
      float duration = 0;
      std::vector<std::future<float>> t_arr;
      for (int tpu_id : tpu_id_list) {
        t_arr.push_back(
            std::async(perf_thread, path, tpu_id, net_id, stage_id, loopnum));
      }
      for (auto& t : t_arr) {
        duration += t.get();
      }
      duration = duration / num_tpu;
      std::stringstream ss_time;
      ss_time << std::fixed << std::setprecision(3) << duration;
      std::string throughput =
          std::to_string(int(num_tpu * 1000 * batch / duration));
      body.push_back({net_names[net_id], tpu_info, std::to_string(batch),
                      input_node, output_node, ss_time.str(), throughput});
    }
  }
  table.setBody(body);
  // printf("\e[1;1H\e[2J");
  table.print();
  free(net_names);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

int main(void) {
  std::string path = "/home/tong.liu/combined.bmodel";
  std::vector<int> tvec = {0, 2};
  multi_tpu_perf(path, tvec);
  return 0;
}
