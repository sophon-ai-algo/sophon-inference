/* Copyright 2016-2022 by Sophgo Technologies Inc. All rights reserved.

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

#ifdef _WIN32
#include "getopt_win.h"
#else
#include <getopt.h>
#endif

#include <string>
#include <vector>
#include <thread>
#include <future>
#include <utility>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
#include "tensor.h"
#include "engine.h"
#include "processor.h"

/**
 * @brief Do inference.
 *
 * @param thread_id    Thread id
 * @param engine       Pointer to an Engine instance
 * @param input_path   Path to input image file
 * @param loops        Number of loops to run
 * @param compare_path Path to correct result file
 * @param status       Status of comparison
 */
void thread_infer(
    int                 thread_id,
    sail::Engine*       engine,
    const std::string&  input_path,
    int                 loops,
    const std::string&  compare_path,
    std::promise<bool>& status) {
  // get model info
  // only one model loaded for this engine
  // only one input tensor and only one output tensor in this graph
  auto graph_name = engine->get_graph_names().front();
  auto input_name = engine->get_input_names(graph_name).front();
  auto output_name = engine->get_output_names(graph_name).front();
  auto input_shape = engine->get_input_shape(graph_name, input_name);
  auto output_shape = engine->get_output_shape(graph_name, output_name);
  auto in_dtype = engine->get_input_dtype(graph_name, input_name);
  auto out_dtype = engine->get_output_dtype(graph_name, output_name);
  // get handle to create input and output tensors
  sail::Handle handle = engine->get_handle();
  // allocate input and output tensors with both system and device memory
  sail::Tensor in(handle, input_shape, in_dtype, true, true);
  sail::Tensor out(handle, output_shape, out_dtype, true, true);
  std::map<std::string, sail::Tensor*> input_tensors = {{input_name, &in}};
  std::map<std::string, sail::Tensor*> output_tensors = {{output_name, &out}};
  // prepare input and output data in system memory with data type of float32
  float* input = nullptr;
  float* output = nullptr;
  int in_size = std::accumulate(input_shape.begin(), input_shape.end(),
                                1, std::multiplies<int>());
  int out_size = std::accumulate(output_shape.begin(), output_shape.end(),
                                 1, std::multiplies<int>());
  if (in_dtype == BM_FLOAT32) {
    input = reinterpret_cast<float*>(in.sys_data());
  } else {
    input = new float[in_size];
  }
  if (out_dtype == BM_FLOAT32) {
    output = reinterpret_cast<float*>(out.sys_data());
  } else {
    output = new float[out_size];
  }
  // set io_mode
  engine->set_io_mode(graph_name, sail::SYSIO);
  // init preprocessor and postprocessor
  PreProcessor preprocessor(input_shape[2], input_shape[3]);
  PostProcessor postprocessor(output_shape[0], output_shape[1], 5);
  auto reference = postprocessor.get_reference(compare_path);
  bool flag = true;
  // pipeline of inference
  for (int i = 0; i < loops; ++i) {
    // read image
    cv::Mat frame = cv::imread(input_path);
    // preprocess
    preprocessor.process(input, frame);
    // scale input data if input data type is int8 or uint8
    if (in_dtype != BM_FLOAT32) {
      float scale = engine->get_input_scale(graph_name, input_name);
      if (in_dtype == BM_INT8) {
        engine->scale_fp32_to_int8(input,
            reinterpret_cast<int8_t*>(in.sys_data()), scale, in_size);
      }else if (in_dtype == BM_UINT8) {
        engine->scale_fp32_to_uint8(input,
            reinterpret_cast<uint8_t*>(in.sys_data()), scale, in_size);
      }
    }
    // inference
    engine->process(graph_name, input_tensors, output_tensors);
    // scale output data if input data type is int8 or uint8
    if (out_dtype != BM_FLOAT32) {
      float scale = engine->get_output_scale(graph_name, output_name);
      if (out_dtype == BM_INT8) {
        engine->scale_int8_to_fp32(reinterpret_cast<int8_t*>(out.sys_data()),
                                   output, scale, out_size);
      } else if (out_dtype == BM_UINT8) {
        engine->scale_uint8_to_fp32(reinterpret_cast<uint8_t*>(out.sys_data()),
                                    output, scale, out_size);
      }
    }
    // postprocess
    auto result = postprocessor.process(output);
    // print result
    for (auto item : result) {
      spdlog::info("Top 5 of loop {} in thread {} on tpu {}: [{}]", i,
                   thread_id, engine->get_device_id(), fmt::join(item, ", "));
      if(!postprocessor.compare(reference, item,
          (out_dtype == BM_FLOAT32) ? "fp32" : "int8")) {
        flag = false;
        break;
      }
    }
    if (!flag) {
      break;
    }
  }
  // free data
  if (in_dtype != BM_FLOAT32) {
    delete [] input;
  }
  if (out_dtype != BM_FLOAT32) {
    delete [] output;
  }
  if (flag) {
    status.set_value(true);
  } else {
    status.set_value(false);
  }
}

/// An example shows inference of one model by multiple threads on one TPU.
int main(int argc, char** argv) {
  const char* opt_strings = "b:i:d:t:l:c:";
  const struct option long_opts[] = {
    {"bmodel", required_argument, nullptr, 'b'},
    {"input", required_argument, nullptr, 'i'},
    {"tpu_id", required_argument, nullptr, 'd'},
    {"threads", required_argument, nullptr, 't'},
    {"loops", required_argument, nullptr, 'l'},
    {"compare", required_argument, nullptr, 'c'},
    {0, 0, 0, 0}
  };
  std::string bmodel_path;
  std::string input_path;
  int thread_num = 2;
  int tpu_id = 0;
  int loops = 1;
  std::string compare_path;
  bool flag = false;
  while (1) {
    int c = getopt_long(argc, argv, opt_strings, long_opts, nullptr);
    if (c == -1) {
      break;
    }
    switch (c) {
      case 'b':
        bmodel_path = optarg;
        break;
      case 'i':
        input_path = optarg;
        break;
      case 'd':
        tpu_id = std::stoi(optarg);
        break;
      case 't':
        thread_num = std::stoi(optarg);
        break;
      case 'l':
        loops = std::stoi(optarg);
        break;
      case 'c':
        compare_path = optarg;
        break;
      case '?':
        flag = true;
        break;
    }
  }
  if (flag || bmodel_path.empty() || input_path.empty() ||
      tpu_id < 0 || thread_num <= 0 || loops <= 0 ) {
    std::string usage("Usage: {} --bmodel bmodel_path --input input_path");
    usage += " [--tpu_id tpu_id(default:0)] [--threads thread_num(default:2)]";
    usage += " [--loops loops_num(default:1)] [--compare verify.ini]";
    spdlog::info(usage.c_str(), argv[0]);
    return -1;
  }
  if (!file_exists(input_path)) {
    spdlog::error("File not exists: {}", input_path);
    return -2;
  }
  // init Engine
  sail::Engine engine(tpu_id);
  // load bmodel without builtin input and output tensors
  // each thread manage its input and output tensors
  int ret = engine.load(bmodel_path);
  // create threads for inference
  std::vector<std::thread> threads(thread_num);
  // use std::promise and std::future to get thread status
  std::vector<std::promise<bool>> status_promise(thread_num);
  std::vector<std::future<bool>> status_future(thread_num);
  for (int i = 0; i < thread_num; ++i) {
    threads[i] = std::thread(thread_infer, i, &engine, input_path,
                             loops, compare_path, std::ref(status_promise[i]));
    status_future[i] = status_promise[i].get_future();
  }
  for (int i = 0; i < thread_num; ++i) {
    threads[i].join();
  }
  for (int i = 0; i < thread_num; ++i) {
    if (!status_future[i].get()) {
      return -1;
    }
  }
  return 0;
}
