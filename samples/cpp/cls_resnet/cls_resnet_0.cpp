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

#include <getopt.h>
#include <string>
#include <numeric>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
#include "engine.h"
#include "processor.h"

/**
 * @brief Load a bmodel and do inference.
 *
 * @param bmodel_path  Path to bmodel
 * @param input_path   Path to input image file
 * @param tpu_id       ID of TPU to use
 * @param loops        Number of loops to run
 * @param compare_path Path to correct result file
 * @return Program state
 *     @retval true  Success
 *     @retval false Failure
 */
bool inference(
    const std::string& bmodel_path,
    const std::string& input_path,
    int                tpu_id,
    int                loops,
    const std::string& compare_path) {
  // init Engine to load bmodel and allocate input and output tensors
  sail::Engine engine(bmodel_path, tpu_id, sail::SYSIO);
  // get model info
  // only one model loaded for this engine
  // only one input tensor and only one output tensor in this graph
  auto graph_name = engine.get_graph_names().front();
  auto input_name = engine.get_input_names(graph_name).front();
  auto output_name = engine.get_output_names(graph_name).front();
  auto input_shape = engine.get_input_shape(graph_name, input_name);
  auto output_shape = engine.get_output_shape(graph_name, output_name);
  auto in_dtype = engine.get_input_dtype(graph_name, input_name);
  auto out_dtype = engine.get_output_dtype(graph_name, output_name);
  // prepare input and output data in system memory with data type of float32
  float* input = nullptr;
  float* output = nullptr;
  int in_size = std::accumulate(input_shape.begin(), input_shape.end(),
                                1, std::multiplies<int>());
  int out_size = std::accumulate(output_shape.begin(), output_shape.end(),
                                 1, std::multiplies<int>());
  if (in_dtype == BM_FLOAT32) {
    input = reinterpret_cast<float*>(
        engine.get_input_tensor(graph_name, input_name)->sys_data());
  } else {
    input = new float[in_size];
  }
  if (out_dtype == BM_FLOAT32) {
    output = reinterpret_cast<float*>(
        engine.get_output_tensor(graph_name, output_name)->sys_data());
  } else {
    output = new float[out_size];
  }
  // init preprocessor and postprocessor
  PreProcessor preprocessor(input_shape[2], input_shape[3]);
  PostProcessor postprocessor(output_shape[0], output_shape[1], 5);
  auto reference = postprocessor.get_reference(compare_path);
  bool status = true;
  // pipeline of inference
  for (int i = 0; i < loops; ++i) {
    // read image
    cv::Mat frame = cv::imread(input_path);
    // preprocess
    preprocessor.process(input, frame);
    // scale input data if input data type is int8 or uint8
    if (in_dtype != BM_FLOAT32) {
      engine.scale_input_tensor(graph_name, input_name, input);
    }
    // inference
    engine.process(graph_name);
    // scale output data if input data type is int8 or uint8
    if (out_dtype != BM_FLOAT32) {
      engine.scale_output_tensor(graph_name, output_name, output);
    }
    // postprocess
    auto result = postprocessor.process(output);
    // print result
    for (auto item : result) {
      spdlog::info("Top 5 of loop {}: [{}]", i, fmt::join(item, ", "));
      if(!postprocessor.compare(reference, item,
          (out_dtype == BM_FLOAT32) ? "fp32" : "int8")) {
        status = false;
        break;
      }
    }
    if (!status) {
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
  return status;
}

/// It is the simplest case for inference of one model on one TPU.
int main(int argc, char** argv) {
  const char* opt_strings = "b:i:t:l:c:";
  const struct option long_opts[] = {
    {"bmodel", required_argument, nullptr, 'b'},
    {"input", required_argument, nullptr, 'i'},
    {"tpu_id", required_argument, nullptr, 't'},
    {"loops", required_argument, nullptr, 'l'},
    {"compare", required_argument, nullptr, 'c'}
  };
  std::string bmodel_path;
  std::string input_path;
  int tpu_id = 0;
  int loops = 1;
  std::string compare_path;
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
      case 't':
        tpu_id = std::stoi(optarg);
        break;
      case 'l':
        loops = std::stoi(optarg);
        break;
      case 'c':
        compare_path = optarg;
        break;
    }
  }
  if (bmodel_path.empty() || input_path.empty() || tpu_id < 0 || loops <= 0) {
    std::string usage("Usage: {} --bmodel bmodel_path --input input_path");
    usage += " [--tpu_id tpu_id(default:0)] [--loops loops_num(default:1)]";
    usage += " [--compare verify.ini]";
    spdlog::info(usage.c_str(), argv[0]);
    return -1;
  }
  // load bmodel and do inference
  bool status = inference(bmodel_path, input_path, tpu_id, loops, compare_path);
  return status ? 0 : -1;
}
