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
#include <thread>
#include <future>
#include <utility>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>
#include "processor.h"
#include "frame_provider.h"
#include "engine.h"
#include "tensor.h"

/**
 * @brief Load a bmodel and do inference
 *
 * @param thread_id    Thread id
 * @param engine       Pointer to an Engine instance
 * @param input_path   Path to input video file or rtsp stream
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
  auto graph_name = engine->get_graph_names().front();
  auto input_name = engine->get_input_names(graph_name).front();
  auto output_name = engine->get_output_names(graph_name).front();
  auto input_shape = engine->get_input_shape(graph_name, input_name);
  auto output_shape = engine->get_output_shape(graph_name, output_name);
  auto in_dtype = engine->get_input_dtype(graph_name, input_name);
  auto out_dtype = engine->get_input_dtype(graph_name, output_name);
  // get handle to create input and output tensors
  sail::Handle handle = engine->get_handle();
  sail::Bmcv bmcv(handle);
  // allocate input tensors with only device memory
  sail::Tensor in(handle, input_shape, in_dtype, false, true);
  // allocate input and output tensors with both system and device memory
  sail::Tensor out(handle, output_shape, out_dtype, true, true);
  std::map<std::string, sail::Tensor*> input_tensors = {{input_name, &in}};
  std::map<std::string, sail::Tensor*> output_tensors = {{output_name, &out}};
  float* output = nullptr;
  if (out_dtype == BM_FLOAT32) {
    output = reinterpret_cast<float*>(out.sys_data());
  }
  // set io_mode
  engine->set_io_mode(graph_name, sail::SYSO);
  // init preprocessor and postprocessor
  float input_scale = engine->get_input_scale(graph_name, input_name);
  PreProcessorBmcv preprocessor(bmcv, input_scale, input_shape[2], input_shape[3]);
  PostProcessor postprocessor(0.5);
  auto reference = postprocessor.get_reference(compare_path);
  // decode with ffmpeg
  int dev_id = engine->get_device_id();
  FFMpegFrameProvider frame_provider(bmcv, input_path, dev_id);
  sail::BMImage img0, img1;
  int i = 0;
  bool flag = true;
  while (!frame_provider.get(img0)) {
    int height = img0.height();
    int width = img0.width();
    img1 = bmcv.tensor_to_bm_image(in, true); // true -> bgr2rgb
    preprocessor.process(img0, img1);
    engine->process(graph_name, input_tensors, output_tensors);
    auto output_shape = engine->get_output_shape(graph_name, output_name);
    if (out_dtype != BM_FLOAT32) {
      // set output system data
      int out_size = std::accumulate(output_shape.begin(), output_shape.end(),
                                     1, std::multiplies<int>());
      output = new float[out_size];
      engine->scale_output_tensor(graph_name, output_name, output);
    }
    auto result = postprocessor.process(output, output_shape[2], height, width);
    if (out_dtype != BM_FLOAT32) {
      delete [] output;
    }
    std::string message("[Thread {} on tpu {}] Frame: {}, Category: {}, ");
    message += "Score: {}, Box: [{}, {}, {}, {}]";
    if (postprocessor.compare(reference, result, i)) {
      // print result
      for (auto& item : result) {
        cv::Rect rc;
        rc.x = item.x1;
        rc.y = item.y1;
        rc.width = item.x2 - item.x1;
        rc.height = item.y2 - item.y1;
        spdlog::info(message.c_str(), thread_id, dev_id, i, item.class_id,
                     item.score, rc.x, rc.y, rc.width, rc.height);
      }
    } else {
      flag = false;
      break;
    }
    ++i;
    if (i == loops) {
      break;
    }
  }
  if (flag) {
    status.set_value(true);
  } else {
    status.set_value(false);
  }
}

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
