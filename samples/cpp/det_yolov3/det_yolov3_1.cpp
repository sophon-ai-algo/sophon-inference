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
 * @param bmodel_path  Bmodel path
 * @param input_path   Path to input video file or rtsp stream
 * @praam tpu_id       TPU id
 * @param loops        Number of loops to run
 * @param compare_path Path to correct result file
 * @param status       Status of comparison
 */
void do_inference(
    int                 thread_id,
    const std::string&  bmodel_path,
    const std::string&  input_path,
    int                 tpu_id,
    int                 loops,
    const std::string&  compare_path,
    std::promise<bool>& status) {
  // init Engine to load bmodel and allocate input&&output tensors
  sail::Engine engine(bmodel_path, tpu_id, sail::SYSO);
  // get model info
  auto graph_name = engine.get_graph_names().front();
  auto input_name = engine.get_input_names(graph_name).front();
  auto output_name = engine.get_output_names(graph_name).front();
  auto input_shape = engine.get_input_shape(graph_name, input_name);
  auto in_dtype = engine.get_input_dtype(graph_name, input_name);
  auto out_dtype = engine.get_input_dtype(graph_name, output_name);
  sail::Handle handle = engine.get_handle();
  sail::Bmcv bmcv(handle);
  float* output = nullptr;
  // set for preprocess
  float input_scale = engine.get_input_scale(graph_name, input_name);
  PreProcessorBmcv preprocessor(bmcv, input_scale, input_shape[2], input_shape[3]);
  PostProcessor postprocessor(0.5);
  auto reference = postprocessor.get_reference(compare_path);
  sail::Tensor* input_tensor = engine.get_input_tensor(graph_name, input_name);
  // set for decode with ffmpeg
  FFMpegFrameProvider frame_provider(bmcv, input_path, tpu_id);
  sail::BMImage img0, img1;
  int i = 0;
  bool flag = true;
  while (!frame_provider.get(img0)) {
    int height = img0.height();
    int width = img0.width();
    img1 = bmcv.tensor_to_bm_image(*input_tensor, true); // true -> bgr2rgb
    preprocessor.process(img0, img1);
    engine.process(graph_name);
    auto output_shape = engine.get_output_shape(graph_name, output_name);
    assert(output_shape.size() == 4);
    if (out_dtype == BM_FLOAT32) {
      output = reinterpret_cast<float*>(
          engine.get_output_tensor(graph_name, output_name)->sys_data());
    } else {
      // set output system data
      int out_size = std::accumulate(output_shape.begin(), output_shape.end(),
                                      1, std::multiplies<int>());
      output = new float[out_size];
      engine.scale_output_tensor(graph_name, output_name, output);
    }
    auto result = postprocessor.process(output, output_shape[2], height, width);
    if (out_dtype != BM_FLOAT32) {
      delete [] output;
    }
    std::string message("[Thread {}] Frame: {}, Category: {}, Score: {}, ");
    message += "Box: [{}, {}, {}, {}]";
    if (postprocessor.compare(reference, result, i)) {
      // print result
      for (auto& item : result) {
        cv::Rect rc;
        rc.x = item.x1;
        rc.y = item.y1;
        rc.width = item.x2 - item.x1;
        rc.height = item.y2 - item.y1;
        spdlog::info(message.c_str(), thread_id, i, item.class_id, item.score,
                     rc.x, rc.y, rc.width, rc.height);
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
    {"compare", required_argument, nullptr, 'c'}
  };
  std::string bmodel_path;
  std::string input_path;
  int thread_num = 2;
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
    }
  }
  if (bmodel_path.empty() || input_path.empty() ||
      tpu_id < 0 || thread_num <= 0 || loops <= 0 ) {
    std::string usage("Usage: {} --bmodel bmodel_path --input input_path");
    usage += " [--tpu_id tpu_id(default:0)] [--threads thread_num(default:2)]";
    usage += " [--loops loops_num(default:1)] [--compare verify.ini]";
    spdlog::info(usage.c_str(), argv[0]);
    return -1;
  }
  std::vector<std::thread> threads(thread_num);
  // use std::promise and std::future to get thread status
  std::vector<std::promise<bool>> status_promise(thread_num);
  std::vector<std::future<bool>> status_future(thread_num);
  for (int i = 0; i < thread_num; ++i) {
    threads[i] = std::thread(do_inference, i, bmodel_path, input_path, tpu_id,
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
