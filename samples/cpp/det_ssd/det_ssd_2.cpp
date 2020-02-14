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
#include <sstream>
#include <string>
#include <numeric>
#include "spdlog/spdlog.h"
#include "cvwrapper.h"
#include "engine.h"
#include "processor.h"

/**
 * @brief Load a bmodel and do inference.
 *
 * @param bmodel_path  Path to bmodel
 * @param input_path   Path to input file
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
  // init Engine
  sail::Engine engine(tpu_id);
  // load bmodel without builtin input and output tensors
  engine.load(bmodel_path);
  // get model info
  // only one model loaded for this engine
  // only one input tensor and only one output tensor in this graph
  auto graph_name = engine.get_graph_names().front();
  auto input_name = engine.get_input_names(graph_name).front();
  auto output_name = engine.get_output_names(graph_name).front();
  std::vector<int> input_shape = {4, 3, 300, 300};
  std::vector<int> output_shape = {1, 1, 800, 7};
  std::map<std::string, std::vector<int>> input_shapes;
  input_shapes[input_name] = input_shape;
  auto input_dtype = engine.get_input_dtype (graph_name, input_name);
  auto output_dtype = engine.get_output_dtype(graph_name, output_name);
  bool is_fp32 = (input_dtype == BM_FLOAT32);
  // get handle to create input and output tensors
  sail::Handle handle = engine.get_handle();
  // allocate input and output tensors with both system and device memory
  sail::Tensor in(handle, input_shape, input_dtype, false, false);
  sail::Tensor out(handle, output_shape, output_dtype, true, true);
  std::map<std::string, sail::Tensor*> input_tensors = {{input_name, &in}};
  std::map<std::string, sail::Tensor*> output_tensors = {{output_name, &out}};
  // set io_mode
  engine.set_io_mode(graph_name, sail::SYSO);
  // init preprocessor and postprocessor
  sail::Bmcv bmcv(handle);
  auto img_dtype = bmcv.get_bm_image_data_format(input_dtype);
  float scale = engine.get_input_scale(graph_name, input_name);
  BmcvPreProcessor preprocessor(bmcv, scale);
  float threshold = is_fp32 ? 0.59 : 0.52;
  PostProcessor postprocessor(threshold);
  auto reference = postprocessor.get_reference_4b(compare_path);
  // init decoder
  sail::Decoder decoder(input_path, true, tpu_id);
  bool status = true;
  // pipeline of inference
  for (int i = 0; i < loops; ++i) {
    sail::BMImageArray<4> imgs_0;
    sail::BMImageArray<4> imgs_1(handle, input_shape[2], input_shape[3],
                                 FORMAT_BGR_PLANAR, img_dtype);
    // read 4 images from image files or a video file
    for (int j = 0; j < 4; ++j) {
      imgs_0[j] = decoder.read_(handle);
    }
    // preprocess
    preprocessor.process(imgs_0, imgs_1);
    bmcv.bm_image_to_tensor(imgs_1, in);
    // inference
    engine.process(graph_name, input_tensors, input_shapes, output_tensors);
    auto real_output_shape = engine.get_output_shape(graph_name, output_name);
    // postprocess
    float* output_data = reinterpret_cast<float*>(out.sys_data());
    std::array<std::vector<DetectRect>, 4> dets;
    postprocessor.process(dets, output_data, real_output_shape,
                          imgs_0[0].width, imgs_0[0].height);
    // print result
    if (postprocessor.compare_4b(reference, dets, i)) {
      std::string message("[Frame {}] Category: {}, Score: {:03.3f}, ");
      message += "Box: [{}, {}, {}, {}]";
      for (int j = 0; j < 4; ++j) {
        int frame_id = i * 4 + j + 1;
        for (auto rect : dets[j]) {
          int x0 = rect.x1;
          int x1 = rect.x2;
          int y0 = rect.y1;
          int y1 = rect.y2;
          spdlog::info(message.c_str(), frame_id, rect.class_id,
                       rect.score, x0, y0, x1, y1);
          int w = x1 - x0 + 1;
          int h = y1 - y0 + 1;
          bmcv.rectangle(imgs_0[j], x0, y0, w, h, std::make_tuple(255, 0, 0), 3);
        }
        // save image with boxes
        std::stringstream ss;
        ss << "result-" << frame_id << ".jpg";
        bmcv.imwrite(ss.str(), imgs_0[j]);
      }
    } else {
      status = false;
      break;
    }
  }
  return status;
}

/// A SSD example with batch size is 4 for acceleration of int8 model, using bm-ffmpeg to
/// decode and using bmcv to preprocess.
int main(int argc, char *argv[]) {
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
