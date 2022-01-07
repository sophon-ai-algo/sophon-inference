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
#include <map>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
#include "engine.h"
#include "processor.h"

/**
 * @brief Run PNet, the first stage of MTCNN.
 *
 * @param engine        Engine instance
 * @param preprocessor  PreProcessor instance
 * @param postprocessor PostProcessor instance
 * @param scales        Scale factors
 * @param frame         Image
 * @return Detected boxes
 */
std::vector<FaceInfo> run_pnet(
    sail::Engine&  engine,
    PreProcessor&  preprocessor,
    PostProcessor& postprocessor,
    cv::Mat&       frame) {
  std::string graph_name("PNet");
  std::vector<FaceInfo> boxes;
  cv::Mat image;
  frame.copyTo(image);
  int height = image.rows;
  int width = image.cols;
  auto input_name = engine.get_input_names(graph_name).front();
  auto  map_maxinput_shape = engine.get_max_input_shapes(graph_name);
  std::vector<int> maxinput_shape;
  for (auto max_shape : map_maxinput_shape){
    if(strcmp(input_name.c_str(),max_shape.first.c_str()) == 0){
        maxinput_shape = max_shape.second;
        break;
    }
  }
  auto scales = preprocessor.generate_scales(height, width);
  float* input = reinterpret_cast<float*>(
      engine.get_input_tensor(graph_name, input_name)->sys_data());
  for (auto scale : scales) {
    int scaled_h = ceil(height * scale);
    int scaled_w = ceil(width * scale);
    if ((scaled_h> maxinput_shape[2]) || (scaled_w > maxinput_shape[3])) {
        continue;
    }
    std::map<std::string, std::vector<int>> shape;
    shape[input_name] = {1, 3, scaled_h, scaled_w};
    // preprocess
    preprocessor.pnet_process(input, image, scaled_h, scaled_w);
    // reshape input tensor
    engine.reshape(graph_name, shape);
    // inference
    engine.process(graph_name);
    // get output tensors
    float* conf = reinterpret_cast<float*>(
        engine.get_output_tensor(graph_name, "prob1")->sys_data());
    float* coord = reinterpret_cast<float*>(
        engine.get_output_tensor(graph_name, "conv4-2")->sys_data());
    // postprocess
    auto output_shape = engine.get_output_shape(graph_name, "prob1");
    auto candidates = postprocessor.pnet_process_per_scale(
        conf, coord, scale, output_shape[2], output_shape[3]);
    // collect detected boxes
    boxes.insert(boxes.end(), candidates.begin(), candidates.end());
  }
  postprocessor.pnet_process(boxes);
  spdlog::info("Box number detected by PNet: {}", boxes.size());
  return std::move(boxes);
}

/**
 * @brief Run RNet, the second stage of MTCNN.
 *
 * @param engine        Engine instance
 * @param preprocessor  PreProcessor instance
 * @param postprocessor PostProcessor instance
 * @param pnet_output   Detected boxes by PNet
 * @param frame         Image
 * @return Detected boxes
 */
std::vector<FaceInfo> run_rnet(
    sail::Engine&          engine,
    PreProcessor&          preprocessor,
    PostProcessor&         postprocessor,
    std::vector<FaceInfo>& pnet_output,
    cv::Mat&               frame) {
  std::string graph_name("RNet");
  cv::Mat image;
  frame.copyTo(image);
  int box_num = pnet_output.size();
  auto input_name = engine.get_input_names(graph_name).front();
  auto input_shape = engine.get_input_shape(graph_name, input_name);
  auto conf_shape  = engine.get_output_shape(graph_name, "prob1");
  auto coord_shape = engine.get_output_shape(graph_name, "conv5-2");
  int image_size = input_shape[1] * input_shape[2] * input_shape[3];
  int input_size = box_num * image_size;
  float* input_data = new float[input_size];
  int conf_size = box_num * conf_shape[1];
  float* conf_data = new float[conf_size];
  int coord_size = box_num * coord_shape[1];
  float* coord_data = new float[coord_size];
  preprocessor.rnet_process(input_data, image, pnet_output,
                            input_shape[2], input_shape[3]);
  std::map<std::string, std::vector<int>> input_shapes;
  input_shapes[input_name] = input_shape;
  std::map<std::string, void*> input_tensors;
  int num_done = 0;
  int num_fix = input_shape[0];
  while (num_done < box_num) {
    if ((num_done + num_fix) > box_num) {
      num_fix = box_num - num_done;
    }
    int num_work = num_fix;
    // only batch size change for input shape
    input_shapes[input_name][0] = num_work;
    float* data = input_data + num_done * image_size;
    input_tensors[input_name] = reinterpret_cast<void*>(data);
    // inference
    engine.process(graph_name, input_shapes, input_tensors);
    // get output tensors
    float* conf = reinterpret_cast<float*>(
		engine.get_output_tensor(graph_name, "prob1")->sys_data());
    float* coord = reinterpret_cast<float*>(
		engine.get_output_tensor(graph_name, "conv5-2")->sys_data());
    // output data joint
    memcpy(conf_data + num_done * conf_shape[1],
           conf, num_work * conf_shape[1] * sizeof(float));
    memcpy(coord_data + num_done * coord_shape[1],
           coord, num_work * coord_shape[1] * sizeof(float));
    num_done += num_work;
  }
  // postprocess
  auto boxes = postprocessor.rnet_process(conf_data, coord_data, pnet_output);
  spdlog::info("Box number detected by RNet: {}", boxes.size());
  delete [] input_data;
  delete [] conf_data;
  delete [] coord_data;
  return std::move(boxes);
}

/**
 * @brief Run ONet, the third stage of MTCNN.
 *
 * @param engine        Engine instance
 * @param preprocessor  PreProcessor instance
 * @param postprocessor PostProcessor instance
 * @param pnet_output   Detected boxes by RNet
 * @param frame         Image
 * @return Detected boxes
 */
std::vector<FaceInfo> run_onet(
    sail::Engine&          engine,
    PreProcessor&          preprocessor,
    PostProcessor&         postprocessor,
    std::vector<FaceInfo>& rnet_output,
    cv::Mat&               frame) {
  std::string graph_name("ONet");
  cv::Mat image;
  frame.copyTo(image);
  int box_num = rnet_output.size();
  auto input_name = engine.get_input_names(graph_name).front();
  auto input_shape = engine.get_input_shape(graph_name, input_name);
  auto conf_shape = engine.get_output_shape(graph_name, "prob1");
  auto coord_shape = engine.get_output_shape(graph_name, "conv6-2");
  auto landmark_shape = engine.get_output_shape(graph_name, "conv6-3");
  int image_size = input_shape[1] * input_shape[2] * input_shape[3];
  int input_size = box_num * image_size;
  float* input_data = new float[input_size];
  int conf_size = box_num * conf_shape[1];
  float* conf_data = new float[conf_size];
  int coord_size = box_num * coord_shape[1];
  float* coord_data = new float[coord_size];
  int landmark_size = box_num * landmark_shape[1];
  float* landmark_data = new float[landmark_size];
  // input image joint
  preprocessor.onet_process(input_data, image, rnet_output,
                            input_shape[2], input_shape[3]);
  std::map<std::string, std::vector<int>> input_shapes;
  input_shapes[input_name] = input_shape;
  std::map<std::string, void*> input_tensors;
  int num_done = 0;
  int num_fix = input_shape[0];
  while (num_done < box_num) {
    if ((num_done + num_fix) > box_num) {
      num_fix = box_num - num_done;
    }
    int num_work = num_fix;
    // only batch size change for input shape
    input_shapes[input_name][0] = num_work;
    float* data = input_data + num_done * image_size;
    input_tensors[input_name] = reinterpret_cast<void*>(data);
    // inference
    engine.process(graph_name, input_shapes, input_tensors);
    // get output tensors
    float* conf = reinterpret_cast<float*>(
        engine.get_output_tensor(graph_name, "prob1")->sys_data());
    float* coord = reinterpret_cast<float*>(
        engine.get_output_tensor(graph_name, "conv6-2")->sys_data());
    float* landmark = reinterpret_cast<float*>(
        engine.get_output_tensor(graph_name, "conv6-3")->sys_data());
    // output data joint
    memcpy(conf_data + num_done * conf_shape[1],
           conf, num_work * conf_shape[1] * sizeof(float));
    memcpy(coord_data + num_done * coord_shape[1],
           coord, num_work * coord_shape[1] * sizeof(float));
    memcpy(landmark_data + num_done * landmark_shape[1],
           landmark, num_work * landmark_shape[1] * sizeof(float));
    num_done += num_work;
  }
  // postprocess
  auto boxes = postprocessor.onet_process(conf_data, coord_data,
                                          landmark_data, rnet_output);
  spdlog::info("Box number detected by ONet: {}", boxes.size());
  delete [] input_data;
  delete [] conf_data;
  delete [] coord_data;
  delete [] landmark_data;
  return boxes;
}

/**
 * @brief Print result of detection
 *
 * @param faceinfos Detected boxes
 * @param tpu_id    TPU ID
 */
void print_result(std::vector<FaceInfo>& faceinfos, int tpu_id, int loop) {
  if (faceinfos.size() == 0) {
    spdlog::error("No face was detected in this image!");
    return;
  }
  spdlog::info("---------  MTCNN DETECTION RESULT ON TPU {} OF LOOP {}---------", tpu_id, loop);
  for (size_t i = 0; i < faceinfos.size(); i++) {
    cv::Rect rc;
    rc.x = (faceinfos[i].bbox.x1) > 0 ? faceinfos[i].bbox.x1 : 0;
    rc.y = (faceinfos[i].bbox.y1) > 0 ? faceinfos[i].bbox.y1 : 0;
    rc.width = faceinfos[i].bbox.x2 - faceinfos[i].bbox.x1;
    rc.height = faceinfos[i].bbox.y2 - faceinfos[i].bbox.y1;
    spdlog::info("Face {} Box: [{}, {}, {}, {}], Score: {}",
                 i, rc.x, rc.y, rc.width, rc.height, faceinfos[i].bbox.score);
  }
}

void draw_result(std::vector<FaceInfo>& faceinfos, int tpu_id, int loop, cv::Mat &img)
{
    if (faceinfos.size() == 0) {
        spdlog::error("No face was detected in this image!");
        return;
    }
    spdlog::info("---------  MTCNN DETECTION RESULT ON TPU {} OF LOOP {}---------", tpu_id, loop);
    for (size_t i = 0; i < faceinfos.size(); i++) {
        cv::Rect rc;
        rc.x = (faceinfos[i].bbox.x1) > 0 ? faceinfos[i].bbox.x1 : 0;
        rc.y = (faceinfos[i].bbox.y1) > 0 ? faceinfos[i].bbox.y1 : 0;
        rc.width = faceinfos[i].bbox.x2 - faceinfos[i].bbox.x1;
        rc.height = faceinfos[i].bbox.y2 - faceinfos[i].bbox.y1;

        cv::rectangle(img, rc, cv::Scalar(0, 255,0), 2);
    }

    cv::imwrite("mtcnn.jpg", img.t());
}

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
bool det_mtcnn(
    const std::string& bmodel_path,
    const std::string& input_path,
    int                tpu_id,
    int                loops,
    const std::string& compare_path) {
  // init Engine to load bmodel and allocate input and output tensors
  sail::Engine engine(bmodel_path, tpu_id, sail::SYSIO);
  // init preprocessor and postprocessor
  PreProcessor preprocessor(127.5, 127.5, 127.5, 0.0078125);
  double threshold[3] = {0.5, 0.3, 0.7};
  PostProcessor postprocessor(threshold);
  auto reference = postprocessor.get_reference(compare_path);
  // read image
  cv::Mat frame = cv::imread(input_path);
  cv::Mat image = frame.t();

  bool status = true;
  for (int i = 0; i < loops; ++i) {
    // run PNet, the first stage of MTCNN
    auto boxes = run_pnet(engine, preprocessor, postprocessor, image);
    if (boxes.size() != 0) {
      // run RNet, the second stage of MTCNN
      boxes = run_rnet(engine, preprocessor, postprocessor, boxes, image);
      if (boxes.size() != 0) {
        // run ONet, the third stage of MTCNN
        boxes = run_onet(engine, preprocessor, postprocessor, boxes, image);
      }
    }
    // print_result
    if (postprocessor.compare(reference, boxes)) {
      print_result(boxes, tpu_id, i);
#if 0
      draw_result(boxes, tpu_id, i, image);
#endif
    } else {
      status = false;
      break;
    }
  }
  return status;
}

/// An example of inference for dynamic model, whose input shapes may change.
int main(int argc, char** argv) {
  const char* opt_strings = "b:i:t:l:c:";
  const struct option long_opts[] = {
    {"bmodel", required_argument, nullptr, 'b'},
    {"input", required_argument, nullptr, 'i'},
    {"tpu_id", required_argument, nullptr, 't'},
    {"loops", required_argument, nullptr, 'l'},
    {"compare", required_argument, nullptr, 'c'},
    {0, 0, 0, 0}
  };
  std::string bmodel_path;
  std::string input_path;
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
      case 't':
        tpu_id = std::stoi(optarg);
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
      tpu_id < 0 || loops <= 0) {
    std::string usage("Usage: {} --bmodel bmodel_path --input input_path");
    usage += " [--tpu_id tpu_id(default:0)] [--loops loops_num(default:1)]";
    usage += " [--compare verify.ini]";
    spdlog::info(usage.c_str(), argv[0]);
    return -1;
  }
  if (!file_exists(input_path)) {
    spdlog::error("File not exists: {}", input_path);
    return -2;
  }
  // load bmodel and do inference
  bool ret;
  while(loops-- > 0) {
      ret = det_mtcnn(bmodel_path, input_path, tpu_id, 5, compare_path);
      if (!ret) return ret;
  }
  return 0;
}
