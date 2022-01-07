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

#include <cmath>
#include <algorithm>
#include <fstream>
#include "inireader.hpp"
#include "spdlog/spdlog.h"
#include "processor.h"

bool file_exists(const std::string& file_path) {
  std::ifstream f(file_path.c_str());
  return f.good();
}

PreProcessor::PreProcessor() : height_(416), width_(416) {}

PreProcessor::PreProcessor(int height, int width)
    : height_(height), width_(width) {}

void PreProcessor::process(float* input, cv::Mat& frame) {
  cv::Mat resize_img;
  cv::Mat img_roi;
  cv::Mat back = cv::Mat(cv::Size(width_, height_),
                 CV_32FC3, cv::Scalar(127.5, 127.5, 127.5));
  int w = frame.cols;
  int h = frame.rows;
  if (w > h) {
    float ratio = (h * 1.0) / w;
    cv::resize(frame, resize_img, cv::Size(width_, width_ * ratio),
               0, 0);
    img_roi = back(cv::Rect(0, ((height_ - width_ * ratio) / 2),
                   resize_img.cols, resize_img.rows));
  } else {
    float ratio = (w * 1.0) / h;
    cv::resize(frame, resize_img, cv::Size(height_ * ratio, height_),
               0, 0);
    img_roi = back(cv::Rect(((width_ - height_ * ratio) / 2), 0,
              resize_img.cols, resize_img.rows));
  }
  resize_img.convertTo(resize_img, CV_32FC3);
  resize_img.copyTo(img_roi);
  back /= 255;
  cv::cvtColor(back, back, cv::COLOR_BGR2RGB);
  std::vector<cv::Mat> input_channels;
  int size_per_channel = height_ * width_;
  for (int i = 0; i < 3; ++i) {
    cv::Mat channel(height_, width_, CV_32FC1, input);
    input_channels.push_back(channel);
    input += size_per_channel;
  }
  cv::split(back, input_channels);
}

void PreProcessor::processv2(float* input, cv::Mat& frame) {
  cv::Mat resize_img;
  cv::resize(frame, resize_img, cv::Size(width_, height_), 0, 0);
  resize_img.convertTo(resize_img, CV_32FC3);
  resize_img /= 255;
  cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);
  std::vector<cv::Mat> input_channels;
  int size_per_channel = height_ * width_;
  for (int i = 0; i < 3; ++i) {
    cv::Mat channel(height_, width_, CV_32FC1, input);
    input_channels.push_back(channel);
    input += size_per_channel;
  }
  cv::split(resize_img, input_channels);
}

PostProcessor::PostProcessor(float threshold) : threshold_(threshold) {
}

std::vector<DetectRect> PostProcessor::process(
    float* output,
    int    detected_number,
    int    height,
    int    width) {
  std::vector<DetectRect> result;
  for (int i = 0; i < detected_number; ++i) {
    float score = output[i * 7 + 2];
    if (score > threshold_) {
      DetectRect rect;
      float x_center = output[i * 7 + 3] * width;
      float y_center = output[i * 7 + 4] * height;
      float w = output[i * 7 + 5] * width;
      float h = output[i * 7 + 6] * height;
      rect.class_id = output[i * 7 + 1];
      rect.score = score;
      rect.x1 = x_center - w / 2;
      rect.x2 = x_center + w / 2;
      rect.y1 = y_center - h / 2;
      rect.y2 = y_center + h / 2;
      result.push_back(rect);
    }
  }
  return std::move(result);
}

std::vector<DetectRect> PostProcessor::get_reference(
    const std::string& compare_path) {
  std::vector<DetectRect> reference;
  if (!compare_path.empty()) {
    INIReader reader(compare_path);
    if (reader.ParseError()) {
      spdlog::error("Can't load reference file: {}!", compare_path);
      std::terminate();
    }
    int num = reader.GetInteger("summary", "num", 0);
    for (int i = 0 ; i < num; ++i) {
      DetectRect box;
      std::string section("object_");
      section += std::to_string(i);
      box.x1 = reader.GetReal(section, "x1", 0.0);
      box.y1 = reader.GetReal(section, "y1", 0.0);
      box.x2 = reader.GetReal(section, "x2", 0.0);
      box.y2 = reader.GetReal(section, "y2", 0.0);
      box.score = reader.GetReal(section, "score", 0.0);
      box.class_id = reader.GetReal(section, "category", 0);
      reference.push_back(box);
    }
  }
  return std::move(reference);
}

bool PostProcessor::compare(
    std::vector<DetectRect>& reference,
    std::vector<DetectRect>& result,
    int                      loop_id) {
  if (reference.empty()) {
    spdlog::info("No verify_files file or verify_files err.");
    return true;
  }
  if (loop_id > 0) {
    return true;
  }
  if (reference.size() != result.size()) {
    spdlog::error("Expected deteted number is {}, but detected {}!",
                  reference.size(), result.size());
    return false;
  }
  bool ret = true;
  std::string message("Category: {}, Score: {}, Box: [{}, {}, {}, {}]");
  std::string fail_info("Compare failed! Expect: ");
  fail_info += message;
  std::string ret_info("Result Box: ");
  ret_info += message;
  for (int i = 0; i < result.size(); ++i) {
    auto& box = result[i];
    auto& ref = reference[i];
    if (!(box == ref)) {
      spdlog::error(fail_info.c_str(), ref.class_id, ref.score,
                    ref.x1, ref.y1, ref.x2, ref.y2);
      spdlog::info(ret_info.c_str(), box.class_id, box.score,
                   box.x1, box.y1, box.x2, box.y2);
      ret = false;
    }
  }
  return ret;
}
