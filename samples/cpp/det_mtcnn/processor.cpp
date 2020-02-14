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

#include <algorithm>
#include "inireader.hpp"
#include "spdlog/spdlog.h"
#include "processor.h"
#include <iostream>

PreProcessor::PreProcessor()
    : scale_factor_(0.0078125), face_factor_(0.89), min_size_(40) {
  float negative_mean[] = {-127.5, -127.5, -127.5};
  negative_mean_.assign(negative_mean, negative_mean + 3);
}

PreProcessor::PreProcessor(
    float mean_0,
    float mean_1,
    float mean_2,
    float scale_factor)
    : scale_factor_(scale_factor), face_factor_(0.89), min_size_(40) {
  float negative_mean[] = {-mean_0, -mean_1, -mean_2};
  negative_mean_.assign(negative_mean, negative_mean + 3);
}

std::vector<double> PreProcessor::generate_scales(int height, int width) {
  int min_hw = std::min(height, width);
  double m_scale = 12.0 / min_size_;
  min_hw *= m_scale;
  std::vector<double> scales;
  int factor_count = 0;
  while (min_hw >= 12) {
    scales.push_back(m_scale * std::pow(face_factor_, factor_count));
    min_hw *= face_factor_;
    ++factor_count;
  }
  return std::move(scales);
}

void PreProcessor::pnet_process(
    float*   input,
    cv::Mat& frame,
    int      height,
    int      width) {
  cv::Mat resized;
  cv::resize(frame, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
  resized.convertTo(resized, CV_32FC3);
  resized += cv::Scalar(negative_mean_[0], negative_mean_[1], negative_mean_[2]);
  resized *= scale_factor_;
  std::vector<cv::Mat> input_channels;
  int size_per_channel = height * width;
  for (int i = 0; i < 3; ++i) {
    cv::Mat channel(height, width, CV_32FC1, input);
    input_channels.push_back(channel);
    input += size_per_channel;
  }
  cv::split(resized, input_channels);
}

void PreProcessor::rnet_process(
    float*                 input,
    cv::Mat&               frame,
    std::vector<FaceInfo>& boxes,
    int                    height,
    int                    width) {
  padding(input, frame, boxes, height, width);
}

void PreProcessor::onet_process(
    float*                 input,
    cv::Mat&               frame,
    std::vector<FaceInfo>& boxes,
    int                    height,
    int                    width) {
  padding(input, frame, boxes, height, width);
}

void PreProcessor::padding(
    float*                 input,
    cv::Mat&               frame,
    std::vector<FaceInfo>& boxes,
    int                    height,
    int                    width) {
  for (size_t i = 0; i < boxes.size(); ++i) {
    FaceInfo box = boxes[i];
    box.bbox.x1 = (boxes[i].bbox.x1 < 0) ? 0 : boxes[i].bbox.x1;
    box.bbox.y1 = (boxes[i].bbox.y1 < 0) ? 0 : boxes[i].bbox.y1;
    box.bbox.x2 = (boxes[i].bbox.x2 > frame.cols - 1) ?
                   frame.cols - 1: boxes[i].bbox.x2;
    box.bbox.y2 = (boxes[i].bbox.y2 > frame.rows - 1) ?
                   frame.rows - 1: boxes[i].bbox.y2;
    int pad_left = std::abs(box.bbox.x1 - boxes[i].bbox.x1);
    int pad_top = std::abs(box.bbox.y1 - boxes[i].bbox.y1);
    int pad_right = std::abs(box.bbox.x2 - boxes[i].bbox.x2);
    int pad_bottom = std::abs(box.bbox.y2 - boxes[i].bbox.y2);
    cv::Mat crop_img = frame(cv::Range(box.bbox.y1, box.bbox.y2 + 1),
                             cv::Range(box.bbox.x1, box.bbox.x2 + 1));
    cv::copyMakeBorder(crop_img, crop_img, pad_top, pad_bottom, pad_left,
                       pad_right, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::Mat resized;
    cv::resize(crop_img, resized, cv::Size(width, height), 0, 0);
    resized.convertTo(resized, CV_32FC3);
    resized += cv::Scalar(negative_mean_[0],
                          negative_mean_[1], negative_mean_[2]);
    resized *= scale_factor_;
    std::vector<cv::Mat> input_channels;
    int size_per_channel = height * width;
    float* data = input + i * 3 * height * width;
    for (int i = 0; i < 3; ++i) {
      cv::Mat channel(height, width, CV_32FC1, data);
      input_channels.push_back(channel);
      data += size_per_channel;
    }
    cv::split(resized, input_channels);
  }
}

PostProcessor::PostProcessor(double* threshold) {
  threshold_.assign(threshold, threshold + 3);
}

static bool compare_bbox(const FaceInfo& a, const FaceInfo& b) {
  return a.bbox.score > b.bbox.score;
}

std::vector<FaceInfo> PostProcessor::nms(
    std::vector<FaceInfo>& boxes,
    float                  threshold,
    char                   method_type) {
  std::vector<FaceInfo> boxes_nms;
  std::sort(boxes.begin(), boxes.end(), compare_bbox);
  int32_t select_idx = 0;
  int32_t num_box = static_cast<int32_t>(boxes.size());
  std::vector<int32_t> mask_merged(num_box, 0);
  bool all_merged = false;
  while (!all_merged) {
    while (select_idx < num_box && mask_merged[select_idx] == 1) {
      ++select_idx;
    }
    if (select_idx == num_box) {
      all_merged = true;
      continue;
    }
    boxes_nms.push_back(boxes[select_idx]);
    mask_merged[select_idx] = 1;
    FaceRect select_box = boxes[select_idx].bbox;
    float area1 = static_cast<float>((select_box.x2 - select_box.x1 + 1) *
                                     (select_box.y2 - select_box.y1 + 1));
    float x1 = static_cast<float>(select_box.x1);
    float y1 = static_cast<float>(select_box.y1);
    float x2 = static_cast<float>(select_box.x2);
    float y2 = static_cast<float>(select_box.y2);
    ++select_idx;
    for (int32_t i = select_idx; i < num_box; ++i) {
      if (mask_merged[i] == 1) {
        continue;
      }
      FaceRect& box_i = boxes[i].bbox;
      float x = std::max<float>(x1, static_cast<float>(box_i.x1));
      float y = std::max<float>(y1, static_cast<float>(box_i.y1));
      float w = std::min<float>(x2, static_cast<float>(box_i.x2)) - x + 1;
      float h = std::min<float>(y2, static_cast<float>(box_i.y2)) - y + 1;
      if (w <= 0 || h <= 0) {
        continue;
      }
      float area2 = static_cast<float>((box_i.x2 - box_i.x1 + 1) *
                                       (box_i.y2 - box_i.y1 + 1));
      float area_intersect = w * h;
      switch (method_type) {
        case 'u':
          if (static_cast<float>(area_intersect) /
                (area1 + area2 - area_intersect) > threshold) {
            mask_merged[i] = 1;
          }
          break;
        case 'm':
          if (static_cast<float>(area_intersect) /
                std::min(area1, area2) > threshold) {
            mask_merged[i] = 1;
          }
          break;
        default:
          break;
      }
    }
  }
  return std::move(boxes_nms);
}

std::vector<FaceInfo> PostProcessor::box_regress(
    std::vector<FaceInfo>& boxes,
    int                    stage) {
  std::vector<FaceInfo> candidates;
  for (size_t id = 0; id < boxes.size(); ++id) {
    FaceRect rect;
    FaceInfo temp;
    float regw = boxes[id].bbox.x2 - boxes[id].bbox.x1 + 1;
    float regh = boxes[id].bbox.y2 - boxes[id].bbox.y1 + 1;
    rect.x1 = boxes[id].bbox.x1 + regw * boxes[id].regression[0] - 1;
    rect.y1 = boxes[id].bbox.y1 + regh * boxes[id].regression[1] - 1;
    rect.x2 = boxes[id].bbox.x2 + regw * boxes[id].regression[2] - 1;
    rect.y2 = boxes[id].bbox.y2 + regh * boxes[id].regression[3] - 1;
    rect.score = boxes[id].bbox.score;
    temp.bbox = rect;
    temp.regression = boxes[id].regression;
    if (stage == 3) {
      temp.face_pts = boxes[id].face_pts;
    }
    temp.imgid = boxes[id].imgid;
    candidates.push_back(temp);
  }
  return std::move(candidates);
}

void PostProcessor::bbox2square(std::vector<FaceInfo>& boxes) {
  for (size_t i = 0; i < boxes.size(); ++i) {
    float w = boxes[i].bbox.x2 - boxes[i].bbox.x1 + 1;
    float h = boxes[i].bbox.y2 - boxes[i].bbox.y1 + 1;
    float side = std::max<float>(w, h);
    boxes[i].bbox.x1 += (w - side) * 0.5;
    boxes[i].bbox.y1 += (h - side) * 0.5;
    boxes[i].bbox.x2 = std::round(boxes[i].bbox.x1 + side - 1);
    boxes[i].bbox.y2 = std::round(boxes[i].bbox.y1 + side - 1);
    boxes[i].bbox.x1 = std::round(boxes[i].bbox.x1);
    boxes[i].bbox.y1 = std::round(boxes[i].bbox.y1);
  }
}

std::vector<FaceInfo> PostProcessor::pnet_process_per_scale(
    float* conf,
    float* coord,
    float  scale,
    int    height,
    int    width) {
  std::vector<FaceInfo> candidates;
  int stride = 2;
  int cell_size = 12;
  int offset = height * width;
  const float* confidence = conf + offset;
  // generate bounding boxes
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int index = y * width + x;
      if (confidence[index] >= threshold_[0]) {
        FaceRect rect;
        rect.y1 = std::round(y * stride / scale);
        rect.x1 = std::round(x * stride / scale);
        rect.y2 = std::round((y * stride + cell_size - 1) / scale);
        rect.x2 = std::round((x * stride + cell_size - 1) / scale);
        rect.score = confidence[index];
        FaceInfo face;
        face.imgid = 0;
        face.bbox = rect;
        face.regression = cv::Vec4f(coord[offset + index], coord[index],
                                    coord[3 * offset + index],
                                    coord[2 * offset + index]);
        candidates.push_back(face);
      }
    }
  }
  // NMS
  std::vector<FaceInfo> boxes = nms(candidates, 0.5, 'u');
  return std::move(boxes);
}

void PostProcessor::pnet_process(std::vector<FaceInfo>& boxes) {
  if (boxes.size() > 0) {
    boxes = nms(boxes, 0.7, 'u');
    boxes = box_regress(boxes, 1);
    bbox2square(boxes);
  }
}

std::vector<FaceInfo> PostProcessor::rnet_process(
    float*                 conf,
    float*                 coord,
    std::vector<FaceInfo>& boxes) {
  std::vector<FaceInfo> candidates;
  int box_num = boxes.size();
  for (int i = 0; i < box_num; ++i) {
    if (conf[i * 2 + 1] > threshold_[1]) {
      FaceRect rect;
      rect.x1 = boxes[i].bbox.x1;
      rect.y1 = boxes[i].bbox.y1;
      rect.x2 = boxes[i].bbox.x2;
      rect.y2 = boxes[i].bbox.y2;
      rect.score = conf[i * 2 + 1];
      FaceInfo face;
      face.bbox = rect;
      face.regression = cv::Vec4f(coord[4 * i + 1], coord[4 * i + 0],
                                  coord[4 * i + 3], coord[4 * i + 2]);
      face.imgid = boxes[i].imgid;
      candidates.push_back(face);
    }
  }
  candidates = nms(candidates, 0.7, 'u');
  candidates = box_regress(candidates, 2);
  bbox2square(candidates);
  return std::move(candidates);
}

std::vector<FaceInfo> PostProcessor::onet_process(
    float*                 conf,
    float*                 coord,
    float*                 landmark,
    std::vector<FaceInfo>& boxes) {
  std::vector<FaceInfo> candidates;
  int box_num = boxes.size();
  for (int i = 0; i < box_num; ++i) {
    if (conf[i * 2 + 1] > threshold_[2]) {
      FaceRect rect;
      rect.x1 = boxes[i].bbox.x1;
      rect.y1 = boxes[i].bbox.y1;
      rect.x2 = boxes[i].bbox.x2;
      rect.y2 = boxes[i].bbox.y2;
      rect.score = conf[i * 2 + 1];
      FaceInfo face;
      face.bbox = rect;
      face.regression = cv::Vec4f(coord[4 * i + 1], coord[4 * i + 0],
                                  coord[4 * i + 3], coord[4 * i + 2]);
      face.imgid = boxes[i].imgid;
      FacePts pts;
      float w = rect.x2 - rect.x1 + 1;
      float h = rect.y2 - rect.y1 + 1;
      for (int j = 0; j < 5; ++j) {
        pts.x[j] = rect.x1 + landmark[j + 10 * i + 5] * w - 1;
        pts.y[j] = rect.y1 + landmark[j + 10 * i] * h - 1;
      }
      face.face_pts = pts;
      candidates.push_back(face);
    }
  }
  candidates = box_regress(candidates, 3);
  candidates = nms(candidates, 0.7, 'm');
  return std::move(candidates);
}

std::vector<FaceRect> PostProcessor::get_reference(
    const std::string& compare_path) {
  std::vector<FaceRect> reference;
  if (!compare_path.empty()) {
    INIReader reader(compare_path);
    if (reader.ParseError()) {
      spdlog::error("Can't load reference file: {}!", compare_path);
      std::terminate();
    }
    for (int i = 0 ; i < 42; ++i) {
      FaceRect box;
      std::string section("face_");
      section += std::to_string(i);
      box.x1 = reader.GetReal(section, "x1", 0.0);
      box.y1 = reader.GetReal(section, "y1", 0.0);
      box.x2 = reader.GetReal(section, "x2", 0.0);
      box.y2 = reader.GetReal(section, "y2", 0.0);
      box.score = reader.GetReal(section, "score", 0.0);
      reference.push_back(box);
    }
  }
  return std::move(reference);
}

bool PostProcessor::compare(
    std::vector<FaceRect>& reference,
    std::vector<FaceInfo>& result) {
  if (reference.empty()) {
    return true;
  }
  if (reference.size() != result.size()) {
    spdlog::error("Expected deteted face number is {}, but detected {}!",
                  reference.size(), result.size());
    return false;
  }
  bool ret = true;
  for (int i = 0; i < result.size(); ++i) {
    auto& box = result[i].bbox;
    auto& ref = reference[i];
    if (!(box == ref)) {
      spdlog::error("Compare failed! Expect: Box: [{}, {}, {}, {}], Score: {}",
                    ref.x1, ref.y1, ref.x2, ref.y2, ref.score);
      spdlog::info("Result Box: [{}, {}, {}, {}], Score: {}",
                    box.x1, box.y1, box.x2, box.y2, box.score);
      ret = false;
    }
  }
  return ret;
}

PostProcessor::~PostProcessor() {}
