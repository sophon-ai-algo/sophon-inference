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

#include <vector>
#include <string>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

/**
 * @brief Judge if a file exists..
 *
 * @param file_path Path to the file
 * @return True for exist, false for not.
 */
bool file_exists(const std::string& file_path);

class PreProcessor {
 public:
  /**
   * @brief Construct an Yolov3 preprocess object
   *
   * using default parameters
   */
  PreProcessor();
  /**
   * @brief Construct an Yolov3 preprocess object
   *
   * @param height  The height of network input
   * @param width  The width of network input
   */
  PreProcessor(int height, int width);
  virtual ~PreProcessor() {}
  /**
   * @brief The execution function of Yolov3 preprocess
   *
   * @param input The input data system mem pointer
   * @param frame The original image
   */
  virtual void process(float* input, cv::Mat& frame);
  /**
   * @brief The execution function of Yolov3 preprocess
   *
   * @param input The input data system mem pointer
   * @param frame The original image
   */
  virtual void processv2(float* input, cv::Mat& frame);

 private:
  int height_;
  int width_;
};

/**
 * @brief Struct to hold detetion result.
 */
struct DetectRect {
  int class_id;
  float score;
  float x1;
  float y1;
  float x2;
  float y2;

  bool operator==(const DetectRect& t) const {
    return (t.class_id == this->class_id &&
            std::abs(t.x1 - this->x1) < 2 &&
            std::abs(t.y1 - this->y1) < 2 &&
            std::abs(t.x2 - this->x2) < 2 &&
            std::abs(t.y2 - this->y2) < 2 &&
            std::abs(t.score - this->score) < 1.8e-1);
  }
};

class PostProcessor {
 public:
  PostProcessor(float threshold);

  std::vector<DetectRect> process(
      float*    output,
      int       detected_number,
      int       height,
      int       width);

  /**
   * @brief Get correct result from given file.
   *
   * @param compare_path Path to correct result file
   * @return correct result
   */
  std::vector<DetectRect> get_reference(const std::string& compare_path);

  /**
   * @brief Compare result.
   *
   * @param reference Correct result
   * @param result    Output result
   * @param dtype     Data type of model
   * @return correct result
   */
  bool compare(
      std::vector<DetectRect>& reference,
      std::vector<DetectRect>& result,
      int                      loop_id);

 private:
  float threshold_;
};
