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
   * @brief Constructor.
   *
   * @param height        Height of network input
   * @param width         Width of network input
   * @param interpolation Interpolation method of opencv resize
   * @param mean_0        Bias of channel B
   * @param mean_0        Bias of channel G
   * @param mean_0        Bias of channel R
   * @param scale_factor  Scale factor of all channel
   * @param convert_color Flag of whether to do BGR2RGB
   */
  PreProcessor(
      int   height,
      int   width,
      float mean_0 = 103.94,
      float mean_1 = 116.78,
      float mean_2 = 123.68);

  /**
   * @brief Destructor.
   */
  ~PreProcessor() {}

  /**
   * @brief Execution function of preprocessing.
   *
   * @param input Input data pointer in system memory
   * @param frame Original image
   */
  void process(float* input, cv::Mat& frame);

 private:
  int height_;
  int width_;
  std::vector<float> negative_mean_;
};

class PostProcessor {
 public:
  /**
   * @brief Constructor.
   *
   * @param batch_size Batch size
   * @param class_num  Class number
   * @param top_k      Number of classification result
   */
  PostProcessor(size_t batch_size, size_t class_num, size_t top_k);

  /**
   * @brief Destructor.
   */
  ~PostProcessor() {}

  /**
   * @brief Execution function of postprocessing
   *
   * @param input The output data of inference
   * @return The classification result of each input
   *     @retval Ex:[[20,3,4,5,7],[6,30,1,2,3]] the index of classes
   */
  std::vector<std::vector<int>> process(float* input);

  /**
   * @brief Get correct result from given file.
   *
   * @param compare_path Path to correct result file
   * @return correct result
   */
  std::map<std::string, std::vector<int>> get_reference(
      const std::string& compare_path);

  /**
   * @brief Compare result.
   *
   * @param reference Correct result
   * @param result    Output result
   * @param dtype     Data type of model
   * @return Program state
   *     @retval true  Success
   *     @retval false Failure
   */
  bool compare(
    std::map<std::string, std::vector<int>>& reference,
    std::vector<int>&                        result,
    const std::string&                       dtype);

 private:
  size_t batch_size_;
  size_t class_num_;
  size_t top_k_;
};
