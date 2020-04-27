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

#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

/**
 * @brief Judge if a file exists..
 *
 * @param file_path Path to the file
 * @return True for exist, false for not.
 */
bool file_exists(const std::string& file_path);

/**
 * @brief Information of face bounding box.
 */
typedef struct FaceRect {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;

  bool operator==(const FaceRect& t) const {
    return (std::abs(t.x1 - this->x1) < 1e-2 &&
            std::abs(t.y1 - this->y1) < 1e-2 &&
            std::abs(t.x2 - this->x2) < 1e-2 &&
            std::abs(t.y2 - this->y2) < 1e-2 &&
            std::abs(t.score - this->score) < 1e-4);
  }
} FaceRect;

/**
 * @brief Information of face landmark.
 */
typedef struct FacePts {
  float x[5];
  float y[5];

  bool operator==(const FacePts &t) const {
    bool ret = true;
    for (int i = 0; i < 5; ++i) {
      if (t.x[i] != this->x[i] || t.y[i] != this->y[i]) {
         ret = false;
         break;
      }
    }
    return ret;
  }
} FacePts;

/**
 * @brief Information of detected face.
 */
typedef struct FaceInfo {
  FaceRect bbox;
  cv::Vec4f regression;
  FacePts face_pts;
  double roll;
  double pitch;
  double yaw;
  double distance;
  int imgid;

  bool operator==(const FaceInfo &t) const {
    return (t.bbox == this->bbox &&
        t.regression == this->regression &&
        t.roll == this->roll &&
        t.pitch == this->pitch &&
        t.yaw == this->yaw &&
        t.distance == this->distance &&
        t.imgid == this->imgid);
  }
} FaceInfo;

/**
 * @brief Preprocessor for MTCNN.
 */
class PreProcessor {
 public:
  /**
   * @brief  Default constructor.
   */
  PreProcessor();
  /**
   * @brief Constructor given configurations.
   *
   * @param mean_0       Bias of channel B
   * @param mean_0       Bias of channel G
   * @param mean_0       Bias of channel R
   * @param scale_factor Scale factor of all channels
   */
  PreProcessor(
      float mean_0,
      float mean_1,
      float mean_2,
      float scale_factor);

  /**
   * @brief Destructor.
   */
  virtual ~PreProcessor() {}

  /**
   * @brief Generate image pyramid scale factors.
   *
   * @param height Image height
   * @param width  Image width
   */
  std::vector<double> generate_scales(int height, int width);

  /**
   * @brief Preprocess function of PNet.
   *
   * @param input  Input data pointer in system memory
   * @param frame  Original image
   * @param height Height of PNet input image
   * @param width  Width of PNet input image
   */
  void pnet_process(float* input, cv::Mat& frame, int height, int width);

  /**
   * @brief Preprocess function of RNet.
   *
   * @param input       Input data pointer in system memory
   * @param frame       Original image
   * @param pnet_output Detected boxes by PNet
   * @param height      Height of RNet input image
   * @param width       Width of RNet input image
   */
  void rnet_process(
    float*                 input,
    cv::Mat&               frame,
    std::vector<FaceInfo>& boxes,
    int                    height,
    int                    width);

  /**
   * @brief Preprocess function of ONet.
   *
   * @param input       Input data pointer in system memory
   * @param frame       Original image
   * @param rnet_output Detected boxes by RNet
   * @param height      Height of ONet input image
   * @param width       Width of ONet input image
   */
  void onet_process(
    float*                 input,
    cv::Mat&               frame,
    std::vector<FaceInfo>& boxes,
    int                    height,
    int                    width);

 private:
  /**
   * @brief Padding function for bounding boxes.
   *
   * @param input  Input data pointer in system memory
   * @param frame  Original image
   * @param boxes  Input face bounding boxes
   * @param height Height of network input
   * @param width  Width of network input
   */
  void padding(
    float*                 input,
    cv::Mat&               frame,
    std::vector<FaceInfo>& boxes,
    int                    height,
    int                    width);

  std::vector<float> negative_mean_;
  double scale_factor_;
  double face_factor_;
  int min_size_;
};

/**
 * @brief A wrapper that performs the caffe-mtcnn postprocess.
 */
class PostProcessor {
 public:
  /**
  * @brief Constructor.
  *
  * @param threshold Array of mtcnn threshold
  */
  explicit PostProcessor(double* threshold);

  /**
   * @brief Postprocess function of PNet for each scale.
   *
   * @param conf   Confidence data
   * @param coord  Regression data
   * @param scale  Scale factor of image
   * @param height Output height of PNet
   * @param width  Output width of PNet
   * @return Detected faces
   */
  std::vector<FaceInfo> pnet_process_per_scale(
      float* conf,
      float* coord,
      float  scale,
      int    height,
      int    width);

  /**
   * @brief Postprocess function of PNet for each scale.
   *
   * @param boxes Regressed face bbox
   */
  void pnet_process(std::vector<FaceInfo>& boxes);

  /**
   * @brief Postprocess function of RNet.
   *
   * @param conf  Confidence data
   * @param coord Regression data
   * @param boxes Regressed face bbox
   * @return Detected faces
   */
  std::vector<FaceInfo> rnet_process(
      float*                 conf,
      float*                 coord,
      std::vector<FaceInfo>& boxes);

  /**
   * @brief Postprocess function of ONet.
   *
   * @param conf     Confidence data
   * @param coord    Regression data
   * @param landmark Landmark data
   * @param boxes    Regressed face bounding boxes
   * @return Detected faces
   */
  std::vector<FaceInfo> onet_process(
      float*                 conf,
      float*                 coord,
      float*                 landmark,
      std::vector<FaceInfo>& boxes);

  /**
   * @brief Get correct result from given file.
   *
   * @param compare_path Path to correct result file
   * @return correct result
   */
  std::vector<FaceRect> get_reference(const std::string& compare_path);

  /**
   * @brief Compare result.
   *
   * @param reference Correct result
   * @param result    Output result
   * @param dtype     Data type of model
   * @return correct result
   */
  bool compare(
    std::vector<FaceRect>& reference,
    std::vector<FaceInfo>& result);

  /**
   * @brief Destructor.
   */
  virtual ~PostProcessor();

 private:
  /**
   * @brief NMS function.
   *
   * @param boxes       Input face bounding boxes
   * @param thresh      Threshold
   * @param method_type Calculation method of NMS
   * @return Faces after NMS
   */
  std::vector<FaceInfo> nms(
      std::vector<FaceInfo>& boxes,
      float                  thresh,
      char                   method_type);

  /**
   * @brief Regression function for bounding boxes.
   *
   * @param boxes  Input face bounding boxes
   * @param stage  Number of stage
   * @return Face bounding boxes after regression
   */
  std::vector<FaceInfo> box_regress(
      std::vector<FaceInfo>& boxes,
      int                    stage);

  /**
   * @brief Square the bounding boxes.
   *
   * @param bboxes  Face bounding boxes
   */
  void bbox2square(std::vector<FaceInfo>& boxes);

  std::vector<double> threshold_;
};
