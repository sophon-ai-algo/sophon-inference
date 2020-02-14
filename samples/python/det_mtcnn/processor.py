""" Copyright 2016-2022 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

"""Preprocessor and postprocessor of mtcnn
"""
import cv2
import json
import numpy as np

class PreProcessor():

  def __init__(self, mean, scale_factor, face_factor=0.89, min_size=40):
    """ Constructor.

    Args:
      mean: List of mean value of each channel
      scale_factor: Scale value to preprocess input image
      face_factor: Initial value to generate image pyramid scale factors
      min_size: Minmum size of detection
    """
    self.mean = mean
    self.scale_factor = scale_factor
    self.face_factor = face_factor
    self.min_size = min_size

  def generate_scales(self, height, width):
    """ Generate image pyramid scale factors.

    Args:
      height: Image height
      width: Image width

    Returns:
      A list of scale factors
    """
    min_hw = min(height, width)
    m_scale = 12.0 / self.min_size
    min_hw = int(min_hw * m_scale)
    scales = []
    factor_count = 0
    while min_hw >= 12:
      scales.append(m_scale * pow(self.face_factor, factor_count))
      min_hw = int(min_hw * self.face_factor)
      factor_count += 1
    return scales

  def pnet_process(self, image, height, width):
    """ Preprocess function of PNet.

    Args:
      image: Input image
      height: Expected image height
      width: Expected image width

    Returns:
      4-dim ndarray
    """
    image = cv2.resize(image, (width, height)).astype(np.float32)
    image[:, :, 0] -= self.mean[0]
    image[:, :, 1] -= self.mean[1]
    image[:, :, 2] -= self.mean[2]
    image *= self.scale_factor
    image = np.transpose(image, (2, 0, 1))
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    return image.copy()

  def rnet_process(self, image, boxes, height, width):
    """ Preprocess function of RNet

    Args:
      image: Input image
      boxes: Detected boxes by PNet
      height: Expected image height
      width: Expected image width

    Returns:
      4-dim ndarray
    """
    data = self.__padding(image, boxes, height, width)
    return data

  def onet_process(self, image, boxes, height, width):
    """ Preprocess function of ONet

    Args:
      image: Input image
      boxes: Detected boxes by RNet
      height: Expected image height
      width: Expected image width

    Returns:
      4-dim ndarray
    """
    data = self.__padding(image, boxes, height, width)
    return data

  def __padding(self, image, boxes, height, width):
    """ Padding function for bounding boxes.

    Args:
      image: Input image
      boxes: Detected bounding boxes
      height: Expected image height
      width: Expected image width

    Returns:
      4-dim ndarray
    """
    temp = boxes[:, :4].astype(np.int)
    y1 = np.where(temp[:, 0] < 0)[0]
    if len(y1) > 0:
      temp[y1, 0] = 0
    x1 = np.where(temp[:, 1] < 0)[0]
    if len(x1) > 0:
      temp[x1, 0] = 0
    y2 = np.where(temp[:, 2] > image.shape[0] - 1)[0]
    if len(y2) > 0:
      temp[y2, 0] = image.shape[0] - 1
    x2 = np.where(temp[:, 3] > image.shape[1] - 1)[0]
    if len(x2) > 0:
      temp[x2, 0] = image.shape[1] - 1
    pad_top = np.abs(temp[:, 0] - boxes[:, 0]).astype(np.int)
    pad_left = np.abs(temp[:, 1] - boxes[:, 1]).astype(np.int)
    pad_bottom = np.abs(temp[:, 2] - boxes[:, 2]).astype(np.int)
    pad_right = np.abs(temp[:, 3] - boxes[:, 3]).astype(np.int)
    input_data = np.empty([boxes.shape[0], 3, height, width], dtype=np.float32)
    for i in range(boxes.shape[0]):
      crop_img = image[temp[i, 0]:temp[i, 2] + 1, temp[i, 1]:temp[i, 3] + 1, :]
      crop_img = cv2.copyMakeBorder(crop_img, pad_top[i], pad_bottom[i], \
                                    pad_left[i], pad_right[i], cv2.BORDER_CONSTANT, value=0)
      if crop_img is None:
        continue
      crop_img = cv2.resize(crop_img, (width, height)).astype(np.float32)
      crop_img[:, :, 0] -= self.mean[0]
      crop_img[:, :, 1] -= self.mean[1]
      crop_img[:, :, 2] -= self.mean[2]
      crop_img *= self.scale_factor
      crop_img = np.transpose(crop_img, (2, 0, 1))
      input_data[i] = crop_img.copy()
    return input_data

class PostProcessor():

  def __init__(self, threshold):
    """ Constructor.

    Args:
      threshold: List of threshold for confidence of each submodel
    """
    self.threshold = threshold

  def __nms(self, boxes, threshold, method_type):
    """ NMS function.

    Args:
      boxes: Detected bounding boxes
      threshold: Threshold
      method_type: Calculation method of NMS, "Min" or "Union"

    Returns:
      Bounding boxes after NMS.
    """
    if boxes.shape[0] == 0:
      return None
    y_1 = boxes[:, 0]
    x_1 = boxes[:, 1]
    y_2 = boxes[:, 2]
    x_2 = boxes[:, 3]
    s_s = boxes[:, 4]
    area = np.multiply(x_2 - x_1 + 1, y_2 - y_1 + 1)
    s_l = np.array(s_s.argsort())
    pick = []
    while s_l.shape[0] > 0:
      xx1 = np.maximum(x_1[s_l[-1]], x_1[s_l[0:-1]])
      yy1 = np.maximum(y_1[s_l[-1]], y_1[s_l[0:-1]])
      xx2 = np.minimum(x_2[s_l[-1]], x_2[s_l[0:-1]])
      yy2 = np.minimum(y_2[s_l[-1]], y_2[s_l[0:-1]])
      w_w = np.maximum(0.0, xx2 - xx1 + 1)
      h_h = np.maximum(0.0, yy2 - yy1 + 1)
      inter = w_w * h_h
      if method_type == 'Min':
        o_b = inter / np.minimum(area[s_l[-1]], area[s_l[0:-1]])
      else:
        o_b = inter / (area[s_l[-1]] + area[s_l[0:-1]] - inter)
      pick.append(s_l[-1])
      s_l = s_l[np.where(o_b <= threshold)[0]]
    return boxes[pick, :]

  def __box_regress(self, boxes):
    """ Regression function for bounding boxes.

    Args:
      boxes: Detected bounding boxes

    Returns:
      Bounding boxes after regression.
    """
    regw = boxes[:, 3] - boxes[:, 1] + 1
    regh = boxes[:, 2] - boxes[:, 0] + 1
    t_1 = boxes[:, 0] + boxes[:, 6] * regh - 1
    t_2 = boxes[:, 1] + boxes[:, 5] * regw - 1
    t_3 = boxes[:, 2] + boxes[:, 8] * regh - 1
    t_4 = boxes[:, 3] + boxes[:, 7] * regw - 1
    t_5 = boxes[:, 4]
    boxes = np.array([t_1, t_2, t_3, t_4, t_5]).T
    return boxes

  def __bbox2square(self, bboxes):
    """ Square the bounding boxes.

    Args:
      boxes: Detected bounding boxes

    Returns:
      Squared bounding boxes.
    """
    height = bboxes[:, 2] - bboxes[:, 0] + 1
    width = bboxes[:, 3] - bboxes[:, 1] + 1
    side = np.maximum(width, height).T
    bboxes[:, 0] += (height - side) * 0.5
    bboxes[:, 1] += (width - side) * 0.5
    bboxes[:, 2] = np.around(bboxes[:, 0] + side - 1);
    bboxes[:, 3] = np.around(bboxes[:, 1] + side - 1);
    bboxes[:, :2] = np.around(bboxes[:, :2])
    return bboxes

  def pnet_process_per_scale(self, pnet_output, scale):
    """ Postprocess function of PNet for each scale.

    Args:
      pnet_output: Output of PNet
      scale: Scale factor

    Returns:
      Bounding boxes.
    """
    conf = pnet_output['prob1'][0, 1, :, :]
    (y, x) = np.where(conf >= self.threshold[0])
    if len(x) == 0:
      return None
    stride = 2
    cell_size = 12
    bounding_box = np.array([y, x])
    bb1 = np.around(stride * bounding_box / scale)
    bb2 = np.around((stride * bounding_box + cell_size - 1) / scale)
    score = np.array([conf[y, x]])
    coord = pnet_output['conv4-2'][0]
    regression = np.array([coord[1, y, x], coord[0, y, x], \
                           coord[3, y, x], coord[2, y, x]])
    boxes = np.concatenate((bb1, bb2, score, regression), axis=0).T
    boxes = self.__nms(boxes, 0.5, 'Union')
    return boxes

  def pnet_process(self, boxes):
    """ Postprocess function of PNet.

    Args:
      boxes: Collection of boxes from each scale

    Returns:
      Bounding boxes.
    """
    boxes_num = 0 if boxes is None else boxes.shape[0]
    if boxes_num > 0:
      boxes = self.__nms(boxes, 0.7, 'Union');
      boxes = self.__box_regress(boxes);
      boxes = self.__bbox2square(boxes);
    return boxes

  def rnet_process(self, rnet_output, boxes):
    """ Postprocess function of RNet.

    Args:
      rnet_output: Output of RNet
      boxes: Bounding boxes

    Returns:
      Bounding boxes.
    """
    score = rnet_output['prob1'][:, 1]
    pass_t = np.where(score > self.threshold[1])[0]
    score = np.array([score[pass_t]]).T
    regression = rnet_output['conv5-2'][pass_t, :]
    regression = np.array([regression[:, 1], regression[:, 0], \
                           regression[:, 3], regression[:, 2]]).T
    boxes = np.concatenate((boxes[pass_t, 0:4], score, regression), axis=1)
    boxes_num = 0 if boxes is None else boxes.shape[0]
    if boxes_num > 0:
      boxes = self.__nms(boxes, 0.7, 'Union')
      boxes = self.__box_regress(boxes)
      boxes = self.__bbox2square(boxes)
    return boxes

  def onet_process(self, onet_output, boxes):
    """ Postprocess function of ONet.

    Args:
      onet_output: Output of ONet
      boxes: Bounding boxes

    Returns:
      Bounding boxes.
    """
    score = onet_output['prob1'][:, 1]
    pass_t = np.where(score > self.threshold[2])[0]
    score = np.array([score[pass_t]]).T
    regression = onet_output['conv6-2'][pass_t, :]
    regression = np.array([regression[:, 1], regression[:, 0], \
                           regression[:, 3], regression[:, 2]]).T
    boxes = np.concatenate((boxes[pass_t, 0:4], score, regression), axis=1)
    boxes_height = boxes[:, 2] - boxes[:, 0] + 1
    boxes_width = boxes[:, 3] - boxes[:, 1] + 1
    points = onet_output['conv6-3'][pass_t, :]
    points_x = np.empty([boxes.shape[0], 5], dtype=np.float32)
    points_y = np.empty([boxes.shape[0], 5], dtype=np.float32)
    for i in range(boxes.shape[0]):
      points_x[i] = boxes[i, 1] + points[i, 5:] * boxes_width[i] - 1
      points_y[i] = boxes[i, 0] + points[i, :5] * boxes_height[i] - 1
    boxes = np.concatenate((boxes, points_x, points_y), axis=1)
    boxes[:, :5] = self.__box_regress(boxes)
    boxes = self.__nms(boxes, 0.7, 'Min')
    return boxes

  def get_reference(self, compare_path):
    """ Get correct result from given file.
    Args:
      compare_path: Path to correct result file

    Returns:
      Correct result.
    """
    if compare_path:
      with open(compare_path, 'r') as f:
        reference = json.load(f)
        return reference["boxes"]
    return None

  def compare(self, reference, result, loop_id):
    """ Compare result.
    Args:
      reference: Correct result
      result: Output result
      loop_id: Loop iterator number

    Returns:
      True for success and False for failure
    """
    if not reference or loop_id > 0:
      return True
    detected_num = len(result)
    reference_num = len(reference)
    if (detected_num != reference_num):
      message = "Expected deteted number is {}, but detected {}!"
      print(message.format(reference_num, detected_num))
      return False
    ret = True
    message = "Face {} Box: [{}, {}, {}, {}], Score: {:.6f}"
    fail_info = "Compare failed! Expect: " + message
    ret_info = "Result Box: " + message
    for i in range(detected_num):
      result_str = list(result[i, :5].copy())
      for j in range(4):
        result_str[j] = "{:.4f}".format(result_str[j])
      result_str[4] = "{:.6f}".format(result_str[4])
      if result_str != reference[i]:
        x = int(float(reference[i][1])) if float(reference[i][1]) > 0 else 0
        y = int(float(reference[i][0])) if float(reference[i][0]) > 0 else 0
        width = int(float(reference[i][3]) - float(reference[i][1]))
        height = int(float(reference[i][2]) - float(reference[i][0]))
        print(fail_info.format(i, x, y, width, height, float(result[i][4])))
        x = int(result[i, 1]) if result[i, 1] > 0 else 0
        y = int(result[i, 0]) if result[i, 0] > 0 else 0
        width = int(result[i, 3] - result[i, 1])
        height = int(result[i, 2] - result[i, 0])
        print(ret_info.format(i, x, y, width, height, result[i, 4]))
        ret = False
    return ret
