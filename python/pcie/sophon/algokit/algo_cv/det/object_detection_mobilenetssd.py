# Copyright 2016-2022 Bitmain Technologies Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The implementation of object detection mobilenetssd
"""
from __future__ import print_function
from __future__ import division
import numpy as np
import cv2
from ...engine.base_engine import BaseEngine
from ...engine import Engine
from ...utils.box_operation import nms


class ObjectDetectionMOBILENETSSD(BaseEngine):
  """Construct mobilenetssd detector
  """

  def __init__(self,
               arch,
               mode,
               priorbox_num,
               detected_size,
               threshold=0.25,
               nms_threshold=0.45,
               num_classes=21):
    super(ObjectDetectionMOBILENETSSD, self).__init__()
    self.detected_size = detected_size
    self.threshold = threshold
    self.nms_threshold = nms_threshold
    self.num_classes = num_classes
    self.priorbox_num = priorbox_num
    self.net = Engine(arch, mode)
    self.input_names = arch['input_names']

  def map_coord(self, image, out):
    """detected bbox coordinates map
    """
    org_h = image.shape[0]
    org_w = image.shape[1]
    box = out[:, 0:4] * np.array([org_w, org_h, org_w, org_h])
    cls = out[:, 5]
    conf = out[:, 4]
    return (box, cls.astype(np.int32) + 1, conf)

  def detection_output_layer(self, conf_data, loc_data, priorbox_data):
    """the implementation of ssd detection output layer

    different from original implementation
    key value:
        priorbox_num
        num_classes
    """
    # process loc_data: x y w h
    loc_pre = loc_data.reshape((self.priorbox_num, -1))  # 1917*4
    # process conf_data
    conf_pre = conf_data.reshape(
        (self.priorbox_num, self.num_classes))  # 1917*21
    # process priorbox_data: x_min y_min x_max y_max
    prior_bboxes = priorbox_data[:, 0].reshape(self.priorbox_num, -1)  # 1917*4
    prior_variances = priorbox_data[:, 1].reshape(self.priorbox_num,
                                                  -1)  # 1917*4
    prediction = np.concatenate([loc_pre, prior_bboxes, prior_variances,\
        conf_pre], axis=1) # 1917 * (4+4+4+21)
    # bbox filter
    bbox_mask = (np.amax(conf_pre[:, 1:21], axis=1) > self.threshold)
    prediction = prediction[np.nonzero(bbox_mask)]
    # fill out
    out = np.zeros((prediction.shape[0], 6))
    out[:, 4] = np.amax(prediction[:, 13:33], axis=1)
    out[:, 5] = np.argmax(prediction[:, 13:33], axis=1)
    # decode the coordinates 0:4 loc | 4:8 prior | 8:12 var (0.1,0.2)
    prior_w = prediction[:, 6] - prediction[:, 4]
    prior_h = prediction[:, 7] - prediction[:, 5]
    prior_x = (prediction[:, 6] + prediction[:, 4]) / 2.0
    prior_y = (prediction[:, 7] + prediction[:, 5]) / 2.0
    bbox_x = prediction[:, 8] * prediction[:, 0] * prior_w + prior_x
    bbox_y = prediction[:, 9] * prediction[:, 1] * prior_h + prior_y
    bbox_w = np.exp(prediction[:, 10] * prediction[:, 2]) * prior_w
    bbox_h = np.exp(prediction[:, 11] * prediction[:, 3]) * prior_h
    out[:, 0] = np.minimum(np.maximum((bbox_x - bbox_w / 2), 0.0), 1.0)
    out[:, 1] = np.minimum(np.maximum((bbox_y - bbox_h / 2), 0.0), 1.0)
    out[:, 2] = np.minimum(np.maximum((bbox_x + bbox_w / 2), 0.0), 1.0)
    out[:, 3] = np.minimum(np.maximum((bbox_y + bbox_h / 2), 0.0), 1.0)
    nms_out = np.zeros((1, 6))
    for i in range(self.num_classes - 1):
      temp = out[out[:, 5].astype(np.int32) == i]
      if temp.shape[0] != 0:
        pick = nms(temp, self.nms_threshold, 'union')
        nms_out = np.concatenate((nms_out, temp[pick]))
    return nms_out[1:]

  def preprocess_org(self, image):
    """mobilenetssd org preprocess

    direct resize image
    """
    img = cv2.resize(image, (self.detected_size[0], self.detected_size[1]))
    img = img - 127.5
    img = img / 127.5
    return img.transpose((2, 0, 1))

  # def preprocess(self, image: np.ndarray):
  #     """mobilenetssd new preprocess
  #
  #     resize image with unchanged aspect ratio using padding 127.5
  #     submean: 127.5
  #     scale: 1/127.5
  #     hwc->chw
  #     to be added to BaseEngine
  #     """
  #     img_h, img_w = image.shape[0], image.shape[1]
  #     net_h, net_w = self.detected_size[0], self.detected_size[1]
  #     new_w = int(img_w * min(net_w / img_w, net_h / img_h))
  #     new_h = int(img_h * min(net_w / img_w, net_h / img_h))
  #     resized_image = cv2.resize(image, (new_w,new_h), \
  #       interpolation =cv2.INTER_NEAREST)
  #     # canvas
  #     canvas = np.full((net_w, net_h, 3), 127.5)
  #     canvas[(net_h - new_h) // 2:(net_h - new_h) // 2 + new_h,\
  #         (net_w - new_w) // 2:(net_w - new_w) // 2 + new_w,:]\
  #         = resized_image
  #     canvas = canvas - 127.5  # pad 0.0 and skip this step
  #     return canvas.transpose((2, 0, 1)) / 127.5

  def postprocess(self, image, out):
    """mobilenetssd postprocess
    """
    # extract output data
    conf_data = out['mbox_conf_flatten']  # 1*(21*1917)*1*1
    loc_data = out['mbox_loc']  # 1*(4*1917)*1*1
    priorbox_data = out['mbox_priorbox']  # 1*2*(4*1917)*1*1
    # decode data
    detection_out = self.detection_output_layer(\
        conf_data, loc_data, priorbox_data)
    bboxes, classes, probs = self.map_coord(image, detection_out)
    return bboxes, classes, probs

  def predict(self, image):
    """mobilenet ssd
    Args:
        image(np.ndarray). detected image
    Returns:
        bboxes, classes, probs(tuple)
    """
    result_bboxes = []
    result_classes = []
    result_probs = []
    preprocess_data = self.preprocess_org(image)
    input_data = {
        self.input_names[0]: np.array([preprocess_data], dtype="float32")
    }
    out = self.net.predict(input_data)
    result_bboxes, result_classes, result_probs =\
        self.postprocess(image, out)
    return (result_bboxes, result_classes, result_probs)
