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

"""The implementation of object detection yolov3
"""
from __future__ import print_function
from __future__ import division
import numpy as np
import cv2
from ...engine.base_engine import BaseEngine
from ...engine import Engine
from ...utils.box_operation import nms


class ObjectDetectionYOLOV3(BaseEngine):
  """Construct yolov3 detector
  """

  def __init__(self,
               arch,
               mode,
               anchors,
               detected_size,
               threshold=0.5,
               nms_threshold=0.45,
               num_classes=80):

    super(ObjectDetectionYOLOV3, self).__init__()
    self.mode = mode
    self.threshold = threshold
    self.nms_threshold = nms_threshold
    self.num_classes = num_classes
    self.anchors = anchors
    self.detected_size = detected_size
    self.net = Engine(arch, mode)
    self.input_names = arch['input_names']

  # def predict_transform(self, prediction, anchors):
  #     """Transform the logspace offset to linear space coordinates
  #     and rearrange the row-wise output
  #
  #     Args:
  #         prediction: np.ndarray. ex: shape(1,255,13,13)
  #         anchors: list. anchors group
  #     """
  #     inp_dim = self.detected_size[0]
  #     num_classes = self.num_classes
  #     batch_size = prediction.shape[0]
  #     stride =  inp_dim // prediction.shape[2] # ex: 416 // 52
  #     grid_size = inp_dim // stride
  #     bbox_attrs = 5 + num_classes
  #     num_anchors = len(anchors)
  #     prediction = np.reshape(prediction, (batch_size,\
  #         bbox_attrs * num_anchors, grid_size * grid_size))
  #     prediction = np.swapaxes(prediction, 1, 2)
  #     prediction = np.reshape(prediction, (batch_size,\
  #         grid_size * grid_size * num_anchors, bbox_attrs))
  #     anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
  #
  #     # Sigmoid the  centre_X, centre_Y. and object confidencce
  #     prediction[:,:,0] = 1 / (1 + np.exp(-prediction[:,:,0]))
  #     prediction[:,:,1] = 1 / (1 + np.exp(-prediction[:,:,1]))
  #     prediction[:,:,4] = 1 / (1 + np.exp(-prediction[:,:,4]))
  #
  #     # Add the center offsets
  #     grid = np.arange(grid_size)
  #     a,b = np.meshgrid(grid, grid)
  #     x_offset = a.reshape(-1,1)
  #     y_offset = b.reshape(-1,1)
  #     x_y_offset = np.concatenate((x_offset, y_offset), 1)
  #     x_y_offset = np.tile(x_y_offset, (1, num_anchors))
  #     x_y_offset = np.expand_dims(x_y_offset.reshape(-1,2), axis=0)
  #     prediction[:,:,:2] += x_y_offset
  #
  #     # log space transform height, width and box corner point x-y
  #     anchors = np.tile(anchors, (grid_size * grid_size, 1))
  #     anchors = np.expand_dims(anchors, axis=0)
  #     prediction[:,:,2:4] = np.exp(prediction[:,:,2:4])*anchors
  #     prediction[:,:,5: 5 + num_classes] = 1 /\
  #         (1 + np.exp(-prediction[:,:, 5 : 5 + num_classes]))
  #     prediction[:,:,:4] *= stride
  #     box_corner = np.zeros(prediction.shape)
  #     box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
  #     box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
  #     box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
  #     box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
  #     prediction[:,:,:4] = box_corner[:,:,:4]
  #     return prediction

  def predict_transform(self, prediction, anchors):
    """Transform the logspace offset to linear space coordinates
        and rearrange the row-wise output
    Args:
        prediction: np.ndarray. ex: shape(1,255,13,13)
        anchors: list. anchors group
    """
    inp_dim = self.detected_size[0]
    num_classes = self.num_classes
    batch_size = prediction.shape[0]
    stride = inp_dim // prediction.shape[2]  # ex: 416 // 52
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    prediction = np.reshape(prediction, (batch_size,\
        bbox_attrs * num_anchors, grid_size * grid_size)) # 1*255*(52*52)
    prediction = np.swapaxes(prediction, 1, 2)  # 1*(52*52)*255
    prediction = np.reshape(prediction, (batch_size,\
        grid_size * grid_size * num_anchors, bbox_attrs)) # 1*(52*52*3)*85
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 4] = 1 / (1 + np.exp(-prediction[:, :, 4]))
    # filter prediction
    prediction_mask = (prediction[:, :, 4] > self.threshold).reshape((-1))
    prediction_mask = np.nonzero(prediction_mask)
    prediction = prediction[:, prediction_mask[0]]
    prediction[:, :, 0] = 1 / (1 + np.exp(-prediction[:, :, 0]))
    prediction[:, :, 1] = 1 / (1 + np.exp(-prediction[:, :, 1]))

    # Add the center offsets
    grid = np.arange(grid_size)
    a_grid, b_grid = np.meshgrid(grid, grid)
    x_offset = a_grid.reshape(-1, 1)
    y_offset = b_grid.reshape(-1, 1)
    x_y_offset = np.concatenate((x_offset, y_offset), 1)
    x_y_offset = np.tile(x_y_offset, (1, num_anchors))
    x_y_offset = np.expand_dims(x_y_offset.reshape(-1, 2), axis=0)
    x_y_offset = x_y_offset[:, prediction_mask[0]]
    prediction[:, :, :2] += x_y_offset

    # log space transform height, width and box corner point x-y
    anchors = np.tile(anchors, (grid_size * grid_size, 1))  # col:1 row: g*g
    anchors = np.expand_dims(anchors, axis=0)
    prediction[:, :, 2:4] = np.exp(prediction[:, :, 2:4]) * \
        anchors[:, prediction_mask[0]]
    prediction[:, :, 5: 5 + num_classes] = 1 /\
        (1 + np.exp(-prediction[:, :, 5 : 5 + num_classes]))
    prediction[:, :, :4] *= stride
    box_corner = np.zeros(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]
    return prediction

  #def bbox_iou(self, bbox1, bbox2):
  #    """Compute intersection of union score between bounding boxes
  #    """
  #    # Get the coordinates of bounding boxes
  #    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[:,0], \
  #      bbox1[:,1], bbox1[:,2], bbox1[:,3]
  #    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[:,0], \
  #      bbox2[:,1], bbox2[:,2], bbox2[:,3]
  #
  #    # get the corrdinates of the intersection rectangle
  #    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
  #    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
  #    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
  #    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
  #    #Intersection area
  #    inter_area =
  #      np.clip(inter_rect_x2 - inter_rect_x1 + 1, a_min=0, a_max=None) \
  #      * np.clip(inter_rect_y2 - inter_rect_y1 + 1, a_min=0, a_max=None)
  #    #Union Area
  #    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
  #    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
  #
  #    iou = inter_area / (b1_area + b2_area - inter_area)
  #    return iou

  def map_coord(self, rects, img_ori):
    """mapping the coordinates to original
    """
    bboxes = []
    classes = []
    probs = []
    scaling_factor = min(1, self.detected_size[0] / img_ori.shape[1])
    for pt1, pt2, cls, prob in rects:
      pt1[0] -= (self.detected_size[0] - scaling_factor *\
          img_ori.shape[1])/2
      pt2[0] -= (self.detected_size[0] - scaling_factor *\
          img_ori.shape[1])/2
      pt1[1] -= (self.detected_size[0] - scaling_factor *\
          img_ori.shape[0])/2
      pt2[1] -= (self.detected_size[0] - scaling_factor *\
          img_ori.shape[0])/2

      pt1[0] = np.clip(int(pt1[0] / scaling_factor),\
          a_min=0, a_max=img_ori.shape[1])
      pt2[0] = np.clip(int(pt2[0] / scaling_factor),\
          a_min=0, a_max=img_ori.shape[1])
      pt1[1] = np.clip(int(pt1[1] / scaling_factor),\
          a_min=0, a_max=img_ori.shape[1])
      pt2[1] = np.clip(int(pt2[1] / scaling_factor),\
          a_min=0, a_max=img_ori.shape[1])
      bboxes.append([int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1])])
      classes.append(cls)
      probs.append(prob)
    return bboxes, classes, probs

  def preprocess(self, image):
    """yolov3 img preprocess
       resize image with unchanged aspect ratio using padding 128
    scale: 1/255
    bgr->rgb
    hwc->chw
    to be added to BaseEngine
    """
    img_h, img_w = image.shape[0], image.shape[1]
    net_h, net_w = self.detected_size[0], self.detected_size[1]
    new_w = int(img_w * min(net_w / img_w, net_h / img_h))
    new_h = int(img_h * min(net_w / img_w, net_h / img_h))
    resized_image = cv2.resize(
        image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # canvas
    canvas = np.full((net_w, net_h, 3), 127.5)
    canvas[(net_h - new_h) // 2:(net_h - new_h) // 2 + new_h,\
        (net_w - new_w) // 2:(net_w - new_w) // 2 + new_w, :]\
        = resized_image
    return canvas[:, :, ::-1].transpose([2, 0, 1]) / 255.0

  def postprocess(self, out, image):
    """Parse the yolov3 output

    To be improve
    """
    prediction = None
    anchors = []
    for key, value in out.items():
      if key == 'layer82-conv':
        anchors = self.anchors[2]
      elif key == 'layer94-conv':
        anchors = self.anchors[1]
      elif key == 'layer106-conv':
        anchors = self.anchors[0]
      if prediction is None:
        prediction = self.predict_transform(value, anchors)
      else:
        prediction = np.concatenate([prediction,\
            self.predict_transform(value, anchors)], axis=1)

    # confidence thresholding
    conf_mask = np.expand_dims((prediction[:, :, 4] > self.threshold), axis=2)
    prediction = prediction * conf_mask
    prediction = prediction[np.nonzero(prediction[:, :, 4])]
    # rearrange results
    img_result = np.zeros((prediction.shape[0], 6))
    max_conf_cls = np.argmax(prediction[:, 5:5 + self.num_classes], 1)
    max_conf_score = np.amax(prediction[:, 5:5 + self.num_classes], 1)
    img_result[:, :4] = prediction[:, :4]
    img_result[:, 5] = max_conf_cls
    img_result[:, 4] = prediction[:, 4] * max_conf_score
    # non-maxima suppression
    result = []
    img_result = img_result[img_result[:, 5].argsort()[::-1]]
    ind = 0
    #while ind < img_result.shape[0]:
    #    bbox_cur = np.expand_dims(img_result[ind], 0)
    #    ious = self.bbox_iou(bbox_cur, img_result[(ind+1):])
    #    nms_mask = np.expand_dims(ious < self.nms_threshold, axis=2)
    #    img_result[(ind+1):] = img_result[(ind+1):] * nms_mask
    #    img_result = img_result[np.nonzero(img_result[:, 5])]
    #    ind += 1
    pick = nms(img_result, self.nms_threshold, 'union')
    # nms result pick: []
    if np.array(pick).shape[0] == 0:
      return [], [], []
    img_result = img_result[pick, :]
    for ind in range(img_result.shape[0]):
      pt1 = [int(img_result[ind, 0]), int(img_result[ind, 1])]
      pt2 = [int(img_result[ind, 2]), int(img_result[ind, 3])]
      cls, prob = int(img_result[ind, 5]), img_result[ind, 4]
      result.append((pt1, pt2, cls, prob))
    # mapping coordinates to original image
    bboxes, classes, probs = self.map_coord(result, image)
    return bboxes, classes, probs

  def predict(self, image):
    """yolov3 inference

    Args:
        image(np.ndarray). detected image
    Returns:
        bboxes, classes, probs(tuple)
    """
    result_bboxes = []
    result_classes = []
    result_probs = []
    preprocess_data = self.preprocess(image)
    input_data = {
        self.input_names[0]: np.array([preprocess_data], dtype="float32")
    }
    out = self.net.predict(input_data)
    result_bboxes, result_classes, result_probs =\
        self.postprocess(out, image)
    return (result_bboxes, result_classes, result_probs)
