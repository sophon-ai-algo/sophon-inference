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

"""The operation of box
"""
from __future__ import division
import math
import numpy as np


def get_largest_box_index(bboxes):
  """Return the largest box index

  Args:
      bboxes: Input bboxes

  Returns:
      bbox: The largest box index
  """
  bboxes = np.array(bboxes)
  delta_x = bboxes[:, 2] - bboxes[:, 0]
  delta_y = bboxes[:, 3] - bboxes[:, 1]
  return np.argmax(delta_x * delta_y)


def extend_box(bbox, image_size, extend_ratio):
  """Zoom the input bbox

  Args:
      bbox: Input bbox
      image_size: Input image original size
      extend_ratio: Zoom ratio

  Returns:
      a box extand from input bbox
  """
  x_x = (bbox[2] - bbox[0]) * extend_ratio
  y_y = (bbox[3] - bbox[1]) * extend_ratio
  extend_bbox = []
  extend_bbox.append(int(max(0, (bbox[2] + bbox[0]) / 2 - x_x / 2)))
  extend_bbox.append(int(max(0, (bbox[3] + bbox[1]) / 2 - y_y / 2)))
  extend_bbox.append(int(min(image_size[1], (bbox[2] + bbox[0]) / 2 + x_x / 2)))
  extend_bbox.append(int(min(image_size[0], (bbox[3] + bbox[1]) / 2 + y_y / 2)))
  return extend_bbox


def fix_bbox_boundary(bboxes, image_size):
  """Fix the coordinates of bbox

    Args:
        bboxes: Detected bboxes to be processed
        image_size: Original input image size

    Returns:
        Fixed bbox
    """
  bboxes[bboxes < 0] = 0
  bboxes[:, 2][bboxes[:, 2] > image_size[1]] = image_size[1]
  bboxes[:, 3][bboxes[:, 3] > image_size[0]] = image_size[0]
  return bboxes


def crop_and_shift(image, bbox, landmark):
  """Crop face && reset the landmark coordinates

    Args:
        image: Input image
        bbox: Detected face box
        landmark: Detected face landmark

    Returns:
        subimage: crop from image
        landmark: Fixed coordinates landmark
    """
  landmark = np.array(landmark)
  radius = math.sqrt((bbox[2] - bbox[0])**2 + (bbox[3] - bbox[1])**2) * 0.6
  new_bbox = [
      int(max(0, (bbox[2] + bbox[0]) / 2 - radius)),
      int(max(0, (bbox[3] + bbox[1]) / 2 - radius)),
      int(min(image.shape[1], (bbox[2] + bbox[0]) / 2 + radius)),
      int(min(image.shape[0], (bbox[3] + bbox[1]) / 2 + radius))
  ]
  landmark[0:5] = landmark[0:5] - new_bbox[0]
  landmark[5:10] = landmark[5:10] - new_bbox[1]
  return image[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2], :], landmark


def restore_param(bboxes, points, rescale_param):
  """restore bboxes with original image size when detecting in different size

  Args:
      bboxes:list. Input bbox
      points:list. Input landmark
      rescale_param: tuple. ex:(scale_y, scale_x, pad_top, pad_left)
                  scale_y or scale_x = original_size/detected_size

  Returns:
      bboxes:list. restore bbox
      points:list. restore point
  """
  bboxes[:, 0] = (bboxes[:, 0] - rescale_param[3]) * rescale_param[0]
  bboxes[:, 1] = (bboxes[:, 1] - rescale_param[2]) * rescale_param[1]
  bboxes[:, 2] = (bboxes[:, 2] - rescale_param[3]) * rescale_param[0]
  bboxes[:, 3] = (bboxes[:, 3] - rescale_param[2]) * rescale_param[1]
  if points is not None:
    points[:, 0:5] = (points[:, 0:5] - rescale_param[3]) * rescale_param[1]
    points[:, 5:10] = (points[:, 5:10] - rescale_param[2]) * rescale_param[1]
  return bboxes, points


def nms(boxes, threshold, dataset_type):
  """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :dataset_type: 'Min' or others ('union')
    :returns: idx list
  """
  if boxes.shape[0] == 0:
    return np.array([])
  x_1 = boxes[:, 0]
  y_1 = boxes[:, 1]
  x_2 = boxes[:, 2]
  y_2 = boxes[:, 3]
  s_s = boxes[:, 4]
  area = np.multiply(x_2 - x_1 + 1, y_2 - y_1 + 1)
  s_l = np.array(s_s.argsort())  # read s using s_l

  pick = []
  while s_l.shape[0] > 0:
    xx1 = np.maximum(x_1[s_l[-1]], x_1[s_l[0:-1]])
    yy1 = np.maximum(y_1[s_l[-1]], y_1[s_l[0:-1]])
    xx2 = np.minimum(x_2[s_l[-1]], x_2[s_l[0:-1]])
    yy2 = np.minimum(y_2[s_l[-1]], y_2[s_l[0:-1]])
    w_w = np.maximum(0.0, xx2 - xx1 + 1)
    h_h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w_w * h_h
    if dataset_type == 'Min':
      o_b = inter / np.minimum(area[s_l[-1]], area[s_l[0:-1]])
    else:
      o_b = inter / (area[s_l[-1]] + area[s_l[0:-1]] - inter)
    pick.append(s_l[-1])
    s_l = s_l[np.where(o_b <= threshold)[0]]
  return pick


def bbox_transform_inv(boxes, deltas):
  """Proposal layer bbox operation

  Args:
      bboxes: RPN output rois
      deltas: bbox regression data

  Returns:
      bboxes: regressed bboxes
  """
  if boxes.shape[0] == 0:
    return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

  boxes = boxes.astype(deltas.dtype, copy=False)

  widths = boxes[:, 2] - boxes[:, 0] + 1.0
  heights = boxes[:, 3] - boxes[:, 1] + 1.0
  ctr_x = boxes[:, 0] + 0.5 * widths
  ctr_y = boxes[:, 1] + 0.5 * heights

  d_x = deltas[:, 0::4]
  d_y = deltas[:, 1::4]
  d_w = deltas[:, 2::4]
  d_h = deltas[:, 3::4]

  pred_ctr_x = d_x * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
  pred_ctr_y = d_y * heights[:, np.newaxis] + ctr_y[:, np.newaxis]

  # potential_of = np.where(dw > 100)[0]
  try:
    pred_w = np.exp(d_w) * widths[:, np.newaxis]
    pred_h = np.exp(d_h) * heights[:, np.newaxis]
  except FloatingPointError:
    # print('WARNING: OVERFLOW OCCURED IN np.exp(dw) and/or np.exp(dh)')
    for i in range(d_w.shape[0]):
      for j in range(d_w.shape[1]):
        if d_w[i][j] > 50:
          d_w[i][j] = 5
        if d_h[i][j] > 50:
          d_h[i][j] = 5
    pred_w = np.exp(d_w) * widths[:, np.newaxis]
    pred_h = np.exp(d_h) * heights[:, np.newaxis]

  pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
  # x1
  pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
  # y1
  pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
  # x2
  pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
  # y2
  pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
  return pred_boxes
