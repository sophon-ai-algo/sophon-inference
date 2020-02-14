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

"""The function of visualization
"""
import numpy as np
import cv2
from ..algo_cv.det.detconfig import get_labels


def draw_face_bbox(image, bboxes, probs, points=None):
  """Draw detected face on the original image

  Args:
      image(np.ndarray). input image
      bboxes(list). detected bbox
      probs(list). confidence of bbox
      points(list). face landmark

  Returns:
      Image after detection
  """
  draw = image.copy()
  if points is None:
    points = [None] * len(bboxes)
  for bbox, prob, point in zip(bboxes, probs, points):
    cv2.putText(draw, str(prob)[:8], (int(bbox[0]), int(bbox[1]-2)), \
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    cv2.rectangle(
        draw,
        pt1=(int(bbox[0]), int(bbox[1])),
        pt2=(int(bbox[2]), int(bbox[3])),
        color=(255, 0, 0),  # (B, G, R)
        thickness=2)
    if point is not None:
      for idx in range(5):
        cv2.circle(draw, (int(point[idx]), int(point[idx+5])), \
            2, (0, 0, 255))
  return draw


def draw_obj_bbox(image, bboxes, classes, probs, dataset_type):
  """Draw detected object on the original image

  Args:
      image(np.ndarray). input image
      bboxes(list): detected bbox
      classes(list): cls index
      probs(list): confidence of bbox
      dataset_type(str): training dataset type

  Returns:
      Image after detection
  """
  draw = image.copy()
  label_list = get_labels(dataset_type)
  for bbox, cls, prob in zip(bboxes, classes, probs):
    label = "{}:{:.6f}".format(label_list[cls], prob)
    # map -> list
    inner_color = np.random.uniform(0, 255, 3).astype(np.uint8)
    color = (int(inner_color[0]), int(inner_color[1]), int(inner_color[2]))
    cv2.rectangle(
        draw,
        pt1=(int(bbox[0]), int(bbox[1])),
        pt2=(int(bbox[2]), int(bbox[3])),
        color=color,
        thickness=1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    pt2 = int(bbox[0]) + t_size[0] + 3, int(bbox[1]) + t_size[1] + 4
    cv2.rectangle(
        draw,
        pt1=(int(bbox[0]), int(bbox[1])),
        pt2=tuple(pt2),
        color=color,
        thickness=-1)
    cv2.putText(draw, label, (int(bbox[0]), t_size[1] + 4 + int(bbox[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
  return draw
