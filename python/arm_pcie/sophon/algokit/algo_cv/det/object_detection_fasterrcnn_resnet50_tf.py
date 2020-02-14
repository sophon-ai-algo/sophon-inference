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

"""The implementation of object detection fasterrcnn
"""
from __future__ import print_function
from __future__ import division
import os
import numpy as np
# from sophon.auto_split.api import split as splitter
from sophon.auto_runner.api import infer
from sophon.auto_runner.api import load
# from sophon.auto_split.api import convert as compiler
from ...engine.base_engine import BaseEngine
from ...utils.box_operation import restore_param, fix_bbox_boundary


class ObjectDetectionFASTERRCNNRESNET50TF(BaseEngine):
  """Construct fasterrcnn_resnet50_tf object detector
  """

  def __init__(self, source_path, subgraph_path, tfmodel_path, framework,
               input_names, output_names, input_shapes, layout, is_dynamic,
               detected_size, target, conf_threshold):
    super(ObjectDetectionFASTERRCNNRESNET50TF, self).__init__()
    # detected_size: (h, w)
    # image will be resize before processing detection
    self.source_path = source_path
    self.subgraph_path = subgraph_path
    self.tfmodel_path = tfmodel_path
    self.framework = framework
    self.input_names = input_names
    self.input_shapes = input_shapes
    self.detected_size = detected_size
    self.conf_threshold = conf_threshold
    self.tensors_dict = {}
    if len(self.input_names) == len(self.input_shapes):
      for input_name, input_shape in zip(input_names, input_shapes):
        self.tensors_dict[input_name] = np.ndarray((input_shape),
                                                   dtype=np.float32)
    else:
      raise ValueError('input names and input shapes sizes do not match!')
    self.output_names = output_names
    self.layout = layout
    self.is_dynamic = is_dynamic
    self.target = target
    # check the subgraph file
    # if not os.path.isdir(self.subgraph_path):
    #   print("attention please: this model needs to be split...")
    #   # autodeploy split
    #   splitter(self.framework, self.tensors_dict, self.subgraph_path, \
    #     self.tfmodel_path, params_path=None, outputs=self.output_names, \
    #     dynamic=self.is_dynamic, layout=self.layout)
    #   print("split done!")
    #   compiler(self.subgraph_path, optimize=1, compare=True, target=self.target)
    #   print("compile done!")
    # else:
    #   subgraph_filenames = os.listdir(self.subgraph_path)
    #   if not any(name.endswith('.pb') for name in subgraph_filenames):
    #     splitter(self.framework, self.tensors_dict, self.subgraph_path, \
    #       self.tfmodel_path, params_path=None, outputs=self.output_names, \
    #       dynamic=self.is_dynamic, layout=self.layout)
    #     print("split done!")
    #   if not any(name.startswith('graph_ir') for name in subgraph_filenames):
    #     compiler(
    #         self.subgraph_path, optimize=1, compare=True, target=self.target)
    #     print("compile done!")

  def coord_map(self, bboxes, classes, scores, restoreparam, origin_size):
    """map fasterrcnn_resnet50_tf detected bboxes

        Returns:
            bboxes:list
            classes:list
            probs:list
        """
    # filter bboxes if score < conf_threshold
    bboxes_mask = (scores > self.conf_threshold)
    bboxes_mask = np.nonzero(bboxes_mask)
    scores = scores[bboxes_mask]
    classes = classes[bboxes_mask]
    bboxes = bboxes[bboxes_mask]
    # map ymin, xmin, ymax, xmax
    # bboxes shape (num_detections, 4)
    bboxes[:, 0] = bboxes[:, 0] * self.detected_size[0]  # y_min
    bboxes[:, 1] = bboxes[:, 1] * self.detected_size[1]  # x_min
    bboxes[:, 2] = bboxes[:, 2] * self.detected_size[0]  # y_max
    bboxes[:, 3] = bboxes[:, 3] * self.detected_size[1]  # x_max
    # mv ymin, xmin, ymax, xmax -> xmin, ymin, xmax, ymax
    bboxes = bboxes[:, [1, 0, 3, 2]]
    # restore bbox
    bboxes, _ = restore_param(bboxes, None, restoreparam)
    bboxes = fix_bbox_boundary(bboxes, origin_size)
    return bboxes, classes.astype(np.int32), scores

  def predict(self, images):
    """fasterrcnn_resnet50_tf forward
        """
    self.time.tic()
    if isinstance(images, list):
      use_batch = True
    else:
      use_batch = False
      images = [images]

    # inference
    result_bboxes = []
    result_classes = []
    result_probs = []
    for image in images:
      origin_size = image.shape
      # this version fasterrcnn_resnet50 input fix 600*800(h*w)
      # pandding location center
      data, rescale_param = self.rescale_image(image, self.detected_size, True)
      data = data[..., [2, 1, 0]]  # BGR2RGB PIL.Image read as RGB
      input_data = {self.input_names[0]: np.array([data])}
      model = load(self.subgraph_path)
      out = infer(model, input_data)
      print("inference done!")
      detected_num = int(out['num_detections:0'][0])
      input_bboxes = out['detection_boxes:0'][0][:detected_num]
      input_classes = out['detection_classes:0'][0][:detected_num]
      input_scores = out['detection_scores:0'][0][:detected_num]
      output_bboxes, output_classes, output_scores = \
          self.coord_map(input_bboxes, input_classes, input_scores, \
                         rescale_param, origin_size)
      result_bboxes.append(output_bboxes)
      result_classes.append(output_classes)
      result_probs.append(output_scores)
    if use_batch:
      return (result_bboxes, result_classes, result_probs)
    return (result_bboxes[0], result_classes[0], result_probs[0])
