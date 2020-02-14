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

"""The implementation of object detection yolov3 mxnet
"""
from __future__ import print_function
from __future__ import division
import os
import numpy as np
# import mxnet as mx # for proprocess
import cv2
# from sophon.auto_split.api import split as splitter
from sophon.auto_runner.api import infer
from sophon.auto_runner.api import load
# from sophon.auto_split.api import convert as compiler
from ...engine.base_engine import BaseEngine
from ...utils.box_operation import restore_param, fix_bbox_boundary


class ObjectDetectionYOLOV3MX(BaseEngine):
  """Construct yolov3_mxnet object detector
  """

  def __init__(self, source_path, subgraph_path, json_path, params_path,
               framework, input_names, output_names, input_shapes,
               detected_size, target, conf_threshold):
    super(ObjectDetectionYOLOV3MX, self).__init__()
    # detected_size: (h, w)
    # image will be resize before processing detection
    self.source_path = source_path
    self.subgraph_path = subgraph_path
    self.json_path = json_path
    self.params_path = params_path
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
    self.target = target
    # check the subgraph file
    # if not os.path.isdir(self.subgraph_path):
    #   os.mkdir(self.subgraph_path)
    #   print("attention please: this model needs to be split...")
    #   # autodeploy split
    #   splitter(self.framework, self.tensors_dict, self.subgraph_path, \
    #     self.json_path, params_path=self.params_path)
    #   print("split done!")
    #   compiler(self.subgraph_path, target=self.target)
    #   print("compile done!")
    # else:
    #   subgraph_filenames = os.listdir(self.subgraph_path)
    #   if not any(name.endswith('.params') for name in subgraph_filenames):
    #     splitter(self.framework, self.tensors_dict, self.subgraph_path, \
    #       self.json_path, params_path=self.params_path)
    #     print("split done!")
    #   if not any(name.startswith('graph_ir') for name in subgraph_filenames):
    #     compiler(self.subgraph_path, target=self.target)
    #     print("compile done!")

  def coord_map(self, bboxes, classes, scores, restoreparam, origin_size):
    """map yolov3_mx detected bboxes

        Args:
            bboxes:list input detected bboxes
            classes:list input detected classesidx
            scores:list input detected class score
            restoreparam:tuple bbox restoreparam
            orgin_size:tuple input image original size
        Returns:
            bboxes:list
            classes:list
            probs:list
        """
    # filter bboxes if score < conf_threshold
    bboxes_mask = (scores[:, 0] > self.conf_threshold)
    bboxes_mask = np.nonzero(bboxes_mask)
    scores = scores[bboxes_mask]
    classes = classes[bboxes_mask]
    bboxes = bboxes[bboxes_mask]
    # map xmin, ymin, xmax, ymax
    # restore bbox
    bboxes, _ = restore_param(bboxes, None, restoreparam)
    bboxes = fix_bbox_boundary(bboxes, origin_size)
    return bboxes, np.squeeze(classes).astype(np.int32), np.squeeze(scores)

  def preprocess_mx(self, image):
    """mxnet yolov3 preprocess"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img = mx.nd.array(img)
    # img = mx.image.imread(image_path) #RGB unint8
    ratio = max(img.shape[1] / self.detected_size[1],
                img.shape[0] / self.detected_size[0])
    resize_size = [int(img.shape[0] / ratio), int(img.shape[1] / ratio)]
    # img = mx.image.imresize(img, resize_size[1], resize_size[0], 2) # AREA
    img = cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=2)
    # img = img.asnumpy()
    pad_left = int((self.detected_size[1] - img.shape[1]) / 2)
    pad_top = int((self.detected_size[0] - img.shape[0]) / 2)
    pad_right = self.detected_size[1] - img.shape[1] - pad_left
    pad_bottom = self.detected_size[0] - img.shape[0] - pad_top
    rescale_param = (ratio, ratio, pad_top, pad_left)
    img = cv2.copyMakeBorder(
        img,
        top=pad_top,
        left=pad_left,
        bottom=pad_bottom,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0])
    # img = mx.nd.array(img)
    # img = mx.nd.image.to_tensor(img) # hwc->chw divide 255
    img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = (img - mean) / std
    # img = mx.nd.image.normalize(img, mean=mean, std=std)
    # img = img.expand_dims(0)
    return img, rescale_param

  def predict(self, images):
    """yolov3_mx forward"""
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
      # this version yolov3 input fix 512*672(h*w) pandding location center
      data, rescale_param = self.preprocess_mx(image)
      input_data = {self.input_names[0]: np.array([data])}
      # print(data.shape, self.input_shapes)
      model = load(self.subgraph_path)
      out = infer(model, input_data)
      print("inference done!")
      input_bboxes = out['yolov30_slice_axis3'][0]
      input_classes = out['yolov30_slice_axis1'][0]
      input_scores = out['yolov30_slice_axis2'][0]
      output_bboxes, output_classes, output_scores = \
          self.coord_map(input_bboxes, input_classes, input_scores, \
            rescale_param, origin_size)
      result_bboxes.append(output_bboxes)
      result_classes.append(output_classes)
      result_probs.append(output_scores)
    if use_batch:
      return (result_bboxes, result_classes, result_probs)
    else:
      return (result_bboxes[0], result_classes[0], result_probs[0])
