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


class SemanticSegmentationDEEPLABV3MOBILENETV2TF(BaseEngine):
  """Construct deeplabv3_mobilenetv2 semantic segmemtation
  """

  def __init__(self, source_path, subgraph_path, tfmodel_path, framework,
               input_names, output_names, input_shapes, layout, is_dynamic,
               process_size, target, conf_threshold):
    super(SemanticSegmentationDEEPLABV3MOBILENETV2TF, self).__init__()
    # process_size: (h, w)
    # image will be resize before processing detection
    self.source_path = source_path
    self.subgraph_path = subgraph_path
    self.tfmodel_path = tfmodel_path
    self.framework = framework
    self.input_names = input_names
    self.input_shapes = input_shapes
    self.process_size = process_size
    self.conf_threshold = conf_threshold
    self.tensors_dict = {}
    if len(self.input_names) == len(self.input_shapes):
      for input_name, input_shape in zip(input_names, input_shapes):
        self.tensors_dict[input_name] = np.ndarray(
            input_shape, dtype=np.float32)
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

  def predict(self, images):
    """deeplabv3_mobilenetv2 forward
    """
    self.time.tic()
    if isinstance(images, list):
      use_batch = True
    else:
      use_batch = False
      images = [images]

    # inference
    for image in images:
      # origin_size = image.shape
      # this version deeplabv3_mobilenetv2 input size fix in (513,513)
      # data, rescale_param
      data, _ = self.rescale_image(image, self.process_size, True)
      data = data[..., [2, 1, 0]]  # BGR2RGB PIL.Image read as RGB
      input_data = {self.input_names[0]: np.array([data])}
      model = load(self.subgraph_path)
      out = infer(model, input_data)
      print("inference done!")
      semanticpredictions = out['SemanticPredictions:0']
    if use_batch:
      return semanticpredictions
    else:
      return semanticpredictions
