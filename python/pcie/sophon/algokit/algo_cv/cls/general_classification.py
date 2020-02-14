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

"""The implementation of general classification
"""
from __future__ import print_function
import cv2
import numpy as np
from ...engine.base_engine import BaseEngine
from ...engine import Engine


class GeneralClassification(BaseEngine):
  """Construct a general clssifier
  """

  def __init__(self, arch, mode, xform, num_classes=1000):
    super(GeneralClassification, self).__init__()
    self.mode = mode
    self.num_classes = num_classes
    self.xform = xform
    self.net = Engine(arch, mode)
    self.input_names = arch["input_names"]
    self.output_names = arch["output_names"]

  def preprocess(self, images):
    """preprocess
        resize
        submean
        transpose...
    """
    output = []
    for image in images:
      output.append(self.xform_img(image, self.xform)[0])
    return np.array(output)

  def preprocess_test(self, images):
    """ classification preprocess test
        """
    output = []
    for image in images:
      for param in self.xform:
        param_0 = param[0]
        param_1 = param[1]
        if param_0 == "resize":
          size = param_1
      image = np.array(image, dtype=np.float32)
      image = cv2.resize(
          image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
      image[:, :, 0] -= 104.00698853
      image[:, :, 1] -= 116.66876984
      image[:, :, 2] -= 122.67891693
      image = np.transpose(image, (2, 0, 1))
      output.append(image)
    return np.array(output)

  def predict(self, images):
    if not isinstance(images, list):
      images = [images]
    # prerocess_data = self.preprocess_test(images)
    prerocess_data = self.preprocess(images)
    input_data = {self.input_names[0]: prerocess_data.astype(np.float32)}
    out = self.net.predict(input_data)[self.output_names[0]]
    # top1 && top5
    if out.ndim > 2:
      out = out.reshape((out.shape[0], out.shape[1]))
    sort_idx = np.argsort(-out)
    top1_idx = sort_idx[:, 0].reshape((-1, 1)).tolist()
    top5_idx = sort_idx[:, :5].tolist()
    top1_score = []
    top5_score = []

    for i, _ in enumerate(top1_idx):
      temp1_score = []
      temp5_score = []
      for idx in top1_idx[i]:
        temp1_score.append(out[i][idx])
      top1_score.append(temp1_score)
      for idx in top5_idx[i]:
        temp5_score.append(out[i][idx])
      top5_score.append(temp5_score)

    return {"top1_idx": top1_idx, "top1_score": top1_score}, \
        {"top5_idx": top5_idx, "top5_score": top5_score}
