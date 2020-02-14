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

"""The implementation of general classification_tf
"""
from __future__ import print_function
import cv2
import numpy as np
from ...engine.base_engine import BaseEngine
from ...engine import Engine


class GeneralClassificationTF(BaseEngine):
  """Construct a general clssifier
  """

  def __init__(self, arch, mode, xform, num_classes=1000):
    super(GeneralClassificationTF, self).__init__()
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

  def inception_preprocess(self, images):
    """ tensorflow classification inception type preprocess
    """
    output = []
    for image in images:
      for param in self.xform:
        param_0 = param[0]
        param_1 = param[1]
        if param_0 == "resize":
          size = param_1
      central_fraction = 0.875
      new_h = int(image.shape[0] * central_fraction)
      new_w = int(image.shape[1] * central_fraction)
      h_begin = int((image.shape[0] - new_h) / 2.0)
      w_begin = int((image.shape[1] - new_w) / 2.0)
      image = image[h_begin:(h_begin+new_h), w_begin:(w_begin+new_w)]
      image = cv2.resize(
          image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
      image = np.transpose(image, (2, 0, 1))
      image = image / 255.0
      image -= 0.5
      image *= 2.0
      image = image[(2, 1, 0), :, :]
      # image = image[np.newaxis, :]
      output.append(image)
    return np.array(output)

  def predict(self, images):
    if not isinstance(images, list):
      images = [images]
    prerocess_data = self.inception_preprocess(images)
    # prerocess_data = self.preprocess(images)
    input_data = {self.input_names[0]: prerocess_data.astype(np.float32)}
    out = self.net.predict(input_data)[self.output_names[0]]
    if self.num_classes == 1001:
      out = out[:, 1:]  # 1001 classes idx offset
    # top1 && top5
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
