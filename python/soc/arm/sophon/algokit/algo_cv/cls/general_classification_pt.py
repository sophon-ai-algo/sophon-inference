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

"""The implementation of general classification_pt
"""
from __future__ import print_function
import cv2
import numpy as np
from PIL import Image
from ...engine.base_engine import BaseEngine
from ...engine import Engine
from ...utils.extend_math import softmax


class GeneralClassificationPT(BaseEngine):
  """Construct a general classifier
  """

  def __init__(self, arch, mode, xform, num_classes=1000):
    super(GeneralClassificationPT, self).__init__()
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

  def pt_cls_preprocess(self, images):
    """ pytorch imagenet classification preprocess
        """
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)).astype(np.float32)
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1)).astype(np.float32)
    output = []
    for image in images:
      for param in self.xform:
        param_0 = param[0]
        param_1 = param[1]
        if param_0 == "resize":
          size = param_1
      # image = cv2.resize(image, (size[1], size[0]), \
      #   interpolation=cv2.INTER_LINEAR)
      # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      org_h, org_w = image.shape[:2]
      scale = 256.0 / min(org_h, org_w)
      resize_h, resize_w = int(scale * org_h), int(scale * org_w)
      image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      image = np.asarray(image.resize((resize_w, resize_h), \
          resample=Image.BILINEAR))
      # image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_LINEAR)
      offset_h, offset_w = (resize_h - size[0]) // 2, (resize_w - size[1]) // 2
      image = image[offset_h:offset_h + size[0], offset_w:offset_w + size[1]]
      image = np.transpose(image, (2, 0, 1)).astype(np.float32)
      image /= 255.0
      image = (image - mean) / std
      output.append(image)
    return np.array(output)

  def predict(self, images):
    if not isinstance(images, list):
      images = [images]
    prerocess_data = self.pt_cls_preprocess(images)
    # prerocess_data = self.preprocess(images)
    input_data = {self.input_names[0]: prerocess_data.astype(np.float32)}
    out = self.net.predict(input_data)[self.output_names[0]]
    if self.num_classes == 1001:
      out = out[:, 1:]  # 1001 classes idx offset
    # top1 && top5
    out = softmax(out, axis=1)
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
