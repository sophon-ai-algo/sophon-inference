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

"""The base model engine

Which include some img preprocess operations
"""
from __future__ import print_function
from __future__ import division
from abc import ABCMeta, abstractmethod
import cv2
import six
import numpy as np
from ..utils.timer import Timer


@six.add_metaclass(ABCMeta)
class BaseEngine(object):
  """Construct a BaseEngine
  """

  def __init__(self):
    self.time = Timer()
    self.profile = False

  @abstractmethod
  def predict(self, inputs):
    """Implemented by an inherited class
    """
    raise NotImplementedError

  def __call__(self, *image, **kwargs):
    """Object instances can behave like functions
    """
    return self.predict(*image, **kwargs)

  @staticmethod
  def xform_img(image, params):
    """Basic img process
    Args:
        image: ndarray. Input image
        params: list. Process params
        ex. [('resize', 2), ('scale', 1/255.)]

    Returns:
        4-dim ndarray
    """
    image = np.array(image, dtype=np.float32)
    for param in params:
      transform_proc = param[0]
      transform_param = param[1]
      # resize
      if transform_proc == 'resize':
        if isinstance(transform_param, list):
          image = cv2.resize(
              image, (transform_param[1], transform_param[0]),
              interpolation=cv2.INTER_LINEAR)
        else:
          image = cv2.resize(
              image, (transform_param, transform_param),
              interpolation=cv2.INTER_LINEAR)
      # sub mean
      if transform_proc == 'submean':
        if isinstance(transform_param, list):
          image[:, :, 0] -= transform_param[0]
          image[:, :, 1] -= transform_param[1]
          image[:, :, 2] -= transform_param[2]
        else:
          image -= transform_param
      # h w c -> c h w
      if transform_proc == 'transpose':
        transpose_channel = transform_param
        image = np.transpose(
            image,
            (transpose_channel[0], transpose_channel[1], transpose_channel[2]))
      # (bgr) h w -> (rgb) h w
      if transform_proc == 'bgr2rgb':
        if transform_param is True:
          image = image[[2, 1, 0], ...]
      # img scale
      if transform_proc == 'scale':
        image *= transform_param
    # return a 4-dim array
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    return image

  @staticmethod
  def bgr2rgb(image):
    """ img channel transform bgr2rgb
        hwc rgb2bgr
        """
    return image[..., [2, 1, 0]]

  @staticmethod
  def rescale_image(image, size, pad):
    """Rescale the input image
        Args:
            image: ndarray. input image [h,w,c]
            size: list. rescale size [h,w]
            pad: bool. is_padding

        Returns:
            image, tuple(scale_y, scale_x, pad_top, pad_left)
        """
    if size is None:
      return image, (1, 1, 0, 0)
    if image.shape[0] == size[0] and image.shape[1] == size[1]:
      return image, (1, 1, 0, 0)

    if isinstance(size[0], int) and isinstance(size[1], int):
      if pad:  # w&&h scale with same ratio pad 0
        ratio = max(image.shape[1] / size[1], image.shape[0] / size[0])
        if ratio != 1.:
          image = cv2.resize(
              image, (int(image.shape[1] / ratio), int(image.shape[0] / ratio)),
              interpolation=cv2.INTER_LINEAR)
        pad_left = int((size[1] - image.shape[1]) / 2)
        pad_top = int((size[0] - image.shape[0]) / 2)
        pad_right = size[1] - image.shape[1] - pad_left
        pad_bottom = size[0] - image.shape[0] - pad_top
        rescale_param = (ratio, ratio, pad_top, pad_left)
        if pad_left != 0 or pad_top != 0 or pad_right != 0 or pad_bottom != 0:
          image = cv2.copyMakeBorder(
              image,
              top=pad_top,
              left=pad_left,
              bottom=pad_bottom,
              right=pad_right,
              borderType=cv2.BORDER_CONSTANT,
              value=[0, 0, 0])
      else:
        rescale_param = (image.shape[1] / size[1], image.shape[0] / size[0],
                         0, 0)
        print('rescale in aspect ratio (H, W): {}'.format(rescale_param))
        image = cv2.resize(
            image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    else:
      raise TypeError('not valid ', size)
    return image, rescale_param
