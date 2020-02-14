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

"""The algorithm basefactory
"""
from __future__ import print_function
import os
from abc import ABCMeta, abstractmethod
import six


@six.add_metaclass(ABCMeta)
class BaseFactory(object):
  """Abstract factory

  Create an algo object instance by input algo type.

  Attributes:
      model_path: algorithm model path.
  """

  def __init__(self, model_path=None):
    """Inits Abstract factory."""
    # localpath = os.path.dirname(os.path.realpath(__file__))
    localpath = os.getenv('SOPHON_MODEL_DIR',os.getenv('HOME'))
    if model_path is None:  # model_path: bm1682/bm1684 model path
      # model_path = os.path.join(localpath, '../../models')
      model_path = os.path.join(localpath, '.sophon/models')
      # print ("bm1682/bm1684 bmodel path: {}".format(model_path))
    else:
      model_path = os.path.join(model_path, 'models')
    self.model_path = model_path

  def set_model_path(self, model_path):
    """Set model path with input model_path"""
    self.model_path = model_path

  @abstractmethod
  def create(self):
    """Abstractmethod Implemented by derived classes"""
    raise NotImplementedError
