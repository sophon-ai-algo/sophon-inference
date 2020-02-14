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

"""The BM1682 model engine
"""
from __future__ import print_function
from __future__ import division
import numpy as np
import sophon.sail as sail # engine core lib
from .base_engine import BaseEngine
from ..kitconfig import ChipMode


class BM1682Engine(BaseEngine):
  """Construct BM1682 engine from lib sail
  """

  def bm1682_init(self):
    """Init bm1682 inference
    """
    inference = sail.Engine(self.context_path, self.tpus, sail.IOMode.SYSIO)
    return inference

  def __init__(self, context_path, is_dynamic, tpus, input_shapes, input_names,
               output_names):
    """Init BM1682Engine

    Args:
        All arguments init from net config json file
        context_path(str). context ir path
        is_dynamic(bool). dynamic status
        tpus(str). tpu number
        input_shapes(List). net input shapes (max shape for dynamic)
    """
    super(BM1682Engine, self).__init__()
    self.is_dynamic = is_dynamic
    self.context_path = context_path
    self.tpus = int(tpus)
    self.inference = self.bm1682_init()  # init bm1682 inference
    self.input_shapes = input_shapes
    self.graph_name = \
        self.inference.get_graph_names()[0]
    self.input_names = \
        self.inference.get_input_names(self.graph_name)
    for input_name in input_names:
      if input_name not in self.input_names:
        raise ValueError("not a valid input name in config file")
    self.output_names = \
        self.inference.get_output_names(self.graph_name)
    for output_name in output_names:
      if output_name not in self.output_names:
        raise ValueError("not a valid output name in config file")

  def _sync_predict(self, inputs):
    """synchronous inference
    """
    for input_name in self.input_names:
      inputs[input_name] = \
              np.array(inputs[input_name], dtype=np.float32)
    output = self.inference.process(self.graph_name, inputs)
    for output_name in self.output_names:
      output_shape = output[output_name].shape
      print("{} output shape: {}".format(output_name, tuple(output_shape)))
    return output

  def predict(self, inputs):
    """The inference interface of BM1682Engine

    status:
        synchronous: _sync_predict
    """
    output = {}
    output = self._sync_predict(inputs)  # for sync
    return output
