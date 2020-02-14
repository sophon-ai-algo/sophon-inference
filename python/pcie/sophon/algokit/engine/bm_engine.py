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

"""BMmodel engine

which can create engine based on chipmode
"""
from __future__ import print_function
import numpy as np
from .base_engine import BaseEngine
from ..kitconfig import ChipMode


class BMEngine(BaseEngine):
  """Construct BMmodel inference engine
  """

  def __init__(self, arch, mode):
    super(BMEngine, self).__init__()
    if mode is ChipMode.BM1682:
      from .bm1682_engine import BM1682Engine as Engine
    elif mode is ChipMode.BM1684:
      from .bm1684_engine import BM1684Engine as Engine
    else:
      raise ValueError('not a valid mode: {}'.format(mode))
    self.mode = mode
    self.nets = Engine(**arch)
    self.input_info = arch['input_names']
    self.max_batch_size = [item[0] for item in arch['input_shapes']][0]

  def get_batch_list(self, num):
    """Get inference batch list from max_input_batch and
       real input batch
    """
    batch_list = []  # [max_batch, max_batch, num - n * max_batch]
    while num > 0:
      if self.max_batch_size <= num:
        batch_num = self.max_batch_size
        batch_list.append(batch_num)
        num -= batch_num
      else:
        batch_list.append(num)
        num -= num
    print("inference batch list: {}".format(batch_list))
    return batch_list

  def predict(self, input_data):
    """Split the input data with batch list
           call the interface of BM1682/BM1684Engine to do inference
        """
    if len(input_data) == 1:
      num = input_data[self.input_info[0]].shape[0]  # single input
    else:
      num = 1  # multi inputs
    batch_list = self.get_batch_list(num)
    batch_num = 0  # done batch
    if len(input_data) == 1:  # single input can get sub_batch
      for sub_batch in batch_list:  # working batch
        subbatch_data = {}
        subbatch_data[self.input_info[0]] = \
                np.ascontiguousarray(\
                  input_data[self.input_info[0]][batch_num:sub_batch+batch_num])
        self.time.tic()
        partial_out = self.nets.predict(subbatch_data)  # do inference
        self.time.toc()
        if batch_num == 0:  # first forward (sync batch_num == 0)
          out = partial_out
        else:
          for key in out:  # concatenate output
            out[key] = np.concatenate((out[key], partial_out[key].copy()),
                                      axis=0)
        batch_num += sub_batch
    else:  # multi_inputs batch_list = [1] for sync
      for sub_batch in batch_list:
        subbatch_data = {}
        if sub_batch == 1:
          subbatch_data = input_data
        else:
          for input_name in self.input_info:
            subbatch_data[input_name] = np.ndarray((0), dtype=np.float32)
        self.time.tic()
        partial_out = self.nets.predict(subbatch_data)
        self.time.toc()
        if batch_num == 0:  # first forward
          out = partial_out
        else:
          for key in out:
            out[key] = np.concatenate((out[key], partial_out[key].copy()),
                                      axis=0)
        batch_num += sub_batch
    assert batch_num == sum(batch_list)
    return out
