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

"""The engine of sailalgokit

import sail lib for model deploy
"""
from ..kitconfig import ChipMode

class Engine(object):
  """Construct a inference engine
  """

  def __init__(self, arch, mode):
    if mode in [ChipMode.BM1682, ChipMode.BM1684]:
      from .bm_engine import BMEngine
      self._model = BMEngine(arch, mode)
    else:
      raise ValueError('unsupport mode: {}'.format(mode.value))

  def predict(self, input_data):
    """network inference
    """
    return self._model.predict(input_data)
