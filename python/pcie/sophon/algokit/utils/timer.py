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

"""Calculate running time
"""
import time


class Timer(object):
  """Algorithm runtime calculation class
  """

  def __init__(self):
    self.start = 0
    self.end = 0
    self.times = 0
    self.total_time = 0

  def tic(self):
    """get start time
    """
    self.start = time.time()

  def toc(self):
    """get end time
    """
    self.end = time.time()
    self.total_time += (self.end - self.start)
    self.times += 1
