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

"""The function of mathematical calculation
"""
from __future__ import division
import numpy as np


def softmax(in_array, theta=1.0, axis=None):
  """Compute the softmax of each element along an axis of in_array.

    Args:
        in_array: ndarray. Probably should be floats.
        theta (optional): float parameter, used as a multiplier
            prior to exponentiation. Default = 1.0
        axis (optional): axis to compute values along. Default
            is the first non-singleton axis.
    Returns:
        an array the same size as X. The result will sum to 1
        along the specified axis.
    """
  # make in_array at least 2d
  pro_array = np.atleast_2d(in_array)
  # find axis
  if axis is None:
    axis = next(j[0] for j in enumerate(pro_array.shape) if j[1] > 1)
  # multiply pro_array against the theta parameter,
  pro_array = pro_array * float(theta)
  # subtract the max for numerical stability
  pro_array = pro_array - np.expand_dims(np.max(pro_array, axis=axis), axis)
  # exponentiate pro_array
  pro_array = np.exp(pro_array)
  # take the sum along the specified axis
  ax_sum = np.expand_dims(np.sum(pro_array, axis=axis), axis)
  # finally: divide elementwise
  output = pro_array / ax_sum
  # flatten if in_array was 1D
  if len(in_array.shape) == 1:
    output = output.flatten()
  return np.array(output).copy()


def get_l2norm(feature):
  """Calculate the Euclidean norm
    """
  return np.asscalar(np.linalg.norm(feature, 2))
