""" Copyright 2016-2022 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import numpy as np
from ..common.base_runner import Runner


class CaffeRunner(Runner):
  """ Running Tf models and bmodels.
  """
  def load_graph_cpu(self, model_info):
    raise NotImplementedError
    
  def load_graph_cpu_from_memory(self, graph_bytes):
    raise NotImplementedError

  def load_graph_gpu(self, model_info):
    raise NotImplementedError

  def infer_on_cpu(self, index, inputs, required_outputs):
    raise NotImplementedError

  def infer_on_gpu(self, index, inputs, required_outputs):
    raise NotImplementedError

  def preprocess_tpu_input_tensors(self, inputs, input_names):
    return inputs


  def postprocess_tpu_output_tensors(self, outputs, required_outputs, output_names):
    return outputs
