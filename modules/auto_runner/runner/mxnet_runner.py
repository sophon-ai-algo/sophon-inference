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
from ..common.base_runner import Runner
from ..external.mxnet_functions import load_mxnet_model
from ..external.mxnet_functions import infer_mxnet


class MxnetRunner(Runner):
  """ Running Mxnet models and bmodels.
  """
  def load_graph_cpu(self, model_info):
    paths = dict()
    for key in model_info.keys():
      paths[key] = os.path.join(self.folder, model_info[key])
    model = load_mxnet_model(device='cpu', folder=self.folder, **model_info)
    return model

  def load_graph_gpu(self, model_info):
    paths = dict()
    for key in model_info.keys():
      paths[key] = os.path.join(self.folder, model_info[key])
    model = load_mxnet_model(device='gpu', folder=self.folder, **model_info)
    return model

  def infer_on_cpu(self, index, inputs, required_outputs):
    model = self.models[index]
    outputs = infer_mxnet(model, inputs, required_outputs, device='cpu')
    return outputs

  def infer_on_gpu(self, index, inputs, required_outputs):
    model = self.models[index]
    outputs = infer_mxnet(model, inputs, required_outputs, device='gpu')
    return outputs

  def preprocess_tpu_input_tensors(self, inputs, input_names):
    return inputs

  def postprocess_tpu_output_tensors(self, outputs, required_outputs, output_names):
    # The postfix of '_output' comes from bmnetm. It make the output tensor
    # names respect to symbol.get_internals().list_outputs().
    result = dict()
    for name_ in outputs.keys():
      if name_.endswith("_output"):
        name = name_[0:-7]
      else:
        tokens = name_.split('_')
        name = '_'.join(tokens[0:-1] + [tokens[-1][6:], "sophon_auto"])
      assert name in required_outputs.keys()
      if required_outputs[name] != outputs[name_].shape:
        data = outputs[name_].reshape(required_outputs[name])
      else:
        data = outputs[name_]
      result[name] = data
    return result
