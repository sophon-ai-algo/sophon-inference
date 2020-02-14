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
import tensorflow as tf
from ..common.base_runner import Runner
from ..external.tensorflow_functions import get_graph
from ..external.tensorflow_functions import load_graph_from_memory


class TensorflowRunner(Runner):
  """ Running Tf models and bmodels.
  """
  def load_graph_cpu(self, model_info):
    graph = get_graph(os.path.join(self.folder, model_info['model_path']), \
                                    device='/cpu:0')
    sess = tf.Session(graph=graph)
    return sess
    
  def load_graph_cpu_from_memory(self, graph_bytes):
    graph = load_graph_from_memory(graph_bytes)
    sess = tf.Session(graph=graph)
    return sess

  def load_graph_gpu(self, model_info):
    graph = get_graph(os.path.join(self.folder, model_info['model_path']), \
                                    device='/gpu:0')
    sess = tf.Session(graph=graph)
    return sess

  def infer_on_cpu(self, index, inputs, required_outputs):
    inputs = dict(inputs)
    required_outputs = dict(required_outputs)
    output_names = list(required_outputs.keys())
    outputs = self.models[index].run(output_names, inputs)
    ret_outputs = dict()
    for i, o_name in enumerate(output_names):
      if not self.dynamic:
        assert tuple(required_outputs[o_name]) == outputs[i].shape
      ret_outputs[o_name] = outputs[i]
    return ret_outputs

  def infer_on_gpu(self, index, inputs, required_outputs):
    inputs = dict(inputs)
    required_outputs = dict(required_outputs)
    output_names = list(required_outputs.keys())
    outputs = self.models[index].run(output_names, inputs)
    ret_outputs = dict()
    for i, o_name in enumerate(output_names):
      if not self.dynamic:
        assert tuple(required_outputs[o_name]) == outputs[i].shape
      ret_outputs[o_name] = outputs[i]
    return ret_outputs

  def preprocess_tpu_input_tensors(self, inputs, input_names):
    ret_inputs = dict()
    for name_ in inputs:
      name = name_.split(':')[0]
      found = False
      found_name = None
      final_name = None
      for i in input_names:
        head_name = '_arg_{0}_'.format(name)
        if i.find(head_name) != 0:
          continue
        replaced_name = i.replace(head_name, '')
        if len(replaced_name) == 3:
        #if i.find('_arg_{0}_'.format(name)) == 0:
          if found:
            raise RuntimeError( \
                    'The name of {0} and {1}'.format(name, found_name) + \
                    ' has same beginning, please change one of them.')
          else:
            found = True
            found_name = name
            final_name = i
      assert found
      if self.layout == 'NHWC' and len(inputs[name_].shape) == 4:
        ret_inputs[final_name] = np.transpose(inputs[name_], \
                                              [0, 3, 1, 2]).copy()
      else:
        ret_inputs[final_name] = inputs[name_]
    return ret_inputs

#  def preprocess_tpu_input_tensors_old(self, inputs, input_names):
#    ret_inputs = dict()
#    for name_ in inputs:
#      name = name_.split(':')[0]
#      found = False
#      found_name = None
#      final_name = None
#      for i in input_names:
#        if i.find('_arg_{0}_'.format(name)) == 0:
#          if found:
#            raise RuntimeError( \
#                    'The name of {0} and {1}'.format(name, found_name) + \
#                    ' has same beginning, please change one of them.')
#          else:
#            found = True
#            found_name = name
#            final_name = i
#      assert found
#      if self.layout == 'NHWC' and len(inputs[name_].shape) == 4:
#        ret_inputs[final_name] = np.transpose(inputs[name_], \
#                                              [0, 3, 1, 2]).copy()
#      #elif len(inputs[name_].shape) < 4:
#      #  new_shape = list(inputs[name_].shape)
#      #  while len(new_shape) < 4:
#      #    new_shape.append(1)
#      #  ret_inputs[final_name] = inputs[name_].reshape(new_shape)
#      else:
#        ret_inputs[final_name] = inputs[name_]
#    return ret_inputs

  def postprocess_tpu_output_tensors(self, outputs, required_outputs, output_names):
    ret_outputs = dict()
    for name_ in outputs:
      name = name_ + ':0'
      r_shape = required_outputs[name]
      if len(r_shape) == 4 and self.layout == 'NHWC':
        ret_outputs[name] = np.transpose(outputs[name_], [0, 2, 3, 1])
      elif len(r_shape) < 4:
        ret_outputs[name] = outputs[name_].reshape(required_outputs[name])
      else:
        ret_outputs[name] = outputs[name_]
    return ret_outputs
