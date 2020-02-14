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
import shutil
#import bmnetc as bm

from ..common.base_splitter import Splitter
from ..external.caffe_functions import get_unsupported_caffe_layers


class CaffeSplitter(Splitter):
  """ Split Caffe graph into subgraphs.
  """
  def initialize(self):
    self.platform = 'caffe'
    proto_path = self.model_descriptor['proto_path']
    weight_path = self.model_descriptor['weight_path']
    unsupported_layers = get_unsupported_caffe_layers(proto_path)
    if unsupported_layers:
      raise RuntimeError("This layers doesnt't support by now: \n {}"\
                         .format(unsupported_layers))
    self.proto_path = proto_path
    self.weight_path = weight_path
    self.ops = "nothing by now"
    self.input_tensors = "nothing by now"
    self.input_shapes = self.model_descriptor['input_shapes']
    self.output_shapes = self.model_descriptor['output_shapes']
    self.input_names = list(self.input_shapes.keys())
    self.output_names = list(self.output_shapes.keys())

  def convert_and_split(self, save_folder):
    ''' override from base_splitter
    '''
    self.graph_infos['graph_num'] = 1
    sub_prototxt_0 = os.path.join(save_folder, "graph_0.prototxt")
    sub_caffemodel_0 = os.path.join(save_folder, "graph_0.caffemodel")
    shutil.copy(self.proto_path, sub_prototxt_0)
    shutil.copy(self.weight_path, sub_caffemodel_0)
    sub_info = dict()
    model_info = dict()
    model_info['proto'] = 'graph_0.prototxt'
    model_info['weight'] = 'graph_0.caffemodel'
    sub_info['model_info'] = model_info
    sub_info['inputs'] = self.input_names
    sub_info['outputs'] = self.output_names
    sub_info['device'] = 'tpu'
    sub_info['context_dir'] = 'graph_ir_0'
    self.graph_infos['graphs'].append(sub_info)

    for name in self.input_names:
      tensor_node = {"shape": self.input_shapes[name]}
      tensor_node["dtype"] = "float32"
      tensor_node["attr"] = "input"
      self.graph_infos["tensors"][name] = tensor_node
    for name in self.output_names:
      tensor_node = {"shape": self.output_shapes[name]}
      tensor_node["dtype"] = "float32"
      tensor_node["attr"] = "output"
      self.graph_infos["tensors"][name] = tensor_node
    self.save_graph_infos(save_folder)
    self.destroy()    

  def get_op_name(self, op):
    raise NotImplementedError

  def if_op_const(self, operator):
    raise NotImplementedError

  def is_op_support(self, operator):
    raise NotImplementedError

  def is_op_compute(self, op):
    raise NotImplementedError

  def is_op_dangerous(self, op):
    raise NotImplementedError

  def is_input_op(self, op):
    raise NotImplementedError

  def is_output_op(self, op):
    raise NotImplementedError

  def get_inputs_list(self, op):
    raise NotImplementedError

  def save_subgraph(self, subgraph, save_folder, index, tensors):
    raise NotImplementedError

  def infer_output_tensors(self, save_folder, model_info, \
                          sub_inputs, sub_outputs, \
                          tensors):
    raise NotImplementedError

  def get_tensor_dtype(self, tensor_name):
    raise NotImplementedError

  def destroy(self):
    pass

