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

from __future__ import print_function
import os
#import re
import json
import copy
#import numpy as np
import mxnet as mx
from mxnet import gluon
import bmnetm
from ..common.base_splitter import Splitter
from ..external.mxnet_functions import load_json_file
from ..external.mxnet_functions import get_index_dict
from ..external.mxnet_functions import get_input_names_from_json
from ..external.mxnet_functions import get_output_names_from_json
from ..external.mxnet_functions import node_is_weight
from ..external.mxnet_functions import get_all_ops
from ..external.mxnet_functions import get_input_names_from_file
from ..external.mxnet_functions import get_output_names_from_file
from ..external.mxnet_functions import sym_has_params
from ..external.mxnet_functions import get_prefix_and_epoch
from ..external.mxnet_functions import load_mxnet_model
from ..external.mxnet_functions import infer_mxnet


def get_more_than_x(numbers, value):
  """ Get numbers more than x in a list
  """
  ret = list()
  for i in numbers:
    if i >= value:
      ret.append(i)
  return ret


def get_input_tensors(sub_graph):
  """ Get all input tensor names of a sub_graph.

  Args:
    sub_graph: A SubGraph instance.

  Returns:
    A set contains all input tensor names of the sub_graph.
  """
  input_tensors = copy.deepcopy(sub_graph.input_ops)
  for name in sub_graph.input_subgraphs:
    input_tensors |= sub_graph.input_subgraphs[name]
  return input_tensors

def get_output_tensors(sub_graph):
  """ Get all output tensor names of a sub_graph.

  Args:
    sub_graph: A SubGraph instance.

  Returns:
    A set contains all output tensor names of the sub_graph.
  """
  output_tensors = copy.deepcopy(sub_graph.output_ops)
  for name in sub_graph.output_subgraphs:
    output_tensors |= sub_graph.output_subgraphs[name]
  return output_tensors

def find_arg_nodes(nodes, input_names, ops, index_dict):
  """ Find indexes of all argument nodes. Argument nodes are input tensors
      and weights.

  Args:
    nodes: A json object contain all the nodes in a mxnet json file.
    input_names: Names of input tensors.
    ops: Names of operaters.
    index_dict: A dict denotes relationships between name and index of nodes.

  Returns:
    A sorted list contains indexes of all argument nodes.
  """
  arg_nodes = set(range(-len(input_names), 0))
  for operator in ops:
    index = index_dict[operator]
    parent_ids = set([parent[0] for parent in nodes[index]["inputs"] \
                      if node_is_weight(nodes[parent[0]])])
    arg_nodes |= parent_ids
  arg_nodes_list = list(arg_nodes)
  arg_nodes_list.sort()
  return arg_nodes_list

def find_heads(output_tensors, index_dict):
  """ Find indexes of all heads. Heads stand for output tensors.

  Args:
    # nodes: A json object contain all the nodes in a mxnet json file.
    output_tensors: Names of output tensors.
    index_dict: A dict denotes relationships between name and index of nodes.

  Returns:
    A sorted list contains indexes of heads.
  """
  heads = list(set([index_dict[op] for op in output_tensors]))
  heads.sort()
  return heads

def find_split_sons(raw_nodes, parent_id, sub_ops_ids):
  """ Find ids of sons given a parent id.

  Args:
    raw_nodes: A json object contain all the nodes of the raw mxnet json file.
    parent_id: Id of a node.
    sub_ops_ids: Ids of all ops in a sub graph.

  Returns:
    Ids of sons of the specified parent.
  """
  split_ids = set()
  if raw_nodes[parent_id]["op"] != "SliceChannel":
    return split_ids
  for op_id in sub_ops_ids:
    for lst in raw_nodes[op_id]["inputs"]:
      if lst[0] == parent_id:
        split_ids.add(lst[1])
  split_ids_list = list(split_ids)
  split_ids_list.sort()
  return split_ids_list

def gen_json(raw_json, sub_graph, index_dict, sub_json_path):
  """ Generate json file of a subgraph.

  Args:
    raw_json: Json object read from json file of raw model.
    sub_graph: A SubGraph instance.
    index_dict: A dict denotes relationships between name and index of nodes.
    sub_json_path: Path of json file to save.

  Returns:
    None.
  """
  data = {"nodes":list(), "arg_nodes":list(), "heads":list(), "attrs":dict()}
  nodes = raw_json["nodes"]
  input_tensors = get_input_tensors(sub_graph)
  output_tensors = get_output_tensors(sub_graph)
  ops_ids = [index_dict[op] for op in sub_graph.ops]
  input_ids = [index_dict[op] for op in input_tensors]
  input_split_ids = list()
  input_names = list()
  for tensor in input_tensors:
    parent_id = index_dict[tensor]
    split_ids = find_split_sons(nodes, parent_id, ops_ids)
    if not split_ids:
      input_names.append(tensor)
      data["nodes"].append({"op":"null", "name":tensor, "inputs":[]})
      continue
    input_split_ids.append(parent_id)
    for i in split_ids:
      name = tensor + "_" + str(i) + "_sophon_auto"
      input_names.append(name)
      data["nodes"].append({"op":"null", "name":name, "inputs":[]})
  arg_nodes = find_arg_nodes(nodes, input_names, \
                                  sub_graph.ops, index_dict)
  total_node_ids = list((set(arg_nodes) | set(ops_ids)) - set(input_ids))
  total_node_ids.sort()
  # heads = find_heads(nodes, output_tensors, index_dict)
  heads = find_heads(output_tensors, index_dict)
  tmp_total_node_ids = get_more_than_x(total_node_ids, 0)
  for i in tmp_total_node_ids:
    #if i >= 0:
    data["nodes"].append(nodes[i])
  new_index_dict = get_index_dict(data["nodes"])
  for node in data["nodes"]:
    inputs = list()
    for i in node["inputs"]:
      if i[0] in input_split_ids:
        new_input_name = nodes[i[0]]["name"] + "_" + str(i[1]) + \
                         "_sophon_auto"
        inputs.append([new_index_dict[new_input_name], 0, 0])
      else:
        inputs.append([new_index_dict[nodes[i[0]]["name"]], i[1], i[2]])
    node["inputs"] = inputs
  data["arg_nodes"] = [total_node_ids.index(i) for i in arg_nodes]
  data["attrs"] = raw_json["attrs"]
  data["heads"] = list()
  for i in heads:
    if nodes[i]["op"] == "SliceChannel":
      for j in range(int(nodes[i]["attrs"]["num_outputs"])):
        data["heads"].append([new_index_dict[nodes[i]["name"]], j, 0])
    else:
      data["heads"].append([new_index_dict[nodes[i]["name"]], 0, 0])
  formatted = json.dumps(data, indent=2, sort_keys=False)
  with open(sub_json_path, 'w') as f_save:
    f_save.write(formatted)

def gen_params(raw_params_path, sub_json_path, sub_params_path, input_tensors):
  """ Get features which are intermediate results of the model.

  Args:
    raw_params_path: Path of params file of the raw mxnet model.
    sub_json_path: Path of json file of the submodel.
    sub_params_path: Path of params file of the submodel.
    input_tensors: A list contains all input tensor names and shapes.
                  Format: [(tensor_name, numpy.ndarray), ]

  Returns:
    True for save parameters to file, False for no parameters and not save.
  """
  sym = mx.sym.load(sub_json_path)
  has_params = sym_has_params(sym, [item[0] for item in input_tensors])
  output_names = get_output_names_from_file(sub_json_path)
  internals = sym.get_internals()
  outputs_ops = sym.get_internals().list_outputs()

  outputs = list()
  for name in output_names:
    if name.endswith("sophon_auto"):
      tokens = name.split('_')
      out_name = "_".join(tokens[0:-3] + ["output" + tokens[-3]])
    else:
      out_name = name + '_output'

    if out_name not in outputs_ops:
      print("Wrong name: {}".format(name))
      return None
    outputs.append(internals[out_name])
  inputs = list()
  for item in input_tensors:
    tensor_name = item[0]
    inputs.append(mx.sym.var(tensor_name))
  net = gluon.nn.SymbolBlock(outputs=outputs, inputs=inputs)
  # Set the params
  net.collect_params().load(raw_params_path, ctx=mx.cpu(), ignore_extra=True)
  input_data = [mx.nd.array(item[1]) for item in input_tensors]
  outputs = net(*input_data)
  prefix, epoch = get_prefix_and_epoch(sub_params_path)
  prefix = os.path.join(os.path.dirname(sub_params_path), prefix)
  net.export(prefix, epoch=epoch)
  return has_params


class MxnetSplitter(Splitter):
  """ Split a Mxnet model into submodels.
  """
  def initialize(self):
    """ Load graph information from mxnet model descriptor.

    ops: Information of all operators, exluding weight nodes.
         Format: {op_name: (op_type, [parent_name])}.
    input_ops: list, names of all input tensors.
    output_ops: list, names of all output tensors.
    json_path: Path to symbol file.
    params_path: Path to parameter file.
    is_dynamic: True means input tensor shapes may change.
    sym_json: Json read from symbol file.
    index_dict: Relationships between name and index of nodes.
                Format: {node_name: node_index}
    input_names: Input tensor names.
    output_names: Output tensor names.
    prefix: Prefix of saved model.
    epoch: Epoch number of saved model.
    """
    self.platform = 'mxnet'
    required_args = ["json_path", "params_path", "dynamic", "input_tensors"]
    for arg in required_args:
      assert arg in self.model_descriptor.keys()
    self.json_path = self.model_descriptor["json_path"]
    self.ops, self.input_ops, self.output_ops = get_all_ops(self.json_path)
    self.params_path = self.model_descriptor["params_path"]
    self.sym_json = load_json_file(self.json_path)
    self.index_dict = get_index_dict(self.sym_json["nodes"])
    self.input_names = get_input_names_from_json(self.sym_json)
    self.output_names = get_output_names_from_json(self.sym_json)
    self.prefix, self.epoch = get_prefix_and_epoch(self.params_path)
    self.input_tensors = self.model_descriptor["input_tensors"]

  def get_op_name(self, op_name):
    return op_name

  def is_op_support(self, op_name):
    param = {"op": self.ops[op_name][0]}
    if self.ops[op_name][0] == 'null' or bmnetm.op_support(param):
      return True
    return False

  def is_op_compute(self, op_name):
    compute_list = [
        'Convolution',
        'Pooling',
        'Activation',
        'elemwise_add',
        'FullyConnected',
        'BatchNorm'
    ]
    if self.ops[op_name][0] in compute_list:
      return True
    return False

  def is_op_dangerous(self, op_name):
    dangerous_list = [
    ]
    if self.ops[op_name][0] in dangerous_list:
      return True
    return False

  def is_input_op(self, op_name):
    if op_name in self.input_ops:
      return True
    return False

  def is_output_op(self, op_name):
    if op_name in self.output_ops:
      return True
    return False

  def get_inputs_list(self, op_name):
    return self.ops[op_name][1]

  def destroy(self):
    pass

  def save_subgraph(self, graph, save_folder, index, tensors):
    """ Save submodel to files.

    Args:
      graph: A SubGraph instances.
      save_folder: Folder path to save json file and params file.
      index: Index of subgraph.
      tensors: A dict contains tensor names and values.

    Returns:
      model_info: A dict contains model information.
                  Format: {"json": json_name, "params": params_name}
      input_names: list, input tensor names of the submodel.
      ouput_names: list, output tensor names of the submodel.
    """
    model_info = dict()
    json_name = '{}_{}-symbol.json'.format(self.prefix, index)
    params_name = '{}_{}-{:0>4}.params'.format(self.prefix, index, self.epoch)
    json_path = os.path.join(save_folder, json_name)
    gen_json(self.sym_json, graph, self.index_dict, json_path)
    input_names = get_input_names_from_file(json_path)
    input_tensors = [(i, tensors[i]) for i in input_names]
    params_path = os.path.join(save_folder, params_name)
    has_params = gen_params(self.params_path, json_path, \
                                 params_path, input_tensors)
    model_info["json"] = json_name
    if has_params:
      model_info["params"] = params_name
    input_names = get_input_names_from_file(json_path)
    output_names = get_output_names_from_file(json_path)
    return model_info, input_names, output_names

  def infer_output_tensors(self, save_folder, model_info, input_names, \
                          output_names, tensors):
    """ Get output shapes of the model.

    Args:
      save_folder: Folder path to save json files.
      model_info: A dict contains model information.
                  Format: {"json": json_name, "params": params_name}
      input_names: list, input tensor names.
      ouput_names: list, output tensor names.
      tensor_tensors: A dict contains tensor names and values.

    Returns:
      A list of numpy.ndarray, contains the output tensors.
    """
    if "params" in model_info:
      model = load_mxnet_model(device='cpu', folder=save_folder, \
              json_file=model_info["json"], params=model_info['params'])
    else:
      model = load_mxnet_model(device='cpu', folder=save_folder, \
              json_file=model_info["json"])
    input_tensors = [(name, tensors[name]) for name in input_names]
    required_outputs = [(name, None) for name in output_names]
    outputs = infer_mxnet(model, input_tensors, required_outputs, device='cpu')
    ret = [outputs[name] for name in output_names]
    return ret

  def get_tensor_dtype(self, tensor_name):
    return 0
