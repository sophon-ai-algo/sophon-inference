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
import re
import json
import mxnet as mx
from mxnet import gluon
import numpy as np


def load_json_file(json_path):
  """ Get json object of the graph from a json file.

  Args:
    json_path: File path of model_name-symbol.json

  Returns:
    data: Json object of the graph.
  """
  with open(json_path, 'r') as json_file:
    data = json.load(json_file)
  return data

def get_index_dict(nodes):
  """ Get a dict denotes relationships between name and index of nodes.

  Args:
    nodes: A json object contain all the nodes in a mxnet json file.

  Returns:
    A dict with format: {node_name: node_index}
  """
  index_dict = dict()
  for index, node in enumerate(nodes):
    index_dict[node["name"]] = index
  return index_dict

def get_input_names_from_json(json_data):
  """ Judge if the node is weight.

  Args:
    json_data: Json object load from mxnet json file.

  Returns:
    A list of input tensor names.
  """
  input_tensor_names = list()
  name_filter = "_(weight|bias|var|mean|gamma|beta|label)"
  for node in json_data["nodes"]:
    if node["op"] == "null" and ("attrs" not in node.keys() and \
        not re.search(name_filter, node["name"])):
      input_tensor_names.append(node["name"])
  return input_tensor_names

def get_output_names_from_json(sym_json):
  """ Judge if the node is weight.

  Args:
    sym_json: Json object load from mxnet json file.

  Returns:
    A list of output tensor names.
  """
  #heads = [lst[0] for lst in sym_json["heads"]]
  output_names = list()
  for index in sym_json["heads"]:
    i = index[0]
    if sym_json["nodes"][i]["op"] != "SliceChannel":
      output_names.append(sym_json["nodes"][i]["name"])
    else:
      output_names.append(sym_json["nodes"][i]["name"] + "_" + \
                          str(index[1]) + "_sophon_auto")
  return output_names

def node_is_weight(node):
  """ Judge if the node is weight.

  Args:
    node: A json node load from mxnet json file.

  Returns:
    True if the node is weight. False for input tensors and operators.
  """
  ret = False
  if node["op"] == "null" and ("attrs" in node.keys() or \
      re.search("_(weight|bias|var|mean|gamma|beta|label)", node["name"])):
    ret = True
  return ret

def get_all_ops(json_path):
  """ Get operators in the mxnet graph.

  Args:
    json_path: File path of model_name-symbol.json

  Returns:
    ops: Information of all operators, exluding weight nodes.
         Format: {op_name: (op_type, [parent_name])}.
    input_ops: list, names of all input tensors.
    output_ops: list, names of all output tensors.
  """
  data = load_json_file(json_path)
  # adjust the data format
  assert isinstance(data["nodes"], list)
  nodes = data["nodes"]
  output_ids = [lst[0] for lst in data["heads"]]

  # format: {op_name: (op_type, [parent_name])}
  ops = {}
  input_ops = []
  output_ops = []
  for index, node in enumerate(nodes):
    if node_is_weight(node):
      continue
    parent_ids = [parent[0] for parent in node["inputs"]]
    ops[node["name"]] = (node["op"], [nodes[i]["name"] for i in parent_ids \
                        if not node_is_weight(nodes[i])])
    if node["op"] == "null" and not node["inputs"]:
      input_ops.append(node["name"])
    elif index in output_ids:
      output_ops.append(node["name"])
  return ops, input_ops, output_ops

def get_input_names_from_file(json_path):
  """ Get all output tensor names from a symbol.json.

  Args:
    json_path: Path of json file of the mxnet model.

  Returns:
    A list of output tensor names.
  """
  sym_json = load_json_file(json_path)
  input_names = get_input_names_from_json(sym_json)
  return input_names

def get_output_names_from_file(json_path):
  """ Get all output tensor names from a symbol.json.

  Args:
    json_path: Path of json file of the mxnet model.

  Returns:
    A list of output tensor names.
  """
  sym_json = load_json_file(json_path)
  output_names = get_output_names_from_json(sym_json)
  return output_names

def predict_with_params(json_path, params_path, input_tensors):
  """ Predict with provided model and input tensors.

  Args:
    json_path: Path of json file of the mxnet model.
    params_path: Path of params file of the mxnet model.
    input_tensors: A list contains all input tensor names and data.
                  Format: [(tensor_name, numpy.ndarray), ]

  Returns:
    output tensors. Format: {tensor_name: numpy.ndarray}
  """
  input_names = [item[0] for item in input_tensors]
  net = gluon.SymbolBlock.imports(json_path, input_names, \
                                  params_path, ctx=mx.cpu())
  inputs = [mx.nd.array(item[1]) for item in input_tensors]
  outputs = net(*inputs)

  result = dict()
  output_names = get_output_names_from_file(json_path)
  if len(output_names) == 1:
    result[output_names[0]] = outputs.asnumpy()
  else:
    for index, value in enumerate(output_names):
      result[value] = outputs[index].asnumpy()
    #for i in range(len(output_names)):
    #  result[output_names[i]] = outputs[i].asnumpy()
  return result

def infer_shape(json_path, input_shapes):
  """ Infer shapes of output tensors.

  Args:
    json_path: Path of json file of the mxnet model.
    input_shapes: A dict contains input names and shapes.

  Returns:
    A list of tuple, contains the output tensor shapes.
  """
  sym = mx.symbol.load(json_path)
  arg_shapes, out_shapes, aux_shapes = sym.infer_shape(**input_shapes)
  del arg_shapes
  del aux_shapes
  return out_shapes

def infer_shape_with_params(json_path, params_path, input_shapes):
  """ Infer shapes of all output tensors given all shapes of input tensors.

  Args:
    json_path: Path of json file of the mxnet model.
    params_path: Path of params file of the mxnet model.
    input_shapes: A list contains all input tensor names and shapes.
                  Format: [(tensor_name, tuple(tensor_shape)), ]

  Returns:
    Shapes of all output tensors. Format: {tensor_name: tuple(tensor_shape)}
  """
  #sym_json = load_json_file(json_path)
  input_tensors = list()
  for item in input_shapes:
    name = item[0]
    tensor = np.random.rand(*item[1])
    input_tensors.append((name, tensor))
  outputs = predict_with_params(json_path, params_path, input_tensors)
  result = dict()
  for key in outputs:
    result[key] = outputs[key].shape
  return result

def infer_shape_without_params(json_path, input_shapes):
  """ Infer shapes of all output tensors given all shapes of input tensors
      when no parameters in the model.

  Args:
    json_path: Path of json file of the mxnet model.
    input_shapes: A list contains all input tensor names and shapes.
                  Format: [(tensor_name, tuple(tensor_shape)), ]

  Returns:
    Shapes of all output tensors. Format: {tensor_name: tuple(tensor_shape)}
  """
  #sym_json = load_json_file(json_path)
  input_tensors = list()
  for item in input_shapes:
    name = item[0]
    tensor = np.random.rand(*item[1])
    input_tensors.append((name, tensor))
  outputs = forward_without_params(json_path, input_tensors)
  result = dict()
  for key in outputs:
    result[key] = outputs[key].shape
  return result

def forward_without_params(json_path, input_tensors):
  """ Forward with provided symbol.json and input tensors.

  Args:
    json_path: Path of json file of the mxnet model.
    input_tensors: A list contains all input tensor names and data.
                  Format: [(tensor_name, numpy.ndarray), ]

  Returns:
    output tensors. Format: {tensor_name: numpy.ndarray}
  """
  sym = mx.symbol.load(json_path)
  inputs = dict()
  for item in input_tensors:
    inputs[item[0]] = mx.nd.array(item[1])
  exe = sym.bind(mx.cpu(), inputs)
  outputs = exe.forward()

  result = dict()
  output_names = get_output_names_from_file(json_path)
  if len(output_names) == 1:
    result[output_names[0]] = outputs.asnumpy()
  else:
    for index, value in enumerate(output_names):
      result[value] = outputs[index].asnumpy()
  return result

def get_features(json_path, params_path, input_tensors, feature_names):
  """ Get features which are intermediate results of the model.

  Args:
    json_path: Path of json file of the mxnet model.
    params_path: Path of params file of the mxnet model.
    input_tensors: A list contains all input tensor names and data.
                  Format: [(tensor_name, numpy.ndarray), ]
    feature_names: A list contains names of featrues you want to get.

  Returns:
    feature tensors. Format: {tensor_name: numpy.ndarray}
  """
  sym = mx.sym.load(json_path)
  internals = sym.get_internals()
  outputs_ops = sym.get_internals().list_outputs()
  outputs = list()
  for name in feature_names:
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
  net.collect_params().load(params_path, ctx=mx.cpu(), ignore_extra=True)
  input_data = [mx.nd.array(item[1]) for item in input_tensors]
  features = net(*input_data)
  result = dict()
  if len(feature_names) == 1:
    result[feature_names[0]] = features.asnumpy()
  else:
    for index, value in enumerate(feature_names):
      result[value] = features[index].asnumpy()
  return result

def sym_has_params(sym, input_names):
  """ Judge if a model has parameters.

  Args:
    sym: A Symbol instance of the model.
    input_names: A list of input tensor names.

  Return:
    True for the model has parameters and Flase for not.
  """
  args = set(sym.list_arguments())
  params = args - set(input_names)
  ret = True
  if not params:
  #if len(params) == 0:
    ret = False
  return ret

def get_prefix_and_epoch(params_path):
  """ Get prefix and epoch from path to a mxnet parameter file.

  Args:
    params_path: Path to parameter file.

  Return:
    (prefix, epoch).
  """
  file_name = os.path.basename(params_path)
  tokens = re.split('-', file_name)
  prefix = tokens[0]
  post_tokens = re.split(r'\.', tokens[1])
  epoch = int(post_tokens[0])
  return (prefix, epoch)

def mxnet_infer(graph_infos, graph_id, tensor_values):
  """ Do inference of a mxnet model on cpu.

  Args:
    info: A dict contains paths of symbol file and parameter file.
    inputs: Input tensors. Format: {name: value}.

  Returns:
    Output tensors. Format: {name: value}.
  """
  info = graph_infos["graphs"][graph_id]["model_info"]
  inputs = [(name, tensor_values[name]) \
            for name in graph_infos["graphs"][graph_id]["inputs"]]
  if "params" in info.keys():
    outputs = predict_with_params(info["json"], info["params"], inputs)
  else:
    outputs = forward_without_params(info["json"], inputs)
  return outputs

def export_model_from_gluoncv(model_name, save_dir, data_shape=None):
  """ Export a pretrained model from gluoncv.

  Args:
    model_name: A pretrained model name.
    save_dir: Path of a directory to save the model.
    data_shape: data_shape of input tensor. Format: (height, width, channel).

  Returns:
    None
  """
  from gluoncv import model_zoo
  from gluoncv.utils import export_block
  net = model_zoo.get_model(model_name, pretrained=True)
  prefix = os.path.join(save_dir, model_name)
  export_block(prefix, net, data_shape=data_shape, \
               preprocess=False, layout='CHW')

def load_mxnet_model(device='cpu', folder=None, json_file=None, params=None):
  """ Load model.

  Args:
    device: "cpu" or "gpu".
    folder: Directory path contains model files.
    json_file: Json file name.
    params: Parameter file name.

  Returns:
    None.
  """
  json_path = os.path.join(folder, json_file)
  if params:
    params_path = os.path.join(folder, params)
    input_names = get_input_names_from_file(json_path)
    if device == 'cpu':
      model = gluon.SymbolBlock.imports(json_path, input_names, \
                                      params_path, ctx=mx.cpu())
    else:
      model = gluon.SymbolBlock.imports(json_path, input_names, \
                                      params_path, ctx=mx.gpu())
    return model
  else:
    return mx.symbol.load(json_path)

def infer_mxnet(model, inputs, required_outputs, device='cpu'):
  """ Run the mxnet model.

  Args:
    model: A loaded mxnet model. Either mx.symbol.symbol.Symbol or
           mx.gluon.block.SymbolBlock.
    inputs: A list contains all input tensor names and data.
            Format: [(tensor_name, numpy.ndarray), ]
    required_outputs: A list of output names and shapes of original model.
                      Format: [(name, shape), ]
    device: "cpu" or "gpu".

  Returns:
    A dict of output tensors. Format: {name: data}
  """
  model_type = type(model)
  if model_type == mx.symbol.symbol.Symbol:
    input_tensors = dict()
    for item in inputs:
      if device == 'cpu':
        input_tensors[item[0]] = mx.nd.array(item[1])
      else:
        input_tensors[item[0]] = mx.nd.array(item[1]).copyto(mx.gpu())
    exe = model.bind(mx.cpu(), input_tensors)
    outputs = exe.forward()
  elif model_type == mx.gluon.block.SymbolBlock:
    if device == 'cpu':
      input_tensors = [mx.nd.array(item[1]) for item in inputs]
    else:
      input_tensors = [mx.nd.array(item[1]).copyto(mx.gpu()) for item in inputs]
    outputs = model(*input_tensors)
  result = dict()
  output_names = [item[0] for item in required_outputs]
  if len(output_names) == 1:
    outputs = [outputs]
  #if len(output_names) == 1:
  #  if device == 'cpu':
  #    result[output_names[0]] = outputs.asnumpy()
  #  else:
  #    result[output_names[0]] = outputs.copyto(mx.cpu()).asnumpy()
  #else:
  for index, value in enumerate(output_names):
    if device == 'cpu':
      result[value] = outputs[index].asnumpy()
    else:
      result[value] = outputs[index].copyto(mx.cpu()).asnumpy()
  return result
