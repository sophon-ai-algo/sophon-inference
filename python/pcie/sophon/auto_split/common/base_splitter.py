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
import abc
import copy
import os
import json
from .graph import Graph, Operator

class Splitter(object):
  """ Virtual class, split a DL model into sub-models.

  A subtype of Splitter can split DL models of a typical framework,
  like tensorflow, mxnet, pytorch, caffe...
  Following is the splitting process:
    First, initialize a splitter for a model using subtype like
    "TensorflowSplitter", "MxnetSplitter"...
    Thus, a subtying need to implement the virtual method "initialize"
    to initialize at least these attributes:
      "self.graph_infos"
      "self.ops"
      "self.input_tensors"
      "self.output_names"
      "self.dynamic"
      "self.layout"
    Then, A Graph which defined in auto_deploy.common.graph will be generated.
    Be aware that the Graph is generated based on a list of Operator,
    which is also defined in auto_deploy.common.graph.
    To get these Operators, the subtype need to implement several
    virtual methods to ensure ops in the DL model to be correctlly parsed.
    They are:
      "get_op_name"
      "is_op_supported"
      "is_op_dangerous"
      "is_input_op"
      "is_output_op"
      "get_input_list"
    please read comments of upon functions to get detailed information.
    After a DL model is parsed to Graph, we exploit a general algorithm
    to get a list of subgraphs, which are in topological sort.
    Last, we save subgraphs into real models and write config file.
    Also, several virtual methods need to be implemented in subtype.
    They are:
      "save_subgraph"
      "infer_output_shapes"
      "get_model_info"
    please read comments of upon functions to get detailed information.

  Attributes:
    model_descriptor: Any type as long as it contains the information
                      of a DL model which we want to split.
    graph_infos: A dict contains information of graphs,
                 finnally will save to a json file,
                 which contains information of subgraphs.
    Format of graph_infos:
    {
      "graph_num": Integer, graph_numbmer,
      "dynamic": Boolean, if input shapes are dynamic,
      "graphs": [
        {
          "device": String, "cpu" or "tpu",
          "inputs": list of input tensor names,
          "outputs": list of output tensor names,
          "model_info": {
            path of model file and weight file(if has).
          }
        }
      ]
      "tensors": [
        {
          "name": String, tensor name,
          "shape": list for shape,
          "attr": "input" or "output" or "intermediate"
        }
      ]
    }
    ops: A list of ops in original DL model.
    input_tensors: A dict of input tensors' names and corresponding tensors.
    output_names: A list of output tensors' names.
    dynamic: A boolean represents if this graph is used in dynamic mode.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, model_descriptor):
    """ Constructor.
    """
    self.model_descriptor = model_descriptor
    self.layout = model_descriptor['layout']
    self.dynamic = model_descriptor['dynamic']
    self.graph_infos = dict()
    self.graph_infos['dynamic'] = self.dynamic
    self.graph_infos['layout'] = self.layout
    self.graph_infos['graphs'] = list()
    self.graph_infos['tensors'] = dict()
    self.platform = None
    self.ops = None
    self.input_tensors = None
    self.input_names = None
    self.output_names = None
    self.initialize()
    assert self.platform
    assert self.ops
    assert self.input_tensors
    assert self.input_names
    assert self.output_names
    self.graph_infos['platform'] = self.platform

  def convert_and_split(self, save_folder):
    """ Convert and split model in self into submodels.

    Args:
      save_folder: Path of folder to store splitted submodels and config files.

    Returns: None
    """
    operators = []
    for operator in self.ops:
      name = self.get_op_name(operator)
      is_support = self.is_op_support(operator)
      is_compute = self.is_op_compute(operator)
      is_dangerous = self.is_op_dangerous(operator)
      is_input = self.is_input_op(operator)
      is_output = self.is_output_op(operator)
      input_ops = self.get_inputs_list(operator)
      operators.append(Operator(name, is_support, is_compute, is_dangerous, \
                                is_input, is_output, input_ops))
    bm_graph = Graph()
    bm_graph.parse_from_operators(operators)
    sub_graphs = bm_graph.get_final_subgraphs()
    self.graph_infos['graph_num'] = len(sub_graphs)

    tensors = copy.deepcopy(self.input_tensors)
    for i, graph in enumerate(sub_graphs):
      print("subgraph: {}".format(i))
      model_info, sub_inputs, sub_outputs = self.save_subgraph(graph, \
                                                   save_folder, i, tensors)
      print("model_info: ", model_info)
      print("sub_inputs: ", sub_inputs)
      print("sub_outputs: ", sub_outputs)
      sub_output_tensors = self.infer_output_tensors(save_folder, model_info, \
                                                   sub_inputs, sub_outputs, \
                                                   tensors)
      tensors.update(dict(zip(sub_outputs, sub_output_tensors)))
      sub_info = dict()
      sub_info['inputs'] = sub_inputs
      sub_info['outputs'] = sub_outputs
      if graph.support == 1:
        sub_info['device'] = 'tpu'
        sub_info['context_dir'] = 'graph_ir_{0}'.format(i)
      else:
        sub_info['device'] = 'cpu'
      sub_info['model_info'] = model_info
      self.graph_infos['graphs'].append(sub_info)
    for name in list(tensors.keys()):
      tensor_node = {"shape": tensors[name].shape}
      tensor_node["dtype"] = self.get_tensor_dtype(name)
      if name in self.input_names:
        tensor_node["attr"] = "input"
      elif name in self.output_names:
        tensor_node["attr"] = "output"
      else:
        tensor_node["attr"] = "intermediate"
      self.graph_infos["tensors"][name] = tensor_node
    self.save_graph_infos(save_folder)
    self.destroy()

  @abc.abstractmethod
  def initialize(self):
    """ Initialize attributes based on self.model_descriptor.

        Attributes to be initialized are:
        graph_infos, ops, input_shapes, output_names, dynamic.
        graph_infos: A dict that is to be saved in json file
                      as the config file of splitted models.
        ops: A list contains all ops in the original model.
        input_tensors: A dict, {input_name: numpy.ndarray}
        output_names: A list contains output tensor names.
        dynamic: if dynamic mode.
        layout: defaul "NCHW".

      Returns: None.
    """
    pass

  @abc.abstractmethod
  def get_op_name(self, operator):
    """ Get the name of an operator

    Args:
      operator: an element of self.ops.

    Returns:
      A String, name of this operator.
    """
    pass

  @abc.abstractmethod
  def is_op_support(self, operator):
    """ Judge if this operator supported on sophon

    Args:
      operator: an element of self.ops.

    Returns:
      A boolean.
    """
    pass

  @abc.abstractmethod
  def is_op_compute(self, operator):
    """ Judge if this operator is a computing operator.
        Computing op usually has large amounts of flops.

    Args:
      operator: an element of self.ops.

    Returns:
      A boolean.
    """
    pass

  @abc.abstractmethod
  def is_op_dangerous(self, operator):
    """ Judge if this operator is a dangerous operator.
        Dangerous operators can't be outputs of a graph.

    Args:
      operator: an element of self.ops.

    Returns:
      A boolean.
    """
    pass

  @abc.abstractmethod
  def is_input_op(self, operator):
    """ Judge if this operator is a input(of the original model) operator.

    Args:
      operator: an element of self.ops.

    Returns:
      A boolean.
    """
    pass

  @abc.abstractmethod
  def is_output_op(self, operator):
    """ Judge if this operator is a output(of the original model) operator.

    Args:
      operator: an element of self.ops.

    Returns:
      A boolean.
    """
    pass

  @abc.abstractmethod
  def get_inputs_list(self, operator):
    """ Get the names of input operators of an operator.

    Args:
      operator: an element of self.ops.

    Returns:
      A list of string.
    """
    pass

  @abc.abstractmethod
  def save_subgraph(self, graph, save_folder, index, tensors):
    """ Save a subgraph into model files.

    Args:
      graph: A Subgraph defined in auto_deploy.common.graph,
             represents a splitted subgraph.
      save_folder: A string, Path to save splitted models.
      index: An integer, the number of this subgraph.
      tensors: A dict contains input tensor shapes of this subgraph,
                     Format: {tensor_name: numpy.ndarray}

    Returns:
      model_info: A dict, contains filenames after saving.
      inputs: A list, contains input tensor names of the saved model.
      outputs: A list, contains output tensor names of the saved model.

    """
    pass

  @abc.abstractmethod
  def infer_output_tensors(self, save_folder, model_info, \
                          sub_inputs, sub_outputs, tensors):
    """ Infer output tensors of a saved model

    Args:
      save_folder: A string, path that contains this splitted submodel.
      model_info: A dict, contains filenames of this submodel.
      sub_inputs: A list, contains input tensor names of this submodel.
      sub_outputs: A list, contains output tensor names of this submodel.
      tensors: A dict, contains tensors. Format:
                     {tensor_name: numpy.ndarray}. input name in sub_inputs
                     should be in keys of tensor_values.

    Returns: A list of tuple, contains the output tensors.
             This list is one-to-one correspondence with sub_outputs.
    """
    pass
    
  @abc.abstractmethod
  def get_tensor_dtype(self, tensor_name):
    """ Infer output tensors of a saved model

    Args:
      tensor_name: string, name of query tensor.

    Returns: A type from specific framework.
    """
    pass

  def save_graph_infos(self, save_folder):
    """ Save the information of splitted subgraphs to a json file.

    Args:
      save_folder: Directory path to save the json file.

    Returns:
      None.
    """
    formatted = json.dumps(self.graph_infos, indent=2, sort_keys=False)
    with open(os.path.join(save_folder, 'graph_infos.json'), 'w') as file_:
      file_.write(formatted)

  @abc.abstractmethod
  def destroy(self):
    """ Destroy resource.
    """
    pass
