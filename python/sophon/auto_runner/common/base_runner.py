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
import os
import json
import collections
#import copy
import numpy as np
import sophon.sail as sail

class Runner(object):
  """ Virtual Class, infer splitted submodels on both cpu and tpu.

  Only one parameter is need:
    folder, path contains splitted submodels, with tpu models compiled.

  Attributes:
    mode: 0,1,2.
      0, all on cpu.
      1, cpu + gpu
      2, cpu + tpu
    folder: absolute path contains submodels to be infered.
    graph_infos_path: equals to "folder + 'graph_infos.json'".
    graph_infos: A dict loaded from the json file of graph_infos_path.
    platform: 'tensorflow' or 'mxnet' or 'pytorch' or 'caffe'.
    dynamic: if dynamic mode.
    layout: 'NCHW'(default) or 'NHWC'.
    graph_num: model numbers in this folder.
    required_input_names: A list, contains input tensor names \
                            of the original(unsplitted) model.
    output_names: A list, contains output tensor names of the original model.
    models: A list, contains reference of loaded submodel, in topological sort.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, folder, mode=2, graphs_in_memory=None, use_cmodel=0):
    """ Constructor.
    Args:
      folder: folders hold all subgraphs.
      mode: 0,1,2.
        0, all on cpu.
        1, cpu + gpu
        2, cpu + tpu
    """
    self.use_cmodel = use_cmodel
    self.mode = mode
    self.folder = os.path.abspath(folder)
    graph_infos_path = os.path.join(self.folder, 'graph_infos.json')
    if not os.path.isfile(graph_infos_path):
      raise RuntimeError("{} not found.".format(graph_infos_path))
    with open(graph_infos_path, 'r') as cfg:
      self.graph_infos = json.load(cfg)
      self.platform = self.graph_infos["platform"]
      self.dynamic = self.graph_infos['dynamic']
      self.layout = self.graph_infos["layout"]
      self.graph_num = self.graph_infos["graph_num"]
      self.required_input_names = [n for n in self.graph_infos["tensors"] \
                    if self.graph_infos["tensors"][n]["attr"] == "input"]
      self.output_names = [n for n in self.graph_infos["tensors"] \
                    if self.graph_infos["tensors"][n]["attr"] == "output"]
    if self.graph_infos['graph_num'] != len(self.graph_infos['graphs']):
      raise RuntimeError("Num of graphs doesn't equal to graph_num in {}".\
        format(graph_infos_path))
    if graphs_in_memory:
      self.models = self.load_models_from_memory(graphs_in_memory)
    else:
      self.models = self.load_models()

  def load_models_from_memory(self, graphs_in_memory):
    """ Load all submodels which already read in memory.
    """
    #if self.use_xx_cmodel:
    #  raise RuntimeError("Can't load from memory while using cmodel.")
    models = list()
    for i in range(self.graph_num):
      graph = self.graph_infos["graphs"][i]
      if graph["device"] == "cpu":
        graph_name = "graph_{}".format(i)
        assert graph_name in graphs_in_memory
        model = self.load_graph_cpu_from_memory(graphs_in_memory[graph_name])
      elif graph["device"] == "tpu":
        graph_name = "graph_{}_bmodel".format(i)
        assert graph_name in graphs_in_memory
        model = sail.Engine(graphs_in_memory[graph_name], \
            len(graphs_in_memory[graph_name]), 0, sail.IOMode.SYSIO)
      else:
        raise RuntimeError('wrong device: {0}!!!'.format(graph['device']))
      models.append(model)
    return models

  def load_models(self):
    """ Load all submodels.
    """
    models = list()
    for i in range(self.graph_num):
      graph = self.graph_infos["graphs"][i]
      if graph["device"] == "cpu":
        model = self.load_graph_cpu(graph['model_info'])
      elif graph["device"] == "tpu":
        if self.mode == 0:
          model = self.load_graph_cpu(graph['model_info'])
        elif self.mode == 1:
          model = self.load_graph_gpu(graph['model_info'])
        else:
          context_dir = str(os.path.join(self.folder, graph["context_dir"]))
          bmodel_path = os.path.join(context_dir, 'compilation.bmodel')
          #if self.use_xx_cmodel:
          #  model = bmodel_path
          #else:
          model = sail.Engine(bmodel_path, 0, sail.IOMode.SYSIO)
      else:
        raise RuntimeError('wrong device: {0}!!!'.format(graph['device']))
      models.append(model)
    return models

  def infer(self, input_tensors):
    """ Inference by running submodels as a pipeline.

    Args:
      input_tensors: A dict of input tensors. Format: {name: data}

    Returns:
      A dict of output tensors. Format: {name: data}
    """
    input_names = list(input_tensors.keys())
    if len(set(input_names) & set(self.required_input_names)) \
           != len(self.required_input_names):
      raise RuntimeError("Wrong graph_infos, inputs dont't match.")
    for name in input_names:
      if not isinstance(input_tensors[name], np.ndarray):
      #if type(input_tensors[name]) != type(np.empty(1)):
        raise ValueError("Value of {}".format(name))
    tensors = dict()
    for name in self.required_input_names:
      tensors[name] = input_tensors[name]
    for i in range(self.graph_num):
      graph = self.graph_infos["graphs"][i]
      inputs = [(name, tensors[name]) for name in graph['inputs']]
      required_outputs = [(n, tuple(self.graph_infos['tensors'][n]["shape"])) \
                          for n in graph['outputs']]
      if graph["device"] == "cpu":
        outputs = self.infer_on_cpu(i, inputs, required_outputs)
      elif graph["device"] == "tpu":
        if self.mode == 0:
          outputs = self.infer_on_cpu(i, inputs, required_outputs)
        elif self.mode == 1:
          outputs = self.infer_on_gpu(i, inputs, required_outputs)
        else:
          outputs = self.infer_on_tpu(i, inputs, required_outputs)
      tensors.update(outputs)
    result = dict()
    for name in self.output_names:
      result[name] = tensors[name]
    return result

  def infer_time(self, input_tensors):
    """ Inference by running submodels as a pipeline.

    Args:
      input_tensors: A dict of input tensors. Format: {name: data}
      pure_cpu: A bool, if true, all submodels are running on cpu.

    Returns:
      A list of time(ms).
    """
    import time
    t_s = time.time()
    t_result = []
    input_names = list(input_tensors.keys())
    if len(set(input_names) & set(self.required_input_names)) \
           != len(self.required_input_names):
      raise RuntimeError("Wrong graph_infos, inputs dont't match.")
    for name in input_names:
      if not isinstance(input_tensors[name], np.ndarray):
      #if type(input_tensors[name]) != type(np.empty(1)):
        raise ValueError("Value of {}".format(name))
    tensors = dict()
    for name in self.required_input_names:
      tensors[name] = input_tensors[name]
    for i in range(self.graph_num):
      time_0 = time.time()
      graph = self.graph_infos["graphs"][i]
      inputs = [(name, tensors[name]) for name in graph['inputs']]
      required_outputs = [(n, tuple(self.graph_infos['tensors'][n]["shape"])) \
                          for n in graph['outputs']]
      if graph["device"] == "cpu":
        outputs = self.infer_on_cpu(i, inputs, required_outputs)
      elif graph["device"] == "tpu":
        if self.mode == 0:
          outputs = self.infer_on_cpu(i, inputs, required_outputs)
        elif self.mode == 1:
          outputs = self.infer_on_gpu(i, inputs, required_outputs)
        else:
          outputs = self.infer_on_tpu(i, inputs, required_outputs)
      tensors.update(outputs)
      time_1 = time.time()
      t_result.append(time_1 - time_0)
    result = dict()
    for name in self.output_names:
      result[name] = tensors[name]
    t_e = time.time()
    t_result.append(t_e - t_s)
    return t_result

  def infer_on_tpu(self, index, inputs, required_outputs):
    """ Run a submodel on TPU.

    Args:
      index: Index of the submodel.
      inputs: A dict of input tensors. Format: {name: data}
      required_outputs: A list of output names and shapes of original model.
                        Format: [(name, shape), ]

    Returns:
      A dict of output tensors. Format: {name: data}
    """
    inputs = collections.OrderedDict(inputs)
    required_outputs = collections.OrderedDict(required_outputs)
    #if self.use_xx_cmodel:
    #  engine = sail.Engine(self.models[index], 0, sail.IOMode.SYSIO)
    #else:
    engine = self.models[index]
    graph_name = engine.get_graph_names()[0]
    input_names = engine.get_input_names(graph_name)
    output_names = engine.get_output_names(graph_name)
    inputs = self.preprocess_tpu_input_tensors(inputs, input_names)
    outputs = engine.process(graph_name, inputs)
    result = self.postprocess_tpu_output_tensors(outputs, required_outputs, output_names)
    #if self.use_xx_cmodel:
    #  del engine
    return result

  @abc.abstractmethod
  def load_graph_cpu_from_memory(self, graph_bytes):
    """ Load a submodel on cpu.

    Args:
      graph_bytes: A byte array, contains contents of this model.

    Returns:
      A model of a typical framework.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def load_graph_cpu(self, model_info):
    """ Load a submodel on cpu.

    Args:
      model_info: A dict, contains filenames of this model.

    Returns:
      A model of a typical framework.
    """
    pass

  @abc.abstractmethod
  def load_graph_gpu(self, model_info):
    """ Load a submodel on gpu.

    Args:
      model_info: A dict, contains filenames of this model.

    Returns:
      A model of a typical framework.
    """
    pass

  @abc.abstractmethod
  def infer_on_cpu(self, index, inputs, required_outputs):
    """ Run a submodel on cpu.

    Args:
      index: number of model in self.modes
      inputs: A dict, input tensors of this model
              Format: {input_name: value}
      required_outputs: A dict, output tensor shapes of this model
                        Format: {output_name: shape}
    """
    pass

  @abc.abstractmethod
  def infer_on_gpu(self, index, inputs, required_outputs):
    """ Run a submodel on gpu.

    Args:
      index: number of model in self.modes
      inputs: A dict, input tensors of this model
              Format: {input_name: value}
      required_outputs: A dict, output tensor shapes of this model
                        Format: {output_name: shape}
    """
    pass

  @abc.abstractmethod
  def preprocess_tpu_input_tensors(self, inputs, input_names):
    """ Preprocess input tensors when running model on TPU.

    When infer on tpu, input tensors need to adjust,gg
    i.e.
    if this IR is compiled from a tensorflow graph,
    if an input tensor shape is 4 dims and the layout is NHWC,
    it should be transposed to NCHW.
    Also, input tensor name in tensorflow need to be changed.

    Args:
      inputs: input tensors of original model.
              Format: {input_name: input_tensor}.
      input_names: input name list of IR(compiled by original model).

    Returns:
      new input tensors, same format as inputs,
      but their keys are input_names,
      values(tensors) are transformed by values in inputs.
    """
    pass

  @abc.abstractmethod
  def postprocess_tpu_output_tensors(self, outputs, required_outputs, output_names):
    """ postprocess output tensors when running model on TPU.

    If original output tensor shape is less than 4 dims,
    it should be reshape, because output tensor shape of tpu will be 4 dims.
    Also, in mxnet, output tensor names need to be changed.

    Mainly same as preprocess above,
    but required_outputs are a dict which contains both names and shapes of
    outputs of original model.

    Args:
      outputs: output tensors of IR.
          Format: {output_name: output_tensor}.
      required_outputs: output name list of original model.

    Returns:
      new output tensors, same format as outputs,
      but their keys are keys of required_outputs,
      values(tensors) are transformed by values in outputs.

    """
    pass
