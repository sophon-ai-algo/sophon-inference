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

import abc
import os
import json

class Compiler(object):
  """ Virtual Class, compile spliited supported submodels into IRs.

  A subtype of Compiler can compile models of a typical framework,
  like tensorflow, mxnet, pytorch, caffe...
  4 parameters are needed when create a Compiler,
  They are:
    folder (required): path that contains splitted models and infos.
    optimize (optional): optimizing mode when compiling to bm models.
    compare (optional): if comparing results with cpu when compiling.
    target (optional): current only 'BM1682', future will support 'BM1684'.

  each subtype need implement the virtual function:
    compile_model_using_bmcompiler
  And check_init, generate_compiling_script are optional.

  Attributes:
    folder: A string, path contains submodels to be compiled by this Compiler.
    optimize: An integer, optimizing mode of bm compiler.
    compare: A boolean, if comparing when compiling.
    target: A string, current 'BM1682' only, future contains 'BM1684'
    graph_infos_path: A string, equals to "${folder}/graph_infos.json",
                      A json file contains information of subgraphs.
    graph_infos: A dict, parsed from graph_infos_path.
    platform: A string, 'tensorflow' or 'mxnet' or 'pytorch' or 'caffe'.
    dynamic: A boolean, if dynamic mode.
    layout: A string, 'NCHW' or NHWC.
    tensors: A dict, contains all input/output tensors in all subgraphs.
    Format of tensors: {
      name: {
        attr: "input" (or "intermediate" or "output")
        shape: A tuple.
      }
    }
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, folder, optimize=None, compare=False, target='BM1682'):
    """ Constructor.
    """
    self.folder = os.path.abspath(folder)
    self.optimize = optimize
    self.compare = compare
    self.target = target
    self.graph_infos_path = os.path.join(self.folder, 'graph_infos.json')
    if not os.path.isfile(self.graph_infos_path):
      raise RuntimeError("{} not found.".format(self.graph_infos_path))
    with open(self.graph_infos_path, 'r') as cfg:
      self.graph_infos = json.load(cfg)
      self.platform = self.graph_infos['platform']
      self.dynamic = self.graph_infos['dynamic']
      self.layout = self.graph_infos['layout']
      self.tensors = self.graph_infos['tensors']
    if self.graph_infos['graph_num'] != len(self.graph_infos['graphs']):
      raise RuntimeError("Num of graphs doesn't equal to graph_num in {}.".\
        format(self.graph_infos_path))
    self.check_init()

  def compile_to_bm_models(self):
    """ Compile submodels which are deployed on TPU to bmodels.
    """
    for i in range(self.graph_infos['graph_num']):
      graph = self.graph_infos['graphs'][i]
      if graph['device'] == 'cpu':
        continue
      compile_info = dict()
      compile_info['context_dir'] = graph['context_dir']
      compile_info['model_info'] = graph['model_info']
      compile_info['input_names'] = graph['inputs']
      compile_info['input_shapes'] = [tuple(self.tensors[name]['shape']) \
                                      for name in graph['inputs']]
      compile_info['output_names'] = graph['outputs']
      self.generate_compiling_script(compile_info)
      self.compile_model_using_bmcompiler(compile_info)

  @abc.abstractmethod
  def check_init(self):
    """ Check initialized attributes.
    """
    pass

  @abc.abstractmethod
  def generate_compiling_script(self, compile_info):
    """ Generate scripts of compiling.

    Args:
      compile_info: Information for compilation.

   Returns:
      None.
    """
    pass

  @abc.abstractmethod
  def compile_model_using_bmcompiler(self, compile_info):
    """ Compile a submodel into bmodel.

    Args:
      compile_info: A dict, contains information needed when compiling.

    Returns: None
    """
    pass
