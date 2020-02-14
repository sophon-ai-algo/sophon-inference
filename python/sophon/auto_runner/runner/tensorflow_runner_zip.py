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
import zipfile
import json
from sophon.utils.functions import read_graph_infos_from_zip, read_binary_from_zip_TF
from .tensorflow_runner import TensorflowRunner


class TensorflowRunnerZip(TensorflowRunner):
  def __init__(self, model_path, mode=2, use_cmodel=0):
    """ Constructor.
    Args:
      model_path: zip file.
      mode: 0,1,2.
        0, all on cpu.
        1, cpu + gpu
        2, cpu + tpu
    """
    self.mode = mode
    self.use_cmodel = use_cmodel
    self.model_path = model_path
    self.graphs_in_memory = dict()
    self.graph_infos = json.loads(read_graph_infos_from_zip(self.model_path))
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
    
    for idx, graph in enumerate(self.graph_infos['graphs']):
      if graph['device'] == 'cpu':
        self.graphs_in_memory['graph_{}'.format(idx)] = read_binary_from_zip_TF(self.model_path, idx, 'cpu')
      else:
        assert graph['device'] == 'tpu'
        self.graphs_in_memory['graph_{}_bmodel'.format(idx)] = read_binary_from_zip_TF(self.model_path, idx, 'tpu')
    self.models = self.load_models_from_memory(self.graphs_in_memory)
