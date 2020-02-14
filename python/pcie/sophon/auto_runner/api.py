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

API for using auto_deploy.
"""

from __future__ import print_function
import os
import json
import logging


def exception_wrapper(message):
  """ Annotation function
  """
  def decorator(func):
    """ Decorator for tackling exception
    """
    def wrapper(*args, **kwargs):
      """ Wrapper for tackling exception
      """
      try:
        ret = func(*args, **kwargs)
      except:
        import traceback
        traceback.print_exc()
        logging.error(message)
        exit(-1)
      else:
        return ret
    return wrapper
  return decorator

@exception_wrapper(message="Met an Error when load model.")
def load(folder, graphs_in_memory=None, use_cmodel=0):
  """ load all the subgraphs in a runner.

  Args:
    folder: Directory path that contains splitted models and 'graph_infos.json'.

    Format of 'graph_infos.json':
    {
      "graph_num": graph_numbmer,
      "platform": "mxnet"
      "layout": "NCHW"
      "dynamic": False
      "graphs": [
        {
          "device": "cpu",
          "inputs": list of input tensor names,
          "outputs": list of output tensor names,
          "model_info": {
            "json": json_file_path,
            "params": params_file_path
          }
        }
      ]
      "tensors": [
        {
          "name": name,
          "shape": list for shape,
          "attr": "input" or "output" or "intermediate"
        }
      ]
    }
    input_tensors: Input tensors. Format: {name: value}.
    graphs_in_memory: A dict of graphs which already read in memory.

  Returns:
    An instance of Runner.
  """
  if not os.path.isfile(os.path.join(folder, 'graph_infos.json')):
    raise RuntimeError("graph_infos.json not under {}.".format(folder))
  with open(os.path.join(folder, 'graph_infos.json'), 'r') as cfg:
    graph_infos = json.load(cfg)
  platform = graph_infos["platform"]
  assert(platform in ["tensorflow", "mxnet", "caffe", "pytorch"])
  if platform == 'tensorflow':
    from .runner.tensorflow_runner import TensorflowRunner as RN
  elif platform == 'mxnet':
    from .runner.mxnet_runner import MxnetRunner as RN
  elif platform == 'caffe':
    from .runner.caffe_runner import CaffeRunner as RN
  elif platform == 'pytorch':
    from .runner.pytorch_runner import PytorchRunner as RN
  else:
    raise NotImplementedError("unsupport platform: {}".format\
            (graph_infos['platform']))
  runner = RN(folder, graphs_in_memory=graphs_in_memory, use_cmodel=use_cmodel)
  return runner

@exception_wrapper(message="Met an Error when infer loaded models.")
def infer(runner, input_tensors):
  """ Run loaded subgraphs.

  Args:
    runner: an instance of Runner.
    input_tensors: Input tensors. Format: {name: value}.

  Returns:
    Output tensors. Format: {name: value}.
  """
  return runner.infer(input_tensors)


# @exception_wrapper(message="Met an Error when infer model.")
# def infer(folder, input_tensors):
#   """ Run all the subgraphs as a pipeline.
# 
#   Args:
#     folder: Directory path that contains splitted models and 'graph_infos.json'.
# 
#     Format of 'graph_infos.json':
#     {
#       "graph_num": graph_numbmer,
#       "platform": "mxnet"
#       "layout": "NCHW"
#       "dynamic": False
#       "graphs": [
#         {
#           "device": "cpu",
#           "inputs": list of input tensor names,
#           "outputs": list of output tensor names,
#           "model_info": {
#             "json": json_file_path,
#             "params": params_file_path
#           }
#         }
#       ]
#       "tensors": [
#         {
#           "name": name,
#           "shape": list for shape,
#           "attr": "input" or "output" or "intermediate"
#         }
#       ]
#     }
#     input_tensors: Input tensors. Format: {name: value}.
# 
#   Returns:
#     Output tensors. Format: {name: value}.
#   """
#   if not os.path.isfile(os.path.join(folder, 'graph_infos.json')):
#     raise RuntimeError("graph_infos.json not under {}.".format(folder))
#   with open(os.path.join(folder, 'graph_infos.json'), 'r') as cfg:
#     graph_infos = json.load(cfg)
#   platform = graph_infos["platform"]
#   assert(platform in ["tensorflow", "mxnet", "caffe", "pytorch"])
#   if platform == 'tensorflow':
#     from .runner.tensorflow_runner import TensorflowRunner as RN
#   elif platform == 'mxnet':
#     from .runner.mxnet_runner import MxnetRunner as RN
#   #elif platform == 'caffe':
#   #  from auto_deploy.runner.caffe_runner import CaffeRunner as RN
#   #elif platform == 'pytorch':
#   #  from auto_deploy.runner.pytorch_runner import PytorchRunner as RN
#   else:
#     raise NotImplementedError("unsupport platform: {}".format\
#             (graph_infos['platform']))
#   runner = RN(folder)
#   return runner.infer(input_tensors)
# 
def infer_time(folder, input_tensors, save_file, mode):
  """ infer time of splitted models.
  """
  import numpy as np
  with open(os.path.join(folder, 'graph_infos.json'), 'r') as cfg:
    graph_infos = json.load(cfg)
  platform = graph_infos["platform"]
  assert(platform in ["tensorflow", "mxnet", "caffe", "pytorch"])
  if platform == 'tensorflow':
    from .runner.tensorflow_runner import TensorflowRunner as RN
  elif platform == 'mxnet':
    from .runner.mxnet_runner import MxnetRunner as RN
  #elif platform == 'caffe':
  #  from auto_deploy.runner.caffe_runner import CaffeRunner as RN
  #elif platform == 'pytorch':
  #  from auto_deploy.runner.pytorch_runner import PytorchRunner as RN
  else:
    raise NotImplementedError("unsupport platform: {}".format\
            (graph_infos['platform']))
  runner = RN(folder, mode)
  for _ in range(10):
    t_s = runner.infer_time(input_tensors)
  time_array = np.zeros(len(t_s))
  for _ in range(100):
    t_s = runner.infer_time(input_tensors)
    time_array += np.array(t_s)
  print("time: " + str(time_array))
  with open(save_file, 'w') as file_:
    file_.write(str(time_array))

@exception_wrapper(message="Met an Error when load models from zip.")
def load_from_zip(model_path, use_cmodel=0):
  """ load all the subgraphs in a runner.

  Args:
    folder: Directory path that contains splitted models and 'graph_infos.json'.

    Format of 'graph_infos.json':
    {
      "graph_num": graph_numbmer,
      "platform": "mxnet"
      "layout": "NCHW"
      "dynamic": False
      "graphs": [
        {
          "device": "cpu",
          "inputs": list of input tensor names,
          "outputs": list of output tensor names,
          "model_info": {
            "json": json_file_path,
            "params": params_file_path
          }
        }
      ]
      "tensors": [
        {
          "name": name,
          "shape": list for shape,
          "attr": "input" or "output" or "intermediate"
        }
      ]
    }
    input_tensors: Input tensors. Format: {name: value}.
    graphs_in_memory: A dict of graphs which already read in memory.

  Returns:
    An instance of Runner.
  """
  if (not os.path.isfile(model_path)) or (not model_path.endswith('.zip')):
    raise RuntimeError("invalid model path")
  from sophon.utils.functions import read_graph_infos_from_zip
  graph_infos = json.loads(read_graph_infos_from_zip(model_path))
  platform = graph_infos["platform"]
  assert(platform in ["tensorflow", "mxnet", "caffe", "pytorch"])
  if platform == 'tensorflow':
    from .runner.tensorflow_runner_zip import TensorflowRunnerZip as RNZ
  #elif platform == 'mxnet':
  #  from .runner.mxnet_runner import MxnetRunner as RN
  #elif platform == 'caffe':
  #  from auto_deploy.runner.caffe_runner import CaffeRunner as RN
  #elif platform == 'pytorch':
  #  from auto_deploy.runner.pytorch_runner import PytorchRunner as RN
  else:
    raise NotImplementedError("unsupport platform: {}".format\
            (graph_infos['platform']))
  runner = RNZ(model_path, use_cmodel=use_cmodel)
  return runner
