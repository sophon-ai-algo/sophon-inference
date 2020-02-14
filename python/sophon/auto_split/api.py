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

@exception_wrapper(message="Met an Error when split model.")
def split(platform, input_tensors, save_dir, graph_path, \
          params_path=None, outputs=None, dynamic=False, layout='NCHW'):
  """ Split the raw model into several submodels.

  Args:
    platform: Platform that trained the model.
              Options: tensorflow, mxnet, pytorch, caffe
    input_tensors: A dict contains the information of input tensors.
                  Format: {input_name: numpy.ndarray}
    save_dir: Path of directory to save submodels and splitting information.
    graph_path: Path to the graph description file of the model.
    params_path: Path to the parameters file of the model. Default None.
    outputs: A list contains the output tensor names. Default None.
    dynamic: True means input tensor shapes may change. Default False.
    layout: Layout of tensor. Default 'NCHW'.

  Returns:
    None.
  """
  save_dir = os.path.abspath(save_dir)
  graph_path = os.path.abspath(graph_path)
  if params_path is not None:
    params_path = os.path.abspath(params_path)
  supported_platforms = ["tensorflow", "mxnet", "pytorch", "caffe"]
  if platform not in supported_platforms:
    raise NotImplementedError("{} not supported by now.".format(platform))
  if platform == "mxnet":
    if params_path is None:
      raise RuntimeError("params_path should not be none for mxnet.")
    if not graph_path.endswith('.json') or not params_path.endswith('.params'):
      raise RuntimeError("models for mxnet should be like" + \
            "xxx.json and xxx.params.")
    from .splitter.mxnet_splitter import MxnetSplitter as SPL
    model_des = {"json_path": graph_path, "params_path": params_path,
                 "dynamic": dynamic, "input_tensors": input_tensors,
                 "layout": layout}
  elif platform == "tensorflow":
    if params_path is not None:
      raise RuntimeError("params_path should be None for tensorflow.")
    if outputs is None:
      raise RuntimeError("outputs should not be none for tensorflow.")
    if layout != 'NHWC':
      raise NotImplementedError("TF only support NHWC by now.")
    from .splitter.tensorflow_splitter import TensorflowSplitter as SPL
    model_des = {"model_path": graph_path, "dynamic": dynamic,
                 "input_tensors": input_tensors, "output_names": outputs,
                 "layout": layout}
  elif platform == "caffe":
    if params_path is None:
      raise RuntimeError("params_path should not be None for caffe.")
    if outputs is None:
      raise RuntimeError("outputs should not be none for caffe.")
    if layout != 'NCHW':
      raise NotImplementedError("Caffe only support NCHW by now.")
    from .splitter.caffe_splitter import CaffeSplitter as SPL
    model_des = {"proto_path": graph_path, "weight_path": params_path,
                 "dynamic": dynamic,
                 "input_shapes": input_tensors, "output_shapes": outputs,
                 "layout": layout}
  elif platform == "pytorch":
    if params_path is not None:
      raise RuntimeError("params_path should be None for pytorch.")
    if outputs is None:
      raise RuntimeError("outputs should not be none for pytorch.")
    if layout != 'NCHW':
      raise NotImplementedError("Pytorch only support NCHW by now.")
    from .splitter.pytorch_splitter import PytorchSplitter as SPL
    model_des = {"model_path": graph_path, "dynamic": dynamic,
                 "input_shapes": input_tensors, "output_shapes": outputs,
                 "layout": layout}
  else:
    raise NotImplementedError(platform)
  splitter = SPL(model_des)
  splitter.convert_and_split(save_dir)

@exception_wrapper(message="Met an Error when compile model.")
def convert(folder, optimize=None, compare=False, target='BM1682'):
  """ Compile all the subgraphs which can deploy on sophon.

  Args:
    folder: path that contains splitted models,
            'graph_infos.json' must be under this folder

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
    optimize: optimizing mode, parameter of bmnet compiler.
    compare: if compare with cpu results when compiling.
    target: 'BM1682' or 'BM1684'(future).

  Returns: None.
  """
  if not os.path.isfile(os.path.join(folder, 'graph_infos.json')):
    raise RuntimeError("graph_infos.json not under {}.".format(folder))
  with open(os.path.join(folder, 'graph_infos.json'), 'r') as cfg:
    graph_infos = json.load(cfg)
  if graph_infos['platform'] == 'mxnet':
    from .compiler.mxnet_compiler import MxnetCompiler as CP
  elif graph_infos['platform'] == 'tensorflow':
    from .compiler.tensorflow_compiler import TensorflowCompiler as CP
  elif graph_infos['platform'] == 'pytorch':
    from .compiler.pytorch_compiler import PytorchCompiler as CP
  elif graph_infos['platform'] == 'caffe':
    from .compiler.caffe_compiler import CaffeCompiler as CP
  else:
    raise NotImplementedError("unsupport platform: {}".format\
            (graph_infos['platform']))
  compiler = CP(folder, optimize=optimize, compare=compare, target=target)
  compiler.compile_to_bm_models()

