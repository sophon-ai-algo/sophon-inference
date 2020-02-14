"""
bmodel converter
"""
from __future__ import print_function

import os
import shutil
from abc import ABCMeta, abstractmethod
import six
import yaml
import numpy as np
from sophon.auto_split.api import split as splitter
from sophon.auto_split.api import convert as compiler

@six.add_metaclass(ABCMeta)
class BaseConverter(object):
  """Abstract converter

  Attributes:
    framework: deeplearning framework
  """
  def __init__(self, framework, base_path):
    """Inits Abstract converter"""
    self.framework = framework
    self.base_path = base_path

  @abstractmethod
  def converter(self):
    """Abstractmethod implemented by derived classes"""
    raise NotImplementedError

  def compile_bmodel(self, model=None, weight=None, target="BM1682",
                     shapes=None, net_name=None, input_names=None,
                     output_names=None, dyn=None, outdir=None):
    """bmcompiler compile input model"""
    if self.framework == "caffe":
      import bmnetc as bmcompiler
      bmcompiler.compile(
          model=model,
          weight=weight,
          outdir=outdir,
          target=target,
          shapes=shapes,
          dyn=dyn)
    elif self.framework == "tensorflow":
      import bmnett as bmcompiler
      bmcompiler.compile(
          model=model,
          outdir=outdir,
          target=target,
          shapes=shapes,
          net_name=net_name,
          input_names=input_names,
          output_names=output_names,
          dyn=dyn)
    elif self.framework == "mxnet":
      import bmnetm as bmcompiler
      bmcompiler.compile(
          model=model,
          weight=weight,
          outdir=outdir,
          target=target,
          shapes=shapes,
          net_name=net_name,
          input_names=input_names,
          dyn=dyn)
    elif self.framework == "pytorch":
      import bmnetp as bmcompiler
      bmcompiler.compile(
          model=model,
          outdir=outdir,
          target=target,
          shapes=shapes,
          net_name=net_name,
          dyn=dyn)
    else:
      raise ValueError('not a valid framework: ', self.framework)

class SplitGraphConverter(BaseConverter):
  """graph which need to be splitted with auto_deploy bmodel convetrter
  source_path, subgraph_path, tfmodel_path,\
         framework, input_names, output_names, layout, is_dynamic, input_shapes
  """
  def __init__(self, model_name, base_path, **graph_info):
    """Init graph bmodel converter"""
    super(SplitGraphConverter, self).__init__(graph_info["framework"],
                                              base_path)
    print("{} bmodel converter init".format(model_name))
    self.converter_config = graph_info
    self.tensors_dict = {}
    if len(self.converter_config['input_names']) == \
       len(self.converter_config['input_shapes']):
      for input_name, input_shape in zip(self.converter_config['input_names'],
                                         self.converter_config['input_shapes']):
        self.tensors_dict[input_name] = np.ndarray((input_shape),
                                                   dtype=np.float32)

  def converter(self):
    if self.framework == "tensorflow":
      # autodeploy split
      splitter(self.converter_config['framework'],
               self.tensors_dict,
               self.converter_config['subgraph_path'][0],
               self.converter_config['tfmodel_path'][0],
               params_path=None,
               outputs=self.converter_config['output_names'],
               dynamic=self.converter_config['is_dynamic'],
               layout=self.converter_config['layout'])
      print("split done!")
      # compile
      compiler(self.converter_config['subgraph_path'][0],
               optimize=1,
               compare=True,
               target=self.converter_config['target'])
      print("compile done!")
    elif self.framework == "mxnet":
      # autodeploy split
      splitter(self.converter_config['framework'],
               self.tensors_dict,
               self.converter_config['subgraph_path'][0],
               self.converter_config['json_path'][0],
               params_path=self.converter_config['params_path'][0])
      print("split done!")
      compiler(self.converter_config['subgraph_path'][0],
               target=self.converter_config['target'])
      print("compile done!")
    else:
      raise ValueError("not a valid framework: {}".format(self.framework))

class PytorchGraphConverter(BaseConverter):
  """pytorch graph bmodel converter
  """
  def __init__(self, model_name, base_path, models_path, shapes, dyns, outdirs,
               nets_name, framework, target):
    """Init pytorch graph bmodel converter"""
    super(PytorchGraphConverter, self).__init__(framework, base_path)
    print("{} bmodel converter init".format(model_name))
    self.model_name = model_name
    self.models_path = models_path
    self.shapes = shapes
    self.dyns = dyns
    self.outdirs = outdirs
    self.nets_name = nets_name
    self.target = target
    assert len(self.models_path) == len(self.nets_name)
    assert len(self.models_path) == len(self.dyns)
    assert len(self.models_path) == len(self.outdirs)
    self.output_base_path = os.path.join(self.base_path,
                                         self.model_name + "_ir")
    if not os.path.exists(self.output_base_path):
      os.mkdir(self.output_base_path)

  def converter(self):
    """convert pytorch graph"""
    print("generate {} bmodel...".format(self.model_name))
    # process pytorch graph
    for i in range(len(self.models_path)):
      super(PytorchGraphConverter, self).compile_bmodel(
          model=self.models_path[i], shapes=self.shapes[i],
          dyn=self.dyns[i], net_name=self.nets_name[i],
          outdir=self.outdirs[i], target=self.target)
    os.system('rm -f ./bm_multi_engine_stas_0.dat')
    os.system('rm -f ./*.grp')
    print("generate bmodel {}".format(self.output_base_path))

class MXNetGraphConverter(BaseConverter):
  """mxnet graph bmodel converter
  """
  def __init__(self, model_name, base_path, models_path,
               weights_path, shapes, dyns, outdirs,
               nets_name, input_names, framework, target):
    """Init mxnet graph bmodel converter"""
    super(MXNetGraphConverter, self).__init__(framework, base_path)
    print("{} bmodel converter init".format(model_name))
    self.model_name = model_name
    self.models_path = models_path
    self.weights_path = weights_path
    self.shapes = shapes
    self.dyns = dyns
    self.outdirs = outdirs
    self.nets_name = nets_name
    self.input_names = input_names
    self.target = target
    assert len(self.models_path) == len(self.weights_path)
    assert len(self.models_path) == len(self.nets_name)
    assert len(self.models_path) == len(self.dyns)
    assert len(self.models_path) == len(self.outdirs)
    assert len(self.models_path) == len(self.input_names)
    self.output_base_path = os.path.join(self.base_path,
                                         self.model_name + "_ir")
    if not os.path.exists(self.output_base_path):
      os.mkdir(self.output_base_path)

  def converter(self):
    """convert mxnet graph"""
    print("generate {} bmodel...".format(self.model_name))
    # process mxnet graph
    for i in range(len(self.models_path)):
      super(MXNetGraphConverter, self).compile_bmodel(
          model=self.models_path[i], weight=self.weights_path[i],
          shapes=self.shapes[i], dyn=self.dyns[i],
          net_name=self.nets_name[i], input_names=self.input_names[i],
          outdir=self.outdirs[i], target=self.target)
    os.system('rm -f ./bm_multi_engine_stas_0.dat')
    print("generate bmodel {}".format(self.output_base_path))

class TfGraphConverter(BaseConverter):
  """tf graph bmodel converter
  """
  def __init__(self, model_name, base_path, models_path,
               shapes, dyns, outdirs, nets_name,
               input_names, output_names, framework, target):
    """Init tf graph bmodel converter"""
    super(TfGraphConverter, self).__init__(framework, base_path)
    print("{} bmodel converter init".format(model_name))
    self.model_name = model_name
    self.models_path = models_path
    self.shapes = shapes
    self.dyns = dyns
    self.outdirs = outdirs
    self.nets_name = nets_name
    self.input_names = input_names
    self.output_names = output_names
    self.target = target
    assert len(self.models_path) == len(self.shapes)
    assert len(self.models_path) == len(self.nets_name)
    assert len(self.models_path) == len(self.dyns)
    assert len(self.models_path) == len(self.outdirs)
    self.output_base_path = os.path.join(self.base_path,
                                         self.model_name + "_ir")
    if not os.path.exists(self.output_base_path):
      os.mkdir(self.output_base_path)

  def converter(self):
    """convert tf graph"""
    print("generate {} bmodel...".format(self.model_name))
    # process tf graph
    for i in range(len(self.models_path)):
      super(TfGraphConverter, self).compile_bmodel(
          model=self.models_path[i], shapes=self.shapes[i],
          dyn=self.dyns[i], net_name=self.nets_name[i],
          input_names=self.input_names[i], output_names=self.output_names[i],
          outdir=self.outdirs[i], target=self.target)
    os.system('rm -f ./bm_multi_engine_stas_0.dat')
    print("generate bmodel {}".format(self.output_base_path))

class CaffeGraphConverter(BaseConverter):
  """caffe graph bmodel converter
  """
  def __init__(self, model_name, base_path, models_path,
               weights_path, shapes, dyns, outdirs,
               framework, target, bmodel_combine):
    """Init caffe graph bmodel converter"""
    super(CaffeGraphConverter, self).__init__(framework, base_path)
    print("{} bmodel converter init".format(model_name))
    self.model_name = model_name
    self.models_path = models_path
    self.weights_path = weights_path
    self.shapes = shapes
    self.dyns = dyns
    self.outdirs = outdirs
    self.target = target
    self.bmodel_combine = bmodel_combine
    assert len(self.models_path) == len(self.weights_path)
    assert len(self.models_path) == len(self.shapes)
    assert len(self.models_path) == len(self.dyns)
    assert len(self.models_path) == len(self.outdirs)
    self.output_base_path = os.path.join(self.base_path,
                                         self.model_name + "_ir")
    if not os.path.exists(self.output_base_path):
      os.mkdir(self.output_base_path)

  def converter(self):
    """convert caffe graph"""
    print("generate {} bmodel...".format(self.model_name))
    # process caffe graph
    for i in range(len(self.models_path)):
      super(CaffeGraphConverter, self).compile_bmodel(
          model=self.models_path[i], weight=self.weights_path[i],
          target=self.target, shapes=self.shapes[i],
          dyn=self.dyns[i], outdir=self.outdirs[i])
    if self.bmodel_combine: # all bmodels need to be combined
      print("combine all sub bmodels to {}".format(self.model_name + "_ir"))
      shell_str = "bm_model.bin --combine"
      for i in range(len(self.models_path)):
        sub_str = os.path.join(self.outdirs[i], "compilation.bmodel")
        shell_str += " "
        shell_str += sub_str
      shell_str += " "
      shell_str += "-o" + " " + os.path.join(self.output_base_path,
                                             "compilation.bmodel")
      print(shell_str)
      os.system(shell_str)
    else:
      pass
    os.system('rm -f ./bm_multi_engine_stas_0.dat')
    print("generate bmodel {}".format(self.output_base_path))


class BmodelConverterCreator(object):
  """bmodel converter creator

  Attributes:
    base_path: basic file system absolute path
  """
  def __init__(self, base_path=None):
    """Inits bmodel converter creator"""
    local_path = os.getenv('SOPHON_MODEL_DIR',os.getenv('HOME'))
    if base_path is None:
      base_path = os.path.join(local_path, '.sophon/models')
   # else:
   #   base_path = os.path.join(local_path, base_path)
    self.base_path = base_path

  def load_config(self, model_name, file_path="./bmodel_config.json"):
    """config load"""
    with open(file_path, 'r') as file_:
      profile = yaml.full_load(file_)
    config = profile[model_name]
    for key in config:
      if key in ["models_path", "weights_path", "outdirs", "source_path",
                 "subgraph_path", "tfmodel_path", "json_path", "params_path"]:
        for i in range(len(config[key])):
          config[key][i] = os.path.join(self.base_path, config[key][i])
    return config

  def is_cached_bmodel(self, model_name):
    # this is for mtcnn, considering refactor the code if
    # more examples like this occurs
    old_model_name = model_name
    if model_name == 'mtcnncxx':
      old_model_name = 'mtcnn'
    model_dir = os.path.join(self.base_path, old_model_name)
    bmodel_dir = os.path.join(self.base_path, model_name + "_ir" )
    try:
      if os.path.exists(bmodel_dir):
        model_cache_md5_file = os.path.join(bmodel_dir, "md5sum.txt")
        if not os.path.exists(model_cache_md5_file):
            return False
        model_cache_md5 = open(model_cache_md5_file,'r').read()  
        model_md5 = open(os.path.join(model_dir, "md5sum.txt"),'r').read()  
        if model_md5 == model_cache_md5:
          print("cached bmodel, no need to compile")
          return True
      else:
          return False
    except OSError:
      raise OSError("md5sum.txt not exist, please download model again")
  
  def copy_md5sum(self, model_name):
    # this is for mtcnn, considering refactor the code if
    # more examples like this occurs
    old_model_name = model_name
    if model_name == 'mtcnncxx':
      old_model_name = 'mtcnn'
    model_dir = os.path.join(self.base_path, old_model_name)
    bmodel_dir = os.path.join(self.base_path, model_name + "_ir" )
    if not os.path.exists(bmodel_dir):
      os.mkdir(bmodel_dir)
    try:
      model_md5 = os.path.join(model_dir, "md5sum.txt")  
      print(model_dir)
      model_cache_md5 = os.path.join(bmodel_dir, "md5sum.txt")  
      shutil.copy(model_md5, model_cache_md5)
    except OSError:
      raise OSError("copy md5 value error.")


  def create(self, model_name):
    """create model"""
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "bmodel_config.json")
    config = self.load_config(model_name, config_path)
    if model_name in ["fasterrcnn_vgg", "mtcnn", "ssh", "mtcnncxx", "googlenet",
                      "mobilenetssd", "mobilenetv1", "mobilenetyolov3",
                      "yolov3", "resnet50", "vgg16"]:
      #Converter = CaffeGraphConverter(model_name, self.base_path, **config)
      return CaffeGraphConverter(model_name, self.base_path, **config)
    elif model_name in ["mobilenetv1_tf"]:
      #Converter = TfGraphConverter(model_name, self.base_path, **config)
      return TfGraphConverter(model_name, self.base_path, **config)
    elif model_name in ["resnext50_mx"]:
      #Converter = MXNetGraphConverter(model_name, self.base_path, **config)
      return MXNetGraphConverter(model_name, self.base_path, **config)
    elif model_name in ["resnet50_pt"]:
      #Converter = PytorchGraphConverter(model_name, self.base_path, **config)
      return PytorchGraphConverter(model_name, self.base_path, **config)
    elif model_name in ["fasterrcnn_resnet50_tf", "deeplabv3_mobilenetv2_tf",
                        "yolov3_mx"]:
      #Converter = SplitGraphConverter(model_name, self.base_path, **config)
      return SplitGraphConverter(model_name, self.base_path, **config)
    else:
      raise ValueError("not a valid model_name: {}".format(model_name))
    #return Converter
