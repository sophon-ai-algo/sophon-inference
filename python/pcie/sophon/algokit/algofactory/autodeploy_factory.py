# Copyright 2016-2022 Bitmain Technologies Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""autodeploy algo factory
   create the algorithm from autodeploy, based on autodeploy
three stages:
    1. split model graph
    2. compile subgraph (bmnetT/bmnetM)
    3. run algorithm with tpu&&cpu
detector/classifier in this factory is different from original caffe
"""
from __future__ import print_function
from ..kitconfig import ObjDetModel, SegmentationModel
from .basefactory import BaseFactory
from ..utils.utils import load_autodeploy_config

# global param for fasterrcnn_resnet50_tf
DEFAULT_FASTERRCNN_RESNET50_TF_PARAM = {
    'detected_size': (600, 800),
    'conf_threshold': 0.5
}
# global param for deeplabv3_mobilenetv2_tf
DEFAULT_DEEPLABV3_MOBILENETV2_TF_PARAM = {
    'process_size': (513, 513),
    'conf_threshold': 0.5
}
# global param for yolov3_mx
DEFAULT_YOLOV3_MX_PARAM = {
    'detected_size': (512, 672),
    'conf_threshold': 0.5
}


class ObjectDetector(BaseFactory):
  """Autodeploy Object detection algo factory.

    The algorithm that need to be split with autodeploy.
    The algo factory construction is different from normal.

    Attributes:
        model_path: algorithm model path.
  """

  def __init__(self, model_path=None):
    """Inits ObjectDetector with input model path.
    """
    super(ObjectDetector, self).__init__(model_path)

  def create(self, detector_model, mode, config_path=None, extra_args=None):
    """Create a Object detector which is generated with autodeploy."""
    if detector_model is ObjDetModel.FASTERRCNN_RESNET50_TF:
      from ..algo_cv.det.object_detection_fasterrcnn_resnet50_tf \
          import ObjectDetectionFASTERRCNNRESNET50TF as Detection
    elif detector_model is ObjDetModel.YOLOV3_MX:
      from ..algo_cv.det.object_detection_yolov3_mx \
          import ObjectDetectionYOLOV3MX as Detection
    else:
      raise ValueError('not a valid detector_model: ', detector_model)
    detector_config = load_autodeploy_config(self.model_path, \
        mode, detector_model.value, config_path)
    detector_config = load_extra_param(detector_model, \
        detector_config, extra_args)
    detector = Detection(**detector_config)
    return detector


class SemanticSegment(BaseFactory):
  """Autodeploy SemanticSegment algo factory

    The algorithm that need to be split with autodeploy.

    Attributes:
        model_path: algorithm model path.
  """

  def __init__(self, model_path=None):
    """Inits SemanticSegmant with input model paph
    """
    super(SemanticSegment, self).__init__(model_path)

  def get_supported_models(self):
    """Get supported segmentation algorithms list in algokit."""
    return list(algo_name.value for algo_name in SegmentationModel)

  def create(self, semantic_model, mode, config_path=None, extra_args=None):
    """Create a Segment which is generated with autodeploy."""
    if semantic_model is SegmentationModel.DEEPLABV3_MOBILENETV2_TF:
      from ..algo_cv.seg.semantic_segmentation_deeplabv3_mobilenetv2_tf \
          import SemanticSegmentationDEEPLABV3MOBILENETV2TF as Segmentation
    else:
      raise ValueError('not a valid detector_model: ', semantic_model)
    semantic_config = load_autodeploy_config(self.model_path, \
        mode, semantic_model.value, config_path)
    semantic_config = load_extra_param(semantic_model, \
        semantic_config, extra_args)
    segment = Segmentation(**semantic_config)
    return segment


def load_extra_param(model, arch_config, extra_args):
  """Unified all arguments to a dict

    Combine all model arguments into a dict

    Args:
        model: model type
        arch_config: model arch config arguments
        extra_args: extra arguments

    Returns:
        A dict mapping the all arguments when algorithm construct
  """
  if model is ObjDetModel.FASTERRCNN_RESNET50_TF:
    default_param = DEFAULT_FASTERRCNN_RESNET50_TF_PARAM
  elif model is ObjDetModel.YOLOV3_MX:
    default_param = DEFAULT_YOLOV3_MX_PARAM
  elif model is SegmentationModel.DEEPLABV3_MOBILENETV2_TF:
    default_param = DEFAULT_DEEPLABV3_MOBILENETV2_TF_PARAM
  else:
    raise ValueError('not a valid model:', model)
  if extra_args is not None:
    for key in extra_args:
      if key not in arch_config:
        arch_config[key] = extra_args[key]
  for key in default_param:
    if key in arch_config:
      pass
    else:
      arch_config[key] = default_param[key]
  return arch_config
