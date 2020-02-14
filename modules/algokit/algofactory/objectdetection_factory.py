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

"""object detection factory
create various object detectors
"""
from __future__ import print_function
from ..utils.utils import load_predefined_config, load_config
from ..kitconfig import ObjDetModel
from .basefactory import BaseFactory

# detected_size: (width, height)
DEFAULT_YOLOV3_PARAM = {
    'detected_size': (416, 416),
    'threshold':
        0.5,
    'nms_threshold':
        0.45,
    'num_classes':
        80,
    'anchors': [[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45), (59, 119)],
                [(116, 90), (156, 198), (373, 326)]]
}

# global param for mobilenetssd
DEFAULT_MOBILENETSSD_PARAM = {
    'detected_size': (300, 300),
    'threshold': 0.25,
    'nms_threshold': 0.45,
    'num_classes': 21,
    'priorbox_num': 1917
}

# global param for mobilenetyolov3
DEFAULT_MOBILENETYOLOV3_PARAM = {
    'detected_size': (416, 416),
    'threshold':
        0.2,
    'nms_threshold':
        0.45,
    'num_classes':
        80,
    'anchors': [[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45), (59, 119)],
                [(116, 90), (156, 198), (373, 326)]]
}

# global param for fasterrcnn
DEFAULT_FASTERRCNN_PARAM = {
    'detected_size': (600, 800),
    'thresholds': [0.6, 0.3],
    'num_classes': 21,
    'global_cfg': {
        'pre_nms_topN': 300,
        'post_nms_topN': 32,
        'anchor_min_size': 16,
        'nms_threshold': 0.7
    }
}


class ObjectDetector(BaseFactory):
  """Object Detection algorithm factory

  The algorithm that completes object detection.
  """

  def __init__(self, model_path=None):
    """Inits ObjectDetector with input model path.
    """
    super(ObjectDetector, self).__init__(model_path)

  def get_supported_models(self):
    """Get supported object detection algorithms list in algokit.
    """
    return list(algo_name.value for algo_name in ObjDetModel)

  def create(self, detector_model, mode, config_path=None, extra_args=None):
    """Create a objectdetector.
    """
    if detector_model is ObjDetModel.YOLOV3:
      from ..algo_cv.det.object_detection_yolov3 import\
          ObjectDetectionYOLOV3 as Detector
    elif detector_model is ObjDetModel.MOBILENETSSD:
      from ..algo_cv.det.object_detection_mobilenetssd import\
          ObjectDetectionMOBILENETSSD as Detector
    elif detector_model is ObjDetModel.MOBILENETYOLOV3:
      from ..algo_cv.det.object_detection_mobilenetyolov3 import\
          ObjectDetectionMOBILENETYOLOV3 as Detector
    elif detector_model is ObjDetModel.FASTERRCNN:
      from ..algo_cv.det.object_detection_fasterrcnn import\
          ObjectDetectionFASTERRCNN as Detector
    else:
      raise ValueError('not a valid detector_model:', detector_model)
    if config_path is None:
      arch_config = load_predefined_config(self.model_path, mode,
                                           detector_model.value)
    else:
      arch_config = load_config(config_path, self.model_path, mode)
    arch_config = self.load_param(detector_model, arch_config, extra_args)
    detector = Detector(**arch_config)
    return detector

  def set_param(self):
    """set algorithm default param
    """
    print("to be added")

  def load_param(self, detector_model, arch_config, extra_args):
    """Load all arguments into a dict
        """
    if detector_model is ObjDetModel.YOLOV3:
      default_param = DEFAULT_YOLOV3_PARAM
    elif detector_model is ObjDetModel.MOBILENETSSD:
      default_param = DEFAULT_MOBILENETSSD_PARAM
    elif detector_model is ObjDetModel.MOBILENETYOLOV3:
      default_param = DEFAULT_MOBILENETYOLOV3_PARAM
    elif detector_model is ObjDetModel.FASTERRCNN:
      default_param = DEFAULT_FASTERRCNN_PARAM
    else:
      raise ValueError('not a valid detector_model:', detector_model)
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
