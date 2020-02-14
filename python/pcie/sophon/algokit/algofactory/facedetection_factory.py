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

"""face detection factory
create various face detectors
"""
from __future__ import print_function
from ..utils.utils import load_predefined_config, load_config
from ..kitconfig import FaceDetModel
from .basefactory import BaseFactory

# FD mtcnn param
DEFAULT_MTCNN_PARAM = {
    'detected_size': (640, 800),
    'thresholds': [0.6, 0.6, 0.7],
    'nms_thresholds': [0.5, 0.7, 0.7],
    'min_size': 40,  # 40 -> 20
    'factor': 0.89
}  # 0.709 -> 0.89
# FD ssh param
DEFAULT_SSH_PARAM = {
    'detected_size': (600, 800),
    'thresholds': [0.5, 0.3, 0.7, 0.7],
    'global_cfg': {
        'pre_nms_topN': 500,
        'anchor_min_size': 0
    }
}


class FaceDetector(BaseFactory):
  """Face Detection algorithm factory.

  The algorithm that completes face detections.
  """

  def __init__(self, model_path=None):
    """Inits FaceDetector with input model path."""
    super(FaceDetector, self).__init__(model_path)

  def get_supported_models(self):
    """Get supported face detection algorithms list in algokit."""
    return list(algo_name.value for algo_name in FaceDetModel)

  def create(self, detector_model, mode, config_path=None, extra_args=None):
    """Create a facedetector."""
    if detector_model is FaceDetModel.MTCNN:
      from ..algo_cv.det.face_detection_mtcnn \
          import FaceDetectionMTCNN as Detection
    elif detector_model is FaceDetModel.SSH:
      from ..algo_cv.det.face_detection_ssh \
          import FaceDetectionSSH as Detection
    else:
      raise ValueError('not a valid detector_model:', detector_model)
    if config_path is None:
      arch_config = load_predefined_config(\
          self.model_path, mode, detector_model.value)
    else:
      arch_config = load_config(config_path, self.model_path, mode)
    arch_config = self.load_param(detector_model, arch_config, extra_args)
    detector = Detection(**arch_config)
    return detector

  def set_param(self):
    """ set algorithm default param
    """
    print("to be added")

  def load_param(self, detector_model, arch_config, extra_args):
    """Load all arguments into a dict
    """
    # unified all args to one
    if detector_model is FaceDetModel.MTCNN:
      default_param = DEFAULT_MTCNN_PARAM
    elif detector_model is FaceDetModel.SSH:
      default_param = DEFAULT_SSH_PARAM
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
