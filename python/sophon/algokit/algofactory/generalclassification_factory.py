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

"""general classification factory
create various general classifier
"""
from __future__ import print_function
from ..utils.utils import load_predefined_config, load_config
from ..kitconfig import ClassificationModel
from .basefactory import BaseFactory

DEFAULT_GOOGLENET_PARAM = {'num_classes': 1000}
DEFAULT_VGG16_PARAM = {'num_classes': 1000}
DEFAULT_RESNET50_PARAM = {'num_classes': 1000}
DEFAULT_MOBILENETV1_PARAM = {'num_classes': 1000}
DEFAULT_MOBILENETV1_TF_PARAM = {'num_classes': 1001}
DEFAULT_RESNEXT50_MX_PARAM = {'num_classes': 1000}
DEFAULT_RESNET50_PT_PARAM = {'num_classes': 1000}


class GeneralClassifier(BaseFactory):
  """GeneralClassifier algorithm factory.

  The algorithm that completes general classification.
  """

  def __init__(self, model_path=None):
    """Inits GeneralClassifier with input model path
    """
    super(GeneralClassifier, self).__init__(model_path)

  def get_supported_models(self):
    """Get supported general classification algorithms list in algokit
    """
    return list(algo_name.value for algo_name in ClassificationModel)

  def create(self, classifier_model, mode, config_path=None, extra_args=None):
    """Create a classifier
    """
    if classifier_model in \
        [ClassificationModel.GOOGLENET, ClassificationModel.VGG16, \
        ClassificationModel.MOBILENETV1, ClassificationModel.RESNET50]:
      from ..algo_cv.cls.general_classification \
          import GeneralClassification as Classification
    elif classifier_model in \
        [ClassificationModel.MOBILENETV1_TF]:
      from ..algo_cv.cls.general_classification_tf \
          import GeneralClassificationTF as Classification
    elif classifier_model in \
        [ClassificationModel.RESNEXT50_MX]:
      from ..algo_cv.cls.general_classification_mx \
          import GeneralClassificationMX as Classification
    elif classifier_model in \
        [ClassificationModel.RESNET50_PT]:
      from ..algo_cv.cls.general_classification_pt \
          import GeneralClassificationPT as Classification
    else:
      raise ValueError('not a valid classifer_model:', classifier_model)
    if config_path is None:
      arch_config = load_predefined_config(\
          self.model_path, mode, classifier_model.value)
    else:
      arch_config = load_config(config_path, self.model_path, mode)
    arch_config = self.load_param(classifier_model, arch_config, extra_args)
    classifier = Classification(**arch_config)
    return classifier

  def set_param(self):
    """set algorithm default param
    """
    print("to be added")

  def load_param(self, classifier_model, arch_config, extra_args):
    """Load all arguments into a dict
    """
    # unified all args to one
    if classifier_model is ClassificationModel.GOOGLENET:
      default_param = DEFAULT_GOOGLENET_PARAM
    elif classifier_model is ClassificationModel.VGG16:
      default_param = DEFAULT_VGG16_PARAM
    elif classifier_model is ClassificationModel.RESNET50:
      default_param = DEFAULT_RESNET50_PARAM
    elif classifier_model is ClassificationModel.MOBILENETV1:
      default_param = DEFAULT_MOBILENETV1_PARAM
    elif classifier_model is ClassificationModel.MOBILENETV1_TF:
      default_param = DEFAULT_MOBILENETV1_TF_PARAM
    elif classifier_model is ClassificationModel.RESNEXT50_MX:
      default_param = DEFAULT_RESNEXT50_MX_PARAM
    elif classifier_model is ClassificationModel.RESNET50_PT:
      default_param = DEFAULT_RESNET50_PT_PARAM
    else:
      raise ValueError('not a valid classifier_model:', classifier_model)

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
