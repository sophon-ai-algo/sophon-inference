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

"""The global config info of algokit
"""
from enum import Enum


class ChipMode(Enum):
  """The deployment environment of algokit
  """
  BM1682 = 'BM1682'
  BM1684 = 'BM1684'


class ObjDetModel(Enum):
  """The object detection algorithm model
  """
  YOLOV3 = 'yolov3'
  MOBILENETSSD = 'mobilenetssd'
  MOBILENETYOLOV3 = 'mobilenetyolov3'
  FASTERRCNN = 'fasterrcnn_vgg'
  FASTERRCNN_RESNET50_TF = 'fasterrcnn_resnet50_tf'
  YOLOV3_MX = 'yolov3_mx'


class FaceDetModel(Enum):
  """The face detection algorithm model
  """
  MTCNN = 'mtcnn'
  SSH = 'ssh'


class ClassificationModel(Enum):
  """The classification algorithm model
  """
  GOOGLENET = 'googlenet'
  VGG16 = 'vgg16'
  RESNET50 = 'resnet50'
  MOBILENETV1 = 'mobilenetv1'
  MOBILENETV1_TF = 'mobilenetv1_tf'
  RESNEXT50_MX = 'resnext50_mx'
  RESNET50_PT = 'resnet50_pt'


class SegmentationModel(Enum):
  """The segmentation algorithm model
  """
  DEEPLABV3_MOBILENETV2_TF = 'deeplabv3_mobilenetv2_tf'
