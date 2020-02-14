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

"""The implementation of face detection ssh + onet
"""
from __future__ import print_function
from __future__ import division
import numpy as np
import yaml
from ...engine.base_engine import BaseEngine
from .face_detection_mtcnn import ONet
from ...libs.extend_layer.sshproposal_layer import SSHProposalLayer
from ...engine import Engine
from ...utils.box_operation import restore_param, fix_bbox_boundary, nms


class FaceDetectionSSH(BaseEngine):
  """Construct ssh face detector
  """

  def __init__(self, ssh, onet, detected_size, thresholds, global_cfg,
               ssh_xform, onet_xform):
    super(FaceDetectionSSH, self).__init__()
    # detected_size: (h, w)
    # image will be resize before processing detection
    # thresholds:
    #   [confidence_of_ssh, ssh_nms_thres, landmark_thres, landmark_nms_thres]
    # global_cfg: pre_nms_topN anchor_min_size
    # return bbox with respect to original image
    self.proposal_net = []  # detection module m1 m2 m3
    for j in range(3):
      proposal_param = yaml.load(get_proposalnet_param()[j])
      self.proposal_net.append(SSHProposalLayer(proposal_param, global_cfg))
    # ssh
    self.net = Engine(ssh['arch'], ssh['mode'])
    # Onet
    if onet is None:
      self.landmark = False
    else:
      self.landmark = True
      self.onet = ONet(
          thresholds=thresholds[2],
          nms_thresholds=thresholds[3],
          only_points=True,
          **onet)

    self.conf_thresh = thresholds[0]
    self.ssh_nms_thresh = thresholds[1]
    self.detected_size = detected_size
    self.pyramid = False
    self.ssh_xform = ssh_xform
    self.onet_xform = onet_xform
    self.input_names = ssh["arch"]["input_names"]

  def ssh_predict(self, images_stack, conf_thresh, pyramid):
    """ssh forward include proposal layer
    """
    ssh_bboxes = []
    if not pyramid:
      for image in images_stack:
        bboxes = []
        probs = []
        input_data = {self.input_names[0]: np.array([image], dtype=np.float32)}
        scale = 1.0  # rescale before detection
        # ssh inference without proposal layer
        out = self.net.predict(input_data)
        # ssh m1 m2 m3 proposal inference (to be optimized)
        for i in range(3):
          input_blob = []
          input_blob.append(out['m{}@ssh_cls_prob_reshape_output'.format(i +
                                                                         1)])
          input_blob.append(out['m{}@ssh_bbox_pred_output'.format(i + 1)])
          input_blob.append(
              np.array([[
                  input_data[self.input_names[0]].shape[2],
                  input_data[self.input_names[0]].shape[3], scale
              ]],
                       dtype=np.float32))
          bbox, score = self.proposal_net[i].forward(input_blob)
          bboxes.append(bbox)
          probs.append(score)
        bboxes = np.vstack((bboxes[0], bboxes[1], bboxes[2]))
        bboxes = bboxes[:, 1:5] / scale
        probs = np.vstack((probs[0], probs[1], probs[2]))
        pred_boxes = np.tile(bboxes, (1, probs.shape[1]))  # ? exist issue
        bboxes = pred_boxes[:, 0:4]
        print("detected face num: {}".format(bboxes.shape))
        inds = np.where(probs[:, 0] > conf_thresh)[0]
        probs = probs[inds, 0]
        print("detected face num: {}".format(bboxes.shape))
        bboxes = bboxes[inds, :]
        nms_boxes = np.hstack((bboxes, probs[:, np.newaxis]))
        keep = nms(nms_boxes, self.ssh_nms_thresh, 'union')
        bboxes = nms_boxes[keep, :]
        print("detected face num: {}".format(bboxes.shape))
        ssh_bboxes.append(bboxes)
    else:
      pass
    return ssh_bboxes

  def predict(self, images, detected_size=None):
    """ssh + onet forward
    """
    self.time.tic()
    if detected_size is None:
      detected_size = self.detected_size
    if isinstance(images, list):
      use_batch = True
    else:
      use_batch = False
      images = [images]

    # (x, y)
    onet_images_stack = []
    ssh_images_stack = []
    rescale_params = []
    image_sizes = []
    # preprocessing
    for image in images:
      if isinstance(type(image), np.ndarray) or image is None:
        raise ValueError("image is not valid np.ndarray {}".format(type(image)))
      image_sizes.append(image.shape)
      image, rescale_param = self.rescale_image(image, detected_size, True)
      onet_images_stack.append(self.xform_img(image, self.onet_xform)[0])
      ssh_images_stack.append(self.xform_img(image, self.ssh_xform)[0])
      rescale_params.append(rescale_param)

    # forward
    bboxes_stack = self.ssh_predict(ssh_images_stack, self.conf_thresh,
                                    self.pyramid)
    # onet or not
    nbatch = 0
    result_bboxes = []
    result_points = []
    result_probs = []
    for image in onet_images_stack:
      bboxes = bboxes_stack[nbatch]
      if self.landmark:
        # MTCNN read image with RGB matlab format
        # image_rgb = self.bgr2rgb(image)
        bboxes, points = self.onet(image, bboxes)
      else:
        points = None
      if np.array(bboxes).shape[0] > 0:
        bboxes, points = restore_param(bboxes, points, rescale_params[nbatch])
        bboxes = fix_bbox_boundary(bboxes, image_sizes[nbatch])
        probs = [np.asscalar(bbox[-1]) for bbox in bboxes]
        bboxes = [np.fix(bbox[:4]).tolist() for bbox in bboxes]
        area = np.array([
            (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes
        ]) * (-1)
        argidx = np.argsort(area)
        bboxes = np.array(bboxes)[argidx]
        probs = np.array(probs)[argidx]
        result_bboxes.append(bboxes.tolist())
        result_probs.append(probs.tolist())
        if points is None:
          result_points.append([None])
        else:
          points = np.array(points)[argidx]
          result_points.append(points.tolist())
      else:
        result_bboxes.append([])
        result_points.append([])
        result_probs.append([])
      nbatch += 1
    if use_batch:
      return (result_bboxes, result_points, result_probs)
    return (result_bboxes[0], result_points[0], result_probs[0])


def get_proposalnet_param():
  """ssh detection module M1 M2 M3 params
  """
  param_str = [
      "{'feat_stride': 8, 'base_size': 16, 'scales': [1,2], 'ratios':[1,]}",
      "{'feat_stride': 16, 'base_size': 16, 'scales': [4,8], 'ratios':[1,]}",
      "{'feat_stride': 32, 'base_size': 16, 'scales': [16,32], 'ratios':[1,]}"
  ]
  return param_str
