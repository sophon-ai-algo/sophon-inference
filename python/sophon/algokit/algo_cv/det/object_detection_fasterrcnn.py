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

"""The implementation of object detection fasterrcnn
"""
from __future__ import print_function
from __future__ import division
import numpy as np
np.set_printoptions(precision=5)
import yaml
from ...engine.base_engine import BaseEngine
from ...libs.extend_layer.fasterrcnnproposal_layer \
  import FasterRcnnProposalLayer, clip_boxes
from ...engine import Engine
from ...utils.box_operation \
  import restore_param, fix_bbox_boundary, nms, bbox_transform_inv


class ObjectDetectionFASTERRCNN(BaseEngine):
  """Construct fasterrcnn object detector
  """

  def __init__(self, stage1, stage2, detected_size, thresholds, num_classes,
               global_cfg, xform):
    super(ObjectDetectionFASTERRCNN, self).__init__()
    # detected_size: (h, w)
    # image will be resize before processing detection
    # thresholds: [confidence_of_ssh, ssh_nms_thres,
    #              landmark_thres, landmark_nms_thres]
    # global_cfg: pre_nms_topN anchor_min_size
    # return bbox with respect to original image
    self.proposal_net = []  # detection module RPN
    proposal_param = yaml.load(get_proposalnet_param()[0])
    self.proposal_net.append(
        FasterRcnnProposalLayer(proposal_param, global_cfg))
    # fasterrcnn stage1 conv
    self.net_stage1 = Engine(stage1['arch'], stage1['mode'])
    self.net_stage2 = Engine(stage2['arch'], stage2['mode'])

    self.conf_thresh = thresholds[0]  # 0.6
    self.nms_thresh = thresholds[1]  # 0.3
    self.detected_size = detected_size  # (h, w)
    self.num_classes = num_classes  # 21 for voc
    self.xform = xform
    self.stage1_input_names = stage1['arch']['input_names']
    self.stage2_input_names = stage2['arch']['input_names']

  def fasterrcnn_postprocess(self, bboxes, probs, restoreparam, image_size):
    """decode fasterrcnn detected bboxes

    Returns:
        bboxes:list
        classes:list
        probs:list
    """
    output_bboxes = None
    output_classes = None
    output_probs = None
    # print(restoreparam, image_size)
    for cls_ind in range(self.num_classes - 1):
      cls_ind += 1  # ignore background
      cls_bboxes = bboxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
      cls_probs = probs[:, cls_ind]
      bboxes_mask = (cls_probs > self.conf_thresh)
      cls_bboxes = cls_bboxes[bboxes_mask, :]
      cls_probs = cls_probs[bboxes_mask]
      cls_classes = np.zeros((cls_bboxes.shape[0], 1)) + cls_ind
      # stack bbox prob classes_id
      dets = np.hstack((cls_bboxes, cls_probs[:, np.newaxis],
                        cls_classes)).astype(np.float32)
      keep = nms(dets, self.nms_thresh, 'union')
      # print("cls_id: {}, keep: {}".format(cls_ind, keep))
      if np.array(keep).shape[0] == 0:
        continue
      dets = dets[keep, :]
      bbox = dets[:, 0:4]
      bbox, _ = restore_param(bbox, None, restoreparam)
      bbox = fix_bbox_boundary(bbox, image_size)
      prob = dets[:, 4]
      cls = dets[:, 5].astype(np.int32)
      if output_bboxes is None:
        output_bboxes = bbox
        output_classes = cls
        output_probs = prob
      else:
        output_bboxes = np.concatenate((output_bboxes, bbox))
        output_classes = np.concatenate((output_classes, cls))
        output_probs = np.concatenate((output_probs, prob))
    return output_bboxes, output_classes, output_probs

  def fasterrcnn_predict(self, images_stack, restore_params, image_sizes):
    """fasterrnn forward include proposal layer
        net_stage1 -> region proposal net -> net_stage2

        network stage     |     input     |     output
        net_stage1        |   input_image   | conv5_3 + rpn_box_pred
                                              + rpn_cls_prob_reshape
        rpn               | rpn_box_pred + rpn_cls_prob_reshape | rois
        net_stage2(roi_pooling)  | rois + conv5_3 | bbox_pred + cls_prob
        """
    fasterrcnn_bboxes = []
    fasterrcnn_classes = []
    fasterrcnn_probs = []
    for image, restoreparam, image_size in zip(images_stack, restore_params,
                                               image_sizes):
      input_data = \
              {self.stage1_input_names[0]: np.array([image], dtype=np.float32)}
      scale = 1.0  # rescale before object detection
      # net_stage1 inference
      stage1_output = self.net_stage1.predict(input_data)
      # print("conv5_3_output:  {} {}".format(\
      #   stage1_output['conv5_3_output'].shape, \
      #   stage1_output['conv5_3_output']))
      # np.save('conv5_3.npy', stage1_output['conv5_3_output'])
      # fasterrcnn region proposal network inference (to be optimized)
      input_blob = []
      input_blob.append(stage1_output['rpn_cls_prob_reshape'])
      input_blob.append(stage1_output['rpn_bbox_pred'])
      im_info = np.array([[input_data[self.stage1_input_names[0]].shape[2], \
              input_data[self.stage1_input_names[0]].shape[3], scale]], \
              dtype=np.float32)
      input_blob.append(im_info)
      rpn_rois, _ = self.proposal_net[0].forward(input_blob) # rois scores
      # print ("rois_score_shape: {}, \
      #   rois_score: {}".format(rois_score.shape, rois_score))
      # print ("rois_shape: {}, rois: {}".format(rpn_rois.shape, rpn_rois))
      # np.save("rois.npy", rpn_rois)
      # net_stage2 inference roi_pooling input_shape: 1*512*h*w; rois_num*5
      # reshape rpn_rois shape rois_num*5 -> rois_num*5*1*1 -> rois_num*5
      # rpn_rois = rpn_rois[np.newaxis, :]
      rois_data = rpn_rois
      input_data = {}
      input_data[self.stage2_input_names[0]] = stage1_output[
          'conv5_3_output']  # conv5_3_input
      input_data[self.stage2_input_names[1]] = rois_data  # rois
      stage2_output = self.net_stage2.predict(input_data)
      # rois_data = rpn_rois[0]
      # unscale back to raw image space
      bboxes = rpn_rois[:, 1:5] / scale
      probs = stage2_output['cls_prob']
      # print("cls_prob: {}".format(probs))
      # apply bbox regression deltas
      bbox_deltas = stage2_output['bbox_pred']
      pred_boxes = bbox_transform_inv(bboxes, bbox_deltas)
      pred_boxes = clip_boxes(pred_boxes, im_info[0])
      # decode coordinate and score
      post_bboxes, post_classes, post_probs = \
          self.fasterrcnn_postprocess(pred_boxes, \
                probs, restoreparam, image_size)
      fasterrcnn_bboxes.append(post_bboxes)
      fasterrcnn_classes.append(post_classes)
      fasterrcnn_probs.append(post_probs)
    return fasterrcnn_bboxes, fasterrcnn_classes, fasterrcnn_probs

  def predict(self, images, detected_size=None):
    """fasterrcnn forward
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
    input_images_stack = []
    rescale_params = []
    image_sizes = []
    # preprocessing
    for image in images:
      if isinstance(type(image), np.ndarray) or image is None:
        raise ValueError("image is not valid np.ndarray {}".format(type(image)))
      image_sizes.append(image.shape)
      # this version fasterrcnn input fix 600*800(h*w) padding location center
      image, rescale_param = self.rescale_image(image, detected_size, True)
      input_images_stack.append(self.xform_img(image, self.xform)[0])
      rescale_params.append(rescale_param)
    # forward
    result_bboxes, result_classes, result_probs = \
        self.fasterrcnn_predict(input_images_stack, rescale_params, image_sizes)
    if use_batch:
      return (result_bboxes, result_classes, result_probs)
    else:
      return (result_bboxes[0], result_classes[0], result_probs[0])


def get_proposalnet_param():
  """fasterrcnn RPN params
    """
  param_str = [
      "{'feat_stride': 16, 'base_size': 16, \
        'scales': [8, 16, 32], 'ratios':[0.5, 1, 2]}"
  ]
  return param_str
