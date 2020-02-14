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

"""The implementation of face detection mtcnn
"""
from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
from ...engine.base_engine import BaseEngine
from ...utils.extend_math import softmax
from ...utils.box_operation import restore_param, fix_bbox_boundary, nms
from ...engine import Engine


class FaceDetectionMTCNN(BaseEngine):
  """Construct mtcnn face detector
  """

  def __init__(self,
               pnet,
               rnet,
               onet,
               detected_size,
               thresholds,
               nms_thresholds,
               xform,
               min_size=20,
               factor=0.709):
    super(FaceDetectionMTCNN, self).__init__()
    # detected_size: (h, w)
    # image will be resize before processing detection
    # init P/R/Onet
    self.pnet = PNet(
        thresholds=thresholds[0],
        nms_thresholds=nms_thresholds[0],
        min_size=min_size,
        factor=factor,
        **pnet)
    self.rnet = RNet(
        thresholds=thresholds[1], nms_thresholds=nms_thresholds[1], **rnet)
    self.onet = ONet(
        thresholds=thresholds[2], nms_thresholds=nms_thresholds[2], **onet)
    self.detected_size = detected_size
    self.xform = xform

  def predict(self, images, detected_size=None):
    self.time.tic()
    if isinstance(images, list):
      use_batch = True
    else:
      use_batch = False
      images = [images]

    result_bboxes = []
    result_points = []
    result_probs = []
    for image in images:
      image_size = image.shape
      # MTCNN read image with RGB matlab format
      # image = self.bgr2rgb(image)

      if detected_size is None:
        detected_size = self.detected_size

      image, rescale_param = self.rescale_image(image, detected_size, True)
      image = self.xform_img(image, self.xform)[0]
      # three stages
      print("========== PNet ==========")
      bboxes = self.pnet.predict(image)
      print("========== RNet ==========")
      bboxes = self.rnet.predict(image, bboxes)
      print("========== ONet ==========")
      bboxes, points = self.onet.predict(image, bboxes)

      if np.array(bboxes).shape[0] > 0:
        bboxes, points = restore_param(bboxes, points, rescale_param)
        bboxes = fix_bbox_boundary(bboxes, image_size)
        # probs = [bbox[-1] for bbox in bboxes]
        probs = [np.asscalar(bbox[-1]) for bbox in bboxes]
        bboxes = [np.fix(bbox[:4]) for bbox in bboxes]
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
    self.time.toc()
    if use_batch:
      return (result_bboxes, result_points, result_probs)
    else:
      return (result_bboxes[0], result_points[0], result_probs[0])

  def set_param(self, config):
    """set mtcnn three stages param
    """
    pnet_param = dict()
    rnet_param = dict()
    onet_param = dict()
    for key in config:
      if key == 'thresholds':
        pnet_param['thresholds'] = config[key][0]
        rnet_param['thresholds'] = config[key][1]
        onet_param['thresholds'] = config[key][2]
      elif key == 'nms_thresholds':
        pnet_param['nms_thresholds'] = config[key][0]
        rnet_param['nms_thresholds'] = config[key][1]
        onet_param['nms_thresholds'] = config[key][2]
      elif key == 'min_size':
        pnet_param['min_size'] = config[key]
    self.pnet.set_param(pnet_param)
    self.rnet.set_param(rnet_param)
    self.onet.set_param(onet_param)


class PNet(BaseEngine):
  """Construct pnet
  """

  def __init__(self, arch, mode, thresholds, nms_thresholds, min_size, factor):

    super(PNet, self).__init__()
    self.net = Engine(arch, mode)
    self.thresholds = thresholds
    self.nms_thresholds = nms_thresholds
    self.detected_min_size = min_size  # 20
    self.factor = factor  # 0.709
    self.input_names = arch['input_names']  # ['data',...]

  def set_param(self, param):
    """set pnet param
    """
    for key in param:
      if key == 'thresholds':
        self.thresholds = param[key]
      elif key == 'nms_thresholds':
        self.nms_thresholds = param[key]
      elif key == 'min_size':
        self.detected_min_size = param[key]

  def predict(self, img):
    self.time.tic()
    factor_count = 0
    total_boxes = np.zeros((0, 9), np.float)
    org_h = img.shape[0]
    org_w = img.shape[1]
    minl = min(org_h, org_w)  # minvalue between w&&h
    img = img.astype(np.float32)
    m_scale = 12.0 / self.detected_min_size
    minl = minl * m_scale

    # create scale pyramid
    scales = []
    while minl >= 12:
      scales.append(m_scale * pow(self.factor, factor_count))
      minl *= self.factor
      factor_count += 1

    # first stage
    for scale in scales:
      resize_h = int(np.ceil(org_h * scale))
      resize_w = int(np.ceil(org_w * scale))
      im_data = cv2.resize(img, (resize_w, resize_h))  # default is bilinear
      im_data = np.swapaxes(im_data, 0, 2)
      im_data = np.array([im_data], dtype=np.float32)
      input_data = {self.input_names[0]: im_data}
      # print("Pnet input name:{}".format(self.input_names[0]))
      out = self.net.predict(input_data)
      if 'prob1' not in out:
        out['prob1'] = softmax(out['conv4-1'], 1, 1)
      boxes = generatebbox(out['prob1'][0, 1, :, :],\
          out['conv4-2'][0], scale, self.thresholds)
      if boxes.shape[0] != 0:
        pick = nms(boxes, 0.5, 'Union')
        if np.array(pick).shape[0] > 0:
          boxes = boxes[pick, :]
      if boxes.shape[0] != 0:
        total_boxes = np.concatenate((total_boxes, boxes), axis=0)

    numbox = total_boxes.shape[0]
    if numbox > 0:
      # nms
      pick = nms(total_boxes, self.nms_thresholds, 'Union')
      total_boxes = total_boxes[pick, :]
      # logging.debug("[2]: {:d}".format(total_boxes.shape[0]))

      # revise and convert to square
      regh = total_boxes[:, 3] - total_boxes[:, 1]
      regw = total_boxes[:, 2] - total_boxes[:, 0]
      t_1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
      t_2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
      t_3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
      t_4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
      t_5 = total_boxes[:, 4]
      total_boxes = np.array([t_1, t_2, t_3, t_4, t_5]).T
      total_boxes = rerec(total_boxes)  # convert box to square
      # logging.debug("[4]: {:d}".format(total_boxes.shape[0]))

      total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4])
      # logging.debug("[4.5]: {:d}".format(total_boxes.shape[0]))
    self.time.toc()
    return total_boxes


class RNet(BaseEngine):
  """Construct rnet
  """

  def __init__(self, arch, mode, thresholds, nms_thresholds):

    super(RNet, self).__init__()
    self.net = Engine(arch, mode)
    self.thresholds = thresholds
    self.nms_thresholds = nms_thresholds
    self.input_names = arch['input_names']

  def set_param(self, param):
    """set rnet param
    """
    for key in param:
      if key == 'thresholds':
        self.thresholds = param[key]
      elif key == 'nms_thresholds':
        self.nms_thresholds = param[key]

  def predict(self, image, total_boxes):
    self.time.tic()
    width = image.shape[1]
    height = image.shape[0]
    numbox = total_boxes.shape[0]
    if numbox == 0:
      return total_boxes

    [d_y, edy, d_x, edx, b_y, e_y, b_x, ex, tmpw, tmph] =\
        pad(total_boxes, width, height)
    # second stage
    # construct input for RNet
    tempimg = np.zeros((numbox, 24, 24, 3), np.float32)  # (24, 24, 3, numbox)
    for k in range(numbox):
      tmp = np.zeros((int(tmph[k]) + 1, int(tmpw[k]) + 1, 3))
      tmp[int(d_y[k]):int(edy[k])+1, int(d_x[k]):int(edx[k])+1] =\
          image[int(b_y[k]):int(e_y[k])+1, int(b_x[k]):int(ex[k])+1]
      tempimg[k, :, :, :] = cv2.resize(tmp, (24, 24))
    # RNet
    tempimg = np.swapaxes(tempimg, 1, 3)
    input_data = {self.input_names[0]: tempimg}
    out = self.net.predict(input_data)
    if 'prob1' not in out:
      out['prob1'] = softmax(out['conv5-1'], 1, 1)
    score = out['prob1'][:, 1]
    pass_t = np.where(score > self.thresholds)[0]
    score = score[pass_t].reshape(-1, 1)
    total_boxes = np.concatenate((total_boxes[pass_t, 0:4], score), axis=1)
    mv_out = np.squeeze(out['conv5-2'][pass_t, :]).T
    if total_boxes.shape[0] > 0:
      pick = nms(total_boxes, self.nms_thresholds, 'Union')
      if np.array(pick).shape[0] > 0:
        total_boxes = total_boxes[pick, :]
        total_boxes = bbreg(total_boxes, mv_out[:, pick])
        total_boxes = rerec(total_boxes)
    self.time.toc()
    return total_boxes


class ONet(BaseEngine):
  """Construct onet
  """

  def __init__(self, arch, mode, thresholds, nms_thresholds, only_points=False):

    super(ONet, self).__init__()
    self.net = Engine(arch, mode)
    self.mode = mode
    self.thresholds = thresholds
    self.nms_thresholds = nms_thresholds
    self.only_points = only_points
    self.input_names = arch['input_names']

  def set_param(self, param):
    """set onet param
    """
    for key in param:
      if key == 'thresholds':
        self.thresholds = param[key]
      elif key == 'nms_thresholds':
        self.nms_thresholds = param[key]

  def predict(self, image, bboxes):
    self.time.tic()

    width = image.shape[1]
    height = image.shape[0]
    numbox = bboxes.shape[0]
    if numbox == 0:
      return bboxes, []
    if self.only_points:
      original_bboxes = bboxes.copy()
    bboxes = rerec(bboxes)
    # third stage
    total_boxes = np.fix(bboxes)

    [ddy, edy, ddx, edx, yyy, eyy, xxx, ex, tmpw, tmph] =\
        pad(total_boxes, width, height)

    tempimg = np.zeros((numbox, 48, 48, 3), np.float32)
    for k in range(numbox):
      tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3), np.float32)
      tmp[int(ddy[k]):int(edy[k])+1, int(ddx[k]):int(edx[k])+1] =\
          image[int(yyy[k]):int(eyy[k])+1, int(xxx[k]):int(ex[k])+1]
      tempimg[k, :, :, :] = cv2.resize(tmp, (48, 48))
    # ONet
    tempimg = np.swapaxes(tempimg, 1, 3)
    input_data = {self.input_names[0]: tempimg}
    out = self.net.predict(input_data)
    if 'prob1' not in out:
      out['prob1'] = softmax(out['conv6-1'], 1, 1)
    assert out['prob1'].shape[0] == numbox
    score = out['prob1'][:, 1]
    points = out['conv6-3']
    pass_t = np.where(score > self.thresholds)[0]
    points = np.squeeze(points[pass_t, :])
    score = score[pass_t].reshape(-1, 1)
    total_boxes = np.concatenate((total_boxes[pass_t, 0:4], score), axis=1)

    if self.only_points:
      h_h = total_boxes[:, 3] - total_boxes[:, 1] + 1
      w_w = total_boxes[:, 2] - total_boxes[:, 0] + 1

      points = np.repeat([w_w, h_h], 5, axis=0).T * points
      points += np.repeat(np.stack([total_boxes[:, 0],\
          total_boxes[:, 1]]), 5, axis=0).T -1
      total_boxes = np.concatenate((original_bboxes[pass_t, 0:4], score),
                                   axis=1)
    else:
      mv_out = np.squeeze(out['conv6-2'][pass_t, :]).T

      idxc1 = np.where(total_boxes[:, 3] > image.shape[0])[0]  # ey>imgh
      idxc3 = np.where(total_boxes[:, 2] > image.shape[1])[0]  # ex>imgw
      idxc_union = np.union1d(idxc1, idxc3)

      h_h = total_boxes[:, 3] - total_boxes[:, 1] + 1
      w_w = total_boxes[:, 2] - total_boxes[:, 0] + 1

      w_1 = ex[pass_t] - xxx[pass_t] + 1
      h_1 = eyy[pass_t] - yyy[pass_t] + 1

      w_w[idxc_union] = w_1[idxc_union]
      h_h[idxc_union] = h_1[idxc_union]

      _total_boxes = total_boxes.copy()
      _total_boxes[idxc_union, 0] = xxx[pass_t[idxc_union]]
      _total_boxes[idxc_union, 1] = yyy[pass_t[idxc_union]]

      points = np.repeat([w_w, h_h], 5, axis=0).T * points
      points += np.repeat(
          np.stack([_total_boxes[:, 0], _total_boxes[:, 1]]), 5, axis=0).T - 1

      if total_boxes.shape[0] > 0:
        total_boxes = bbreg(total_boxes, mv_out[:, :])
        pick = nms(total_boxes, self.nms_thresholds, 'Min')
        if np.array(pick).shape[0] > 0:
          total_boxes = total_boxes[pick, :]
          points = points[pick, :]

    self.time.toc()
    return total_boxes, points


def generatebbox(in_map, reg, scale, threshold):
  """gen face bbox
  """
  stride = 2
  cellsize = 12
  in_map = in_map.T
  dx1 = reg[0, :, :].T
  dy1 = reg[1, :, :].T
  dx2 = reg[2, :, :].T
  dy2 = reg[3, :, :].T
  (x_x, y_y) = np.where(in_map >= threshold)

  yyy = y_y
  xxx = x_x

  score = in_map[x_x, y_y]
  reg = np.array([dx1[x_x, y_y], dy1[x_x, y_y], dx2[x_x, y_y], dy2[x_x, y_y]])

  if reg.shape[0] == 0:
    pass
  boundingbox = np.array([yyy, xxx]).T

  bb1 = np.fix((stride * (boundingbox) + 1) /
               scale).T  # matlab index from 1, so with "boundingbox-1"
  bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) /
               scale).T  # while python don't have to
  score = np.array([score])

  boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)
  return boundingbox_out.T


def rerec(bbox):
  """
  convert bbox to square
  """
  w_w = bbox[:, 2] - bbox[:, 0]
  h_h = bbox[:, 3] - bbox[:, 1]
  l_l = np.maximum(w_w, h_h).T
  bbox[:, :4] += np.array([w_w - l_l, h_h - l_l, l_l - w_w, l_l - h_h]).T * 0.5

  return bbox


def bbreg(boundingbox, reg):
  """face bbox regression
  """
  reg = reg.T

  # calibrate bouding boxes
  if reg.shape[1] == 1:
    print("reshape of reg")
    # pass  # reshape of reg
  w_w = boundingbox[:, 2] - boundingbox[:, 0] + 1
  h_h = boundingbox[:, 3] - boundingbox[:, 1] + 1

  boundingbox[:, 0:4] += reg * np.array([w_w, h_h, w_w, h_h]).T

  return boundingbox


def pad(boxes_in, w_in, h_in):
  """pad face bbox
  """
  boxes = boxes_in.copy()
  tmph = boxes[:, 3] - boxes[:, 1] + 1
  tmpw = boxes[:, 2] - boxes[:, 0] + 1
  numbox = boxes.shape[0]

  ddx = np.ones(numbox)
  ddy = np.ones(numbox)
  edx = tmpw
  edy = tmph

  xxx = boxes[:, 0:1][:, 0]
  yyy = boxes[:, 1:2][:, 0]
  ex = boxes[:, 2:3][:, 0]
  eey = boxes[:, 3:4][:, 0]

  tmp = np.where(ex > w_in)[0]
  if tmp.shape[0] != 0:
    edx[tmp] = -ex[tmp] + w_in - 1 + tmpw[tmp]
    ex[tmp] = w_in - 1

  tmp = np.where(eey > h_in)[0]
  if tmp.shape[0] != 0:
    edy[tmp] = -eey[tmp] + h_in - 1 + tmph[tmp]
    eey[tmp] = h_in - 1

  tmp = np.where(xxx < 1)[0]
  if tmp.shape[0] != 0:
    ddx[tmp] = 2 - xxx[tmp]
    xxx[tmp] = np.ones_like(xxx[tmp])

  tmp = np.where(yyy < 1)[0]
  if tmp.shape[0] != 0:
    ddy[tmp] = 2 - yyy[tmp]
    yyy[tmp] = np.ones_like(yyy[tmp])

  # for python index from 0, while matlab from 1
  ddy = np.maximum(0, ddy - 1)
  ddx = np.maximum(0, ddx - 1)
  yyy = np.maximum(0, yyy - 1)
  xxx = np.maximum(0, xxx - 1)
  edy = np.maximum(0, edy - 1)
  edx = np.maximum(0, edx - 1)
  eey = np.maximum(0, eey - 1)
  ex = np.maximum(0, ex - 1)

  return [ddy, edy, ddx, edx, yyy, eey, xxx, ex, tmpw, tmph]
