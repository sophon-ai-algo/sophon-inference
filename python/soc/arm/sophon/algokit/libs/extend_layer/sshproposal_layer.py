"""The implementation of ssh proposal layer
"""
from __future__ import division
import numpy as np
from .base_layer import BaseLayer
from ...utils.box_operation import bbox_transform_inv


class SSHProposalLayer(BaseLayer):
  """Output detected bbox

  Outputs object detection proposals by applying estimated bounding-box
  transformations to a set of regular boxes (called "anchors").
  """

  def __init__(self, layer_params, global_cfg):
    super(SSHProposalLayer, self).__init__()
    self._feat_stride = layer_params['feat_stride']
    self._anchor_ratios = layer_params.get('ratios',
                                           (0.5, 1, 2))  # default anchor ratio
    self._anchor_scales = layer_params.get('scales',
                                           (8, 16, 32))  # default anchor scale
    self._base_size = layer_params.get('base_size', 16)  # default base_size
    self._anchors = generate_anchors(base_size=self._base_size,\
        ratios=np.array(self._anchor_ratios), \
        scales=np.array(self._anchor_scales))
    # print("anchors: \n {}".format(self._anchors))
    self._num_anchors = self._anchors.shape[0]
    self._pre_nms_topn = global_cfg.get('pre_nms_topN', 200)
    self._min_size = global_cfg.get('anchor_min_size', 0)

  def forward(self, inputs):
    """SSHProposalLayer Algorithm:

    for each (H, W) location i
    generate A anchor boxes centered on cell i
    apply predicted bbox deltas at cell i to each of the A anchors
    clip predicted boxes to image
    remove predicted boxes with either height or width < threshold
    sort all (proposal, score) pairs by score from highest to lowest
    take top pre_nms_topN proposals before NMS
    apply NMS with threshold 0.7 to remaining proposals
    take after_nms_topN proposals after NMS
    return the top proposals (-> RoIs top, scores top)
    """
    # the first set of _num_anchors channels are background probs get second set
    # scores original shape (n, 2*num_anchors, h, w)
    scores = inputs[0][:, self._num_anchors:, :, :]
    bbox_deltas = inputs[1]  # shape (n, num_anchors*4, h, w)
    im_info = inputs[2][0, :]  # (1,3) -> h w scale

    # generate proposals from bbox deltas and shifted anchors
    height, width = scores.shape[-2:]
    # enumerate all shifts
    shift_x = np.arange(0, width) * self._feat_stride
    shift_y = np.arange(0, height) * self._feat_stride
    shift_x, shift_y = np.meshgrid(
        shift_x, shift_y)  # shape (len(shift_y), len(shift_x))
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                        shift_y.ravel())).transpose()  # shape (width*height, 4)

    # enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    a_n = self._num_anchors
    k_n = shifts.shape[0]
    anchors = self._anchors.reshape((1, a_n, 4)) \
        + shifts.reshape((1, k_n, 4)).transpose((1, 0, 2)) # (K, A, 4)
    anchors = anchors.reshape((k_n * a_n, 4))

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

    # Same operation for the scores:
    #
    # scores are (1, A, H, W) format
    # transpose to (1, H, W, A)
    # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

    # Convert anchors into proposals via bbox transformations
    proposals = bbox_transform_inv(anchors, bbox_deltas)

    # clip predicted boxes to image
    proposals = clip_boxes(proposals, im_info[:2])

    # remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    keep = filter_boxes(proposals, self._min_size * im_info[2])
    proposals = proposals[keep, :]
    scores = scores[keep]

    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN
    order = scores.ravel().argsort()[::-1]
    if self._pre_nms_topn > 0:
      order = order[:self._pre_nms_topn]
    proposals = proposals[order, :]
    scores = scores[order]

    # return proposals and scores
    # this proposal layer only supports a single input image batch idx = 0
    if proposals.shape[0] == 0:
      # print('this level no proposals')
      return np.array([[0, 0, 0, 16, 16]],
                      dtype=np.float32), np.array([[0]], dtype=np.float32)

    batch_idx = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    proposals = np.hstack(
        (batch_idx, proposals.astype(np.float32, copy=False)))
    return proposals, scores


def generate_anchors(base_size=16,
                     ratios=np.array([0.5, 1, 2]),
                     scales=2**np.arange(3, 6)):
  """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.

  Args:
      base_size: int. anchor base size
      ratios: List. anchor ratios
      scales: np.ndarray. anchor different scales
  Returns:
      anchors array
  """
  base_anchor = np.array([1, 1, base_size, base_size
                         ]) - 1  # default (0, 0, 15, 15)
  ratio_anchors = ratio_enum(base_anchor, ratios)  # (len(ratios), 4)
  anchors = np.vstack([scale_enum(ratio_anchors[i, :], scales)\
      for i in range(ratio_anchors.shape[0])]) # (len(scales)*len(ratios), 4)
  return anchors


def ratio_enum(anchor, ratios):
  """Enumerate a set of anchors for each aspect ratio wrt an anchor
  """
  base_w, base_h, x_ctr, y_ctr = genwhctrs(anchor)
  size = base_w * base_h  # 16 * 16
  size_ratios = size / ratios  # (2*16**2, 1*16**2, 0.5*16**2)
  w_s = np.round(np.sqrt(size_ratios))
  h_s = np.round(w_s * ratios)
  anchors = makeanchors(w_s, h_s, x_ctr, y_ctr)  # (len(ratios), 4)
  return anchors


def genwhctrs(anchor):
  """Return width, height, x center, and y center for an anchor (window).
  """
  base_w = anchor[2] - anchor[0] + 1  # 15 + 1
  base_h = anchor[3] - anchor[1] + 1
  x_ctr = anchor[0] + 0.5 * (base_w - 1)
  y_ctr = anchor[1] + 0.5 * (base_h - 1)
  return base_w, base_h, x_ctr, y_ctr


def makeanchors(a_w, a_h, x_ctr, y_ctr):
  """calculation anchors from a_w, a_h, x_ctr, y_ctr
    Given a vector of widths (a_w) and heights (a_h) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
  """
  a_w = a_w[:, np.newaxis]  # shape (n,) -> (n, 1)
  a_h = a_h[:, np.newaxis]
  anchors = np.hstack(
      (x_ctr - 0.5 * (a_w - 1), y_ctr - 0.5 * (a_h - 1),
       x_ctr + 0.5 * (a_w - 1),
       y_ctr + 0.5 * (a_h - 1)))  # (n, 4)
  return anchors


def scale_enum(anchor, scales):
  """Enumerate a set of anchors for each scale wrt an anchor.
  """
  a_w, a_h, x_ctr, y_ctr = genwhctrs(anchor)
  w_s = a_w * scales
  h_s = a_h * scales
  anchors = makeanchors(w_s, h_s, x_ctr, y_ctr)
  return anchors


def clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries.
  """
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
  return boxes


def filter_boxes(boxes, min_size):
  """Remove all boxes with any side smaller than min_size.
    """
  w_s = boxes[:, 2] - boxes[:, 0] + 1
  h_s = boxes[:, 3] - boxes[:, 1] + 1
  keep = np.where((w_s >= min_size) & (h_s >= min_size))[0]
  return keep
