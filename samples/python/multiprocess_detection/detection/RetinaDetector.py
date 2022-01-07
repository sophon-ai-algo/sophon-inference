# -*- coding:utf-8 -*-
# @author  : cbingcan
# @time    : 2021/8/24/024 15:29

import cv2
import os
import time
import numpy as np
from threading import Thread
import queue
from PIL import Image, ImageDraw

import sophon.sail as sail

import threading


from detection.rcnn.processing.bbox_transform import clip_boxes
from detection.rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from detection.rcnn.processing.nms import gpu_nms_wrapper, cpu_nms_wrapper
from utils import *

class RetinaDetector():
    """
        从queue读取4张图片，不够4张等待
    """

    def __init__(self, engine, detect_queue, idx, queue_mutex, result_queue, result_mutex, predict_queue=None):
    # def __init__(self, engine, detect_queue, idx, queue_mutex, predict_queue=None):
        super().__init__()
        print(idx,'RetinaDetector init')
        self.engine = engine #检测引擎
        self.handle = self.engine.get_handle()
        self.net_out_queue = queue.Queue()
        self.bmcv = sail.Bmcv(self.handle)
        self.detect_queue = detect_queue
        self.queue_mutex = queue_mutex
        self.result_queue = result_queue
        self.result_mutex = result_mutex
        self.predict_queue = predict_queue
        self.graph_name = self.engine.get_graph_names()[0]
        self.input_name = self.engine.get_input_names(self.graph_name)[0]

        self.input_shape = self.engine.get_input_shape(self.graph_name, self.input_name)
        self.input_w = int(self.input_shape[-1])
        self.input_h = int(self.input_shape[-2])

        self.idx = idx
        self.anchors_list = {}
        self.gen_anchors()
        self.image_idx = 0
        self.batch_idx = 0
        print(idx,'RetinaDetector init end')
    
    def calc_im_scale_split(self, w, h):
        scales_w = float(w)/float(self.input_w)
        scales_h = float(h)/float(self.input_h)
        return scales_w,scales_h

    def post_process_run(self):
        while True:
            boxs_list = [[],[],[],[]]
            net_out = self.net_out_queue.get()
            if net_out is not None:
                #det_list = self.parse_net_out(net_out)
                detect_out_data = net_out['detection_out'][0][0]
                order = np.where(([int(i[1]) in [1] for i in detect_out_data[:]]))
                detect_out_data = detect_out_data[order]
                #print('[post_process_run] {}'.format(detect_out_data.shape))
                for out_data in detect_out_data:
                    if out_data[2] < 0.6:
                        continue
                    if int(out_data[1]) != 1:
                        continue
                    batch_idx = int(out_data[0])
                    im_scale_h = 6
                    im_scale_w = 6
                    pad_x = 0
                    pad_y = 0
                    x1 = (out_data[3] * self.input_w - pad_x) * im_scale_w
                    y1 = (out_data[4] * self.input_h - pad_y) * im_scale_h
                    x2 = (out_data[5] * self.input_w - pad_x) * im_scale_w
                    y2 = (out_data[6] * self.input_h - pad_y) * im_scale_h
                    score = out_data[2]
                    boxs_list[batch_idx].append([x1,y1,x2,y2,score])
            #print('[post_process_run] get box {}'.format(boxs_list))

    def cssd_post_process(self, net_out):
        boxs_list = [[], [], [], []]
        detect_out_data = net_out['detection_out'][0][0]
        order = np.where(([int(i[1]) in [1] for i in detect_out_data[:]]))
        detect_out_data = detect_out_data[order]
        # print('[post_process_run] {}'.format(detect_out_data.shape))
        for out_data in detect_out_data:
            if out_data[2] < 0.92:
                continue
            #if int(out_data[1]) != 1:
            #    continue
            batch_idx = int(out_data[0])
            im_scale_h = 6
            im_scale_w = 6
            pad_x = 0
            pad_y = 0
            x1 = (out_data[3] * self.input_w - pad_x) * im_scale_w 
            y1 = (out_data[4] * self.input_h - pad_y) * im_scale_h
            x2 = (out_data[5] * self.input_w - pad_x) * im_scale_w 
            y2 = (out_data[6] * self.input_h - pad_y) * im_scale_h
            score = out_data[2]
            boxs_list[batch_idx].append([x1, y1, x2, y2, score])
        return boxs_list


    def postprocess(self, im_tensors, output_tensors):
        """
        后处理
        """
        boxs_list = []
        try:
            #t = time.time()
            # print("推理开始：shape = " , im_tensors.shape)
            self.engine.process(self.graph_name, {self.input_name: im_tensors}, output_tensors)
            net_out = {}
            #t1 = time.time()
            #print('[postprocess] process time {}'.format((t1 - t) * 1000))
            for output_name, output_tensor in output_tensors.items():
                output_scale = 1.0#self.engine.get_output_scale(self.graph_name, output_name)
                out_net = output_tensor.scale_to(output_scale)
                net_out[output_name] = out_net

            t2 = time.time()

            det_list = []
            det_list = self.cssd_post_process(net_out)
            self.batch_idx += 1
            for det_box, bmimage, image_id, frame_number in zip(det_list, self.ost_frame_list,self.img_id_list,self.frame_number):
                self.image_idx += 1
                scale_w,scale_h = self.calc_im_scale_split(bmimage.width(), bmimage.height())
                for idx_temp in range(len(det_box)):
                    det_box[idx_temp][0]*=scale_w
                    det_box[idx_temp][1]*=scale_h
                    det_box[idx_temp][2]*=scale_w
                    det_box[idx_temp][3]*=scale_h
                # if len(det_box) > 0:
                #     for idx_temp, box in enumerate(det_box):
                #         self.bmcv.rectangle(bmimage, \
                #             int(box[0]/6), int(box[1]/6), \
                #                 int((box[2]-box[0])/6), int((box[3]-box[1])/6), (255, 0, 0), 3)
                #     self.bmcv.imwrite('/data/video/save_result/{}_result_{}.jpg'.format(image_id,t2), bmimage)
                self.result_mutex.acquire()
                if len(self.result_queue) > 40:
                    print("Result Queue Length more than 40")
                    self.result_queue.pop(0)
                self.result_queue.append({"id": image_id, "frame": bmimage, 'detection': det_box, "frame_number": frame_number})
                self.result_mutex.release()

        except Exception as e:
            # print("erro: {}".format(e.errno))
            print('error in postprocess:', e)
            pass
    
    def run(self, idx, num):
        #try:
        print(idx,'wxc_run_start')
        self.postprocess_t = Thread(target=self.post_process_run, args=())
        self.postprocess_t.start()
        input_shape = [4,3,self.input_h,self.input_w]
        input_dtype = self.engine.get_input_dtype(self.graph_name, self.input_name)
        input_scale = self.engine.get_input_scale(self.graph_name, self.input_name)
        img_dtype = self.bmcv.get_bm_image_data_format(input_dtype)
        output_name = self.engine.get_output_names(self.graph_name)
        #print('[Retina] output_tensors {}'.format(output_tensors))
        self.im_scale_list = []
        #scale = self.engine.get_input_scale(graph_name, input_name)
        scale = 1.0
        #ab = [x * scale for x in [1, 103.939, 1, 116.779, 1, 123.68]]
        ab = [x * input_scale for x in [1, 0, 1, 0, 1, 0]]
        #print('scale:', scale, 'ab:', ab)
        use_local_img=False
        input = sail.Tensor(self.handle, input_shape, input_dtype, False, False)


        while True:
            #try:
            self.img_list = []
            self.ost_frame_list = []
            self.img_id_list = []
            self.frame_number = []
            i = 0
            tmp_img = sail.BMImageArray4D()
            output = sail.BMImageArray4D(self.handle, input_shape[2], input_shape[3], \
                                 sail.Format.FORMAT_BGR_PLANAR, img_dtype)
            while True:
                if len(self.detect_queue) == 0:
                    time.sleep(0.02)
                    continue
                if use_local_img:
                    im_tensor = read_img_as_array('100.jpg', (800,450))
                else:
                    self.queue_mutex.acquire()


                    data = self.detect_queue.pop(0)
                    im_tensor =  data["frame"]
                    ost_frame_tensor =  data["ost_frame"]

                    # im_tensor_ost =  data["frame"]
                    # im_tensor = sail.BMImage(self.handle, input_shape[2], input_shape[3], sail.Format.FORMAT_BGR_PLANAR, sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE)
                    # self.bmcv.vpp_resize(im_tensor_ost, im_tensor, self.input_w, self.input_h)

                    im_id = data["id"]
                    frame_number = data["frame_number"]
                    self.queue_mutex.release()

                self.im_scale_list.append(data["im_scale"])
                # print("im_tensor_ost.format(): {}".format(ost_frame_tensor.format()))
                # print("    im_tensor.format(): {}".format(im_tensor.format()))
                self.ost_frame_list.append(ost_frame_tensor)
                self.img_list.append(im_tensor)
                self.frame_number.append(frame_number)


                # print("    im_tensor.format(): {}".format(im_tensor.format()))
                self.img_id_list.append(im_id)
                tmp_img[i] = im_tensor.data()

                i += 1
                if i > 3:
                    break
            self.bmcv.convert_to(tmp_img, output, ((ab[0], ab[1]),(ab[2], ab[3]),(ab[4], ab[5])))
            self.bmcv.bm_image_to_tensor(output, input)
            
            t1 = time.time()
            output_tensors = {}
            for a in output_name:
                output_dtype = self.engine.get_output_dtype(self.graph_name, a)
                output_shape = [1, 1, 400, 7]  
                output_tensor = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
                output_tensors[a] = output_tensor
            self.postprocess(input, output_tensors)
         
            num.value +=4.0


    def bbox_pred(self, boxes, box_deltas):
            """
        Transform the set of class-agnostic boxes into class-specific boxes
        by applying the predicted offsets (box_deltas)
        :param boxes: !important [N 4]
        :param box_deltas: [N, 4 * num_classes]
        :return: [N 4 * num_classes]
        """
            t = time.time()
            if boxes.shape[0] == 0:
                return np.zeros((0, box_deltas.shape[1]))

            boxes = boxes.astype(np.float, copy=False)
            widths = boxes[:, 2] - boxes[:, 0] + 1.0
            heights = boxes[:, 3] - boxes[:, 1] + 1.0
            ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
            ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

            dx = box_deltas[:, 0:1]
            dy = box_deltas[:, 1:2]
            dw = box_deltas[:, 2:3]
            dh = box_deltas[:, 3:4]

            pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
            pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
            pred_w = np.exp(dw) * widths[:, np.newaxis]
            pred_h = np.exp(dh) * heights[:, np.newaxis]

            pred_boxes = np.zeros(box_deltas.shape)
            #print('[bbox_pred] 1 {}'.format((time.time() - t) * 1000))
            t = time.time()
            # x1
            pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
            # y1
            pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
            # x2
            pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
            # y2
            pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

            if box_deltas.shape[1] > 4:
                pred_boxes[:, 4:] = box_deltas[:, 4:]
            #print('[bbox_pred] {}'.format(pred_boxes))
            return pred_boxes

    def gen_anchors(self):
        ctx_id = 0
        self.nms_threshold = 0.4
        fpn_keys = []

        _ratio = (1.,)
        self._feat_stride_fpn = [32, 16, 8]
        height_list = [14, 28, 56]
        width_list = [25, 50, 100]

        anchor_cfg = {
            '32': {
                'SCALES': (32, 16),
                'BASE_SIZE': 16,
                'RATIOS': _ratio,
                'ALLOWED_BORDER': 9999
            },
            '16': {
                'SCALES': (8, 4),
                'BASE_SIZE': 16,
                'RATIOS': _ratio,
                'ALLOWED_BORDER': 9999
            },
            '8': {
                'SCALES': (2, 1),
                'BASE_SIZE': 16,
                'RATIOS': _ratio,
                'ALLOWED_BORDER': 9999
            }
        }

        for s in self._feat_stride_fpn:
            fpn_keys.append('stride%s' % s)

        dense_anchor = False

        _anchors_fpn = dict(
            zip(
                fpn_keys,
                generate_anchors_fpn(dense_anchor=dense_anchor,
                                     cfg=anchor_cfg)))
        for k in _anchors_fpn:
            v = _anchors_fpn[k].astype(np.float32)
            _anchors_fpn[k] = v

        self._num_anchors = dict(
            zip(fpn_keys,
                [anchors.shape[0] for anchors in _anchors_fpn.values()]))

        self.nms = gpu_nms_wrapper(self.nms_threshold, ctx_id)

        for _idx, s in enumerate(self._feat_stride_fpn):
            stride = int(s)
            height = height_list[_idx]
            width = width_list[_idx]
            A = self._num_anchors['stride%s' % s]
            K = height * width
            anchors_fpn = _anchors_fpn['stride%s' % s]
            anchors = anchors_plane(height, width, stride, anchors_fpn)
            anchors = anchors.reshape((K * A, 4))
            self.anchors_list[s] = anchors
        #print('[gen_anchors] {}'.format(self.anchors_list))

    def parse_net_out(self, net_out):
        #ctx_id = 0
        decay4 = 0.5
        #nms_threshold = 0.4
        #vote = False
        #nocrop = False
        #fpn_keys = []
        #anchor_cfg = None

        #preprocess = False
        _ratio = (1., )
        #_feat_stride_fpn = [32, 16, 8]
        im_info = [self.input_h, self.input_w]
        #im_scale = 0.416666
        '''
        anchor_cfg = {
            '32': {
                'SCALES': (32, 16),
                'BASE_SIZE': 16,
                'RATIOS': _ratio,
                'ALLOWED_BORDER': 9999
            },
            '16': {
                'SCALES': (8, 4),
                'BASE_SIZE': 16,
                'RATIOS': _ratio,
                'ALLOWED_BORDER': 9999
            },
            '8': {
                'SCALES': (2, 1),
                'BASE_SIZE': 16,
                'RATIOS': _ratio,
                'ALLOWED_BORDER': 9999
            }
        }
        '''
        cascade = 0

        bbox_stds = [1.0, 1.0, 1.0, 1.0]

        det_list = []
        proposals_list = []
        scores_list = []
        strides_list = []
        threshold=0.5
        sym_idx = 0

        bt = time.time()
        for c_idx in range(4):
            im_scale = self.im_scale_list[c_idx]
            for _idx, s in enumerate(self._feat_stride_fpn):
                _key = 'stride%s' % s
                stride = int(s)
                is_cascade = False
                if cascade:
                    is_cascade = True

                #scores = np.expand_dims(net_out['face_rpn_cls_prob_reshape_' + _key + '_output'][c_idx,:,:,:], axis=0)
                scores = np.expand_dims(net_out['rpn_cls_prob_reshape_' + _key + '_output'][c_idx,:,:,:], axis=0)
                scores = scores[:, self._num_anchors['stride%s' % s]:, :, :]
                #bbox_deltas = np.expand_dims(net_out['face_rpn_bbox_pred_' + _key + '_output'][c_idx,:,:,:], axis=0)

                #height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

                A = self._num_anchors['stride%s' % s]
                #K = height * width
                #anchors_fpn = _anchors_fpn['stride%s' % s]
                #anchors = anchors_plane(height, width, stride, anchors_fpn)
                #anchors = anchors.reshape((K * A, 4))
                #print('[parse_net_out] anchors {}'.format(anchors))

                scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
                if stride == 4 and decay4 < 1.0:
                    scores *= decay4
                scores_ravel = scores.ravel()
                order = np.where(scores_ravel >= threshold)[0]
                #print('[parse_net_out] order {}'.format(order))
                scores = scores[order]
                if len(scores) == 0:
                    #scores_list.append([])
                    #proposals_list.append([])
                    continue
                bbox_deltas = np.expand_dims(net_out['rpn_bbox_pred_' + _key + '_output'][c_idx, :, :, :], axis=0)
                anchors = self.anchors_list[s][order, :]
                bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
                bbox_pred_len = bbox_deltas.shape[3] // A
                bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
                bbox_deltas = bbox_deltas[order, :]
                bbox_deltas[:, 0::4] = bbox_deltas[:, 0::4] * bbox_stds[0]
                bbox_deltas[:, 1::4] = bbox_deltas[:, 1::4] * bbox_stds[1]
                bbox_deltas[:, 2::4] = bbox_deltas[:, 2::4] * bbox_stds[2]
                bbox_deltas[:, 3::4] = bbox_deltas[:, 3::4] * bbox_stds[3]
                proposals = self.bbox_pred(anchors, bbox_deltas)
                proposals = clip_boxes(proposals, im_info)

                #proposals = proposals[order, :]

                proposals[:, 0:4] /= im_scale
                proposals_list.append(proposals)
                scores_list.append(scores)
                if self.nms_threshold < 0.0:
                    _strides = np.empty(shape=(scores.shape),
                                        dtype=np.float32)
                    _strides.fill(stride)
                    strides_list.append(_strides)

                sym_idx += 2
            if len(proposals_list) == 0:
                det_list.append([])
                continue
            proposals = np.vstack(proposals_list)
            scores = np.vstack(scores_list)
            scores_ravel = scores.ravel()
            order = scores_ravel.argsort()[::-1]
            proposals = proposals[order, :]
            scores = scores[order]
            pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
            t0 = time.time()
            keep = self.nms(pre_det)
            #print('c_idx:{} nms {} ms'.format(c_idx, (time.time()-t0)*1000))
            det = np.hstack((pre_det, proposals[:, 4:]))
            det = det[keep, :]
            det_list.append(det)
        #print('total cost {} ms'.format((time.time()-bt)*1000))
        return det_list
