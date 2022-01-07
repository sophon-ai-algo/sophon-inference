# -*- coding:utf-8 -*-
# @author  : cbingcan
# @time    : 2021/8/24/024 15:29

import cv2
import os
import time
import numpy as np
from threading import Thread
import queue

import sophon.sail as sail

import threading

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

class YOLOX():
    """
        从queue读取4张图片，不够4张等待
    """

    def __init__(self, engine, detect_queue, idx, queue_mutex, result_queue, result_mutex, predict_queue=None):
    # def __init__(self, engine, detect_queue, idx, queue_mutex, predict_queue=None):
        super().__init__()
        print(idx,'YOLOX init')
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

        self.output_name = self.engine.get_output_names(self.graph_name)[0]

        self.output_dtype = self.engine.get_output_dtype(self.graph_name, self.output_name)
        self.output_shape = self.engine.get_output_shape(self.graph_name, self.output_name)
        self.output = sail.Tensor(self.handle,self.output_shape,self.output_dtype,True,True)
        self.output_tensors = {self.output_name:self.output}

        self.idx = idx
        self.image_idx = 0
        self.batch_idx = 0
        mkdir("/data/video/save_result/")
        print(idx,'YOLX init end')
    
    def calc_im_scale_split(self, w, h):
        scales_w = float(w)/float(self.input_w)
        scales_h = float(h)/float(self.input_h)
        return scales_w,scales_h

    def get_batchsize(self):
        return int(self.input_shape[0])

    def get_detectresult(self,predictions,dete_threshold,nms_threshold):
        # print(predictions.shape)
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        # print("boxes")
        # print(boxes.shape)
        # print(boxes)
        # print("scores")
        # print(scores.shape)
        # print(scores)

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_threshold, score_thr=dete_threshold)
        return dets

    def yolox_postprocess(self, outputs, input_w, input_h, p6=False):
        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [input_h // stride for stride in strides]
        wsizes = [input_w // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

    def postprocess(self, im_tensors, output_tensors):
        """
        后处理
        """
        boxs_list = []
        try:
            #t = time.time()
            # print("推理开始：shape = " , im_tensors.shape)
            self.engine.process(self.graph_name, {self.input_name: im_tensors}, output_tensors)
            out_temp = output_tensors[self.output_name].asnumpy()
            predictions = self.yolox_postprocess(out_temp, self.input_w, self.input_w)
            t2 = time.time()
            det_list = []

            for image_idx, image_ost in enumerate(self.ost_frame_list):
                dete_boxs_temp = []
                dete_boxs = self.get_detectresult(predictions[image_idx],0.25, 0.45)
                ratio_w,ratio_h = self.calc_im_scale_split(image_ost.width(), image_ost.height())
                self.image_idx += 1

                if dete_boxs is not None:
                    dete_boxs[:,0] *= ratio_w
                    dete_boxs[:,1] *= ratio_h
                    dete_boxs[:,2] *= ratio_w
                    dete_boxs[:,3] *= ratio_h

                for dete_box in dete_boxs:
                    dete_boxs_temp.append(dete_box)
                    # self.bmcv.rectangle(image_ost, int(dete_box[0]), int(dete_box[1]), 
                    #     int(dete_box[2]-dete_box[0]), int(dete_box[3]-dete_box[1]), (255, 0, 0), 3)
                    # self.bmcv.imwrite('/data/video/save_result/loop{}_result_{}.jpg'.format(self.image_idx,image_idx), image_ost)

                det_list.append(dete_boxs_temp)

            self.batch_idx += 1
            for det_box, bmimage, image_id, frame_number in zip(det_list, self.ost_frame_list,self.img_id_list,self.frame_number):
                self.image_idx += 1
                for idx_temp, box in enumerate(det_box):
                    self.bmcv.rectangle(bmimage, \
                        int(box[0]), int(box[1]), \
                            int((box[2]-box[0])), int((box[3]-box[1])), (255, 0, 0), 3)
                # self.bmcv.imwrite('/data/video/save_result/{}_result_{}.jpg'.format(image_id,t2), bmimage)
               
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
            self.postprocess(input,  self.output_tensors)
         
            num.value +=4.0

