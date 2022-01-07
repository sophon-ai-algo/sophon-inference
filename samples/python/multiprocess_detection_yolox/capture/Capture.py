# -*- coding:utf-8 -*-
# @author  : cbingcan
# @time    : 2021/8/24/024 11:17
import cv2
import time
import sophon.sail as sail
from utils import DETECTION_TPU_ID
import threading
import pickle
import os
import sys, errno
from utils import *
import numpy as np
import threading


class Capture():

    """
     1、读取视频流进程，一个进程一个处理一个rtsp视频流。
     2、将读取到的视频帧写入detect_q
    """

    def __init__(self, threadID, name, engine, data, detect_queue,queue_mutex):
        super().__init__()
        self.threadID = threadID
        self.name = name
        self.data = data
        self.detect_queue = detect_queue
        self.queue_mutex = queue_mutex
        self.engine = engine
        self.handle = engine.get_handle()
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = self.engine.get_graph_names()[0]
        self.input_name = self.engine.get_input_names(self.graph_name)[0]

        self.input_shape = self.engine.get_input_shape(self.graph_name, self.input_name)
        self.input_w = int(self.input_shape[-1])
        self.input_h = int(self.input_shape[-2])
        print("self.input_w: %d,self.input_h: %d"%(self.input_w,self.input_h))

        print("Thread:{},Video Name:{}".format(threadID,name))

    def process(self, im_tensors):
        # 推理
        t = time.time()
        #print("input_name = %s" % self.input_name)
        #print("推理开始：shape = " , im_tensors.shape)
        net_out = self.engine.process(self.graph_name, {self.input_name: im_tensors})
        # print("output shape: %s" % net_out.shape)
        #print('total cost {} ms'.format((time.time() - t) * 1000))

    def calc_im_scale(self, w, h):
        # print('calc_im_scale:',w,h)
        scales = [self.input_h, self.input_w]
        im_shape = np.array([w, h])
        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        return im_scale

    def run(self):
        print("Capture Thread {} Start Run.......".format(self.threadID))
        graph_name = self.engine.get_graph_names()[0]
        input_name = self.engine.get_input_names(graph_name)[0]
        #scale = self.engine.get_input_scale(graph_name, input_name)
        input_dtype = self.engine.get_input_dtype(graph_name, input_name)
        decoder = sail.Decoder(self.data['rtsp'], True, DETECTION_TPU_ID)
        input_shape = [4,3,self.input_h,self.input_w]
        img_dtype = self.bmcv.get_bm_image_data_format(input_dtype)
        output_name = self.engine.get_output_names(graph_name)[0]
        count = 0
        frame_number = 0
        while True:
            t = time.time()
            img = sail.BMImage()
            ret = decoder.read(self.handle, img)
            # print("    img.format(): {}".format(img.format()))
            count += 1
            if ret != 0 or count < 6:
                time.sleep(0.01)
                continue
            tmp = sail.BMImage(self.handle, input_shape[2], input_shape[3], sail.Format.FORMAT_BGR_PLANAR, sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE)
            # ost = sail.BMImage(self.handle, img.height(), img.width(), sail.Format.FORMAT_BGR_PLANAR, sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE)
            self.bmcv.vpp_resize(img, tmp, self.input_w, self.input_h)
            # self.bmcv.vpp_resize(img, ost, img.width(), img.height())

            try:
                self.queue_mutex.acquire()
                count = 0
                if len(self.detect_queue) > 20:
                    self.detect_queue.pop(0)
                #im_scale = self.calc_im_scale(img.width(), img.height())
                im_scale = 1
                frame_number += 1
                self.detect_queue.append({"id": self.data["id"], "frame": tmp, "ost_frame":img, 'im_scale': im_scale, 'frame_number': frame_number})
                self.queue_mutex.release()
                #print('[Cap] queue size {} {}'.format(len(self.detect_queue), (time.time() - t)*1000))
            except IOError as e:
                if e.errno == errno.EPIPE:
                    print("queue full, discard it")
            except BrokenPipeError as k:
                    print("pipeerror: {}".format(k.errno))
                    pass
            except EOFError as l:
                print("EOF error: {}".format(l.errno))
                pass
            except TimeoutError as e:
                print("Timeout :", e)
                pass
            except Exception as e:
                print(e)
                pass
            #print("pid: {} , qsize: {}".format(os.getpid(), self.detect_queue.qsize()))
            #print(pickle.dumps(img))
            #print(pickle.dumps(img.data()))
            #self.detect_queue.put({"id": self.data["id"], "frame":img})
            #print('视频解码用时 {} ms'.format((time.time() - t) * 1000))
            #time.sleep(0.5)

