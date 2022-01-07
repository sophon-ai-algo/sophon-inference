# -*- coding:utf-8 -*-
# @author  : cbingcan
# @time    : 2021/8/24/024 11:17
import cv2
import time
from multiprocessing import Process, Value
import sophon.sail as sail
from utils import DETECTION_TPU_ID
import threading
import pickle
import os
import sys, errno
from utils import *
from PIL import Image
import numpy as np

class Capture():

    """
     1、读取视频流进程，一个进程一个处理一个rtsp视频流。
     2、将读取到的视频帧写入detect_q
    """

    def __init__(self, threadID, name, engine, data, detect_queue):
        super().__init__()
        self.threadID = threadID
        self.name = name
        self.data = data
        self.detect_queue = detect_queue
        self.engine = engine
        self.handle = engine.get_handle()
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = self.engine.get_graph_names()[0]
        self.input_name = self.engine.get_input_names(self.graph_name)[0]

    def process(self, im_tensors):
        # 推理
        t = time.time()
        print("input_name = %s" % self.input_name)
        #print("推理开始：shape = " , im_tensors.shape)
        net_out = self.engine.process(self.graph_name, {self.input_name: im_tensors})
        # print("output shape: %s" % net_out.shape)
        print('total cost {} ms'.format((time.time() - t) * 1000))

    def run(self, exit, num):
       readImg = False
       graph_name = self.engine.get_graph_names()[0]
       input_name = self.engine.get_input_names(graph_name)[0]
       scale = self.engine.get_input_scale(graph_name, input_name)
       input_dtype = self.engine.get_input_dtype(graph_name, input_name)
       if not readImg:
         decoder = sail.Decoder(self.data['rtsp'], True, DETECTION_TPU_ID)
       input_shape = [1,3,input_img_h,input_img_w]
       img_dtype = self.bmcv.get_bm_image_data_format(input_dtype)
       output_name = self.engine.get_output_names(graph_name)[0]
       output_dtype = self.engine.get_output_dtype(graph_name, output_name)
       output_shape = [1, 3, 450, 800]
       input = sail.Tensor(self.handle, output_shape, input_dtype, False, False)
       tmp = sail.BMImage(self.handle,input_shape[2], input_shape[3], sail.Format.FORMAT_BGR_PLANAR, img_dtype)
       while True:
           t = time.time()
           if not readImg:
            img = sail.BMImage()
            ret = decoder.read(self.handle, img)
            #print('视频解码用时 {} ms'.format((time.time() - t) * 1000))
            if ret != 0:
                    continue

           if (self.detect_queue.qsize() > 500):
            if (exit.value == 1.0):
                print("cap exit: ", exit.value)
                break
            time.sleep(0.01)
            continue    
           
           if not readImg:
            #print('wxc w',input_img_w,' h:', input_img_h)
            tmp = self.bmcv.vpp_resize(img, input_img_w, input_img_h)
            self.bmcv.imwrite("tmp.jpg", tmp)
            self.bmcv.bm_image_to_tensor(tmp, input)
           else:
            img =  ('100.jpg', (800,450))
           
           if (exit.value == 1.0):# or num.value > 10000:
            print(" cap exit2: ", exit.value)
            break
           try:
            if not readImg:
              #input.sync_d2s()
              input_img = self.bmcv.tensor_to_bm_image(input)
              self.bmcv.imwrite("input_img.jpg", input_img)
              input_array=input.asnumpy().astype(np.int8)
              print("fps: ", decoder.get_fps())
              print('cap img:',input_array[0,:,449,797:])
              self.detect_queue.put({"id": self.data["id"], "frame":input_array})
              #input.unmap()
            else:
              print('cap img:',img[0,:,1,0:3])
              self.detect_queue.put({"id": self.data["id"], "frame": img})
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

