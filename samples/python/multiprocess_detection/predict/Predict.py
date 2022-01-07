# -*- coding:utf-8 -*-
# @author  : cbingcan
# @time    : 2021/8/24/024 15:49

import time
import numpy as np
import multiprocessing
from multiprocessing import Manager, Queue, Value
from multiprocessing import Process

from sklearn import preprocessing

import sophon.sail as sail
import signal
import os

class Predict(object):

    def __init__(self, engine, predict_queue):
        super().__init__()
        self.engine = engine
        self.queue = predict_queue

    def stop(self):
        print("captute process stop")
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)

    def get_feature(self, image):
        image = np.transpose(image, (2, 0, 1)).astype(np.float32).copy()
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        if self.feature_in_dtype == sail.BM_FLOAT32:
            self.feature_input_tensors[self.feature_input_name].update_data(image)
        else:
            scale = self.feature_engine.get_input_scale(self.feature_graph_name, self.feature_input_name)
            self.feature_input_tensors[self.feature_input_name].scale_from(image, scale)

        start_time = time.time()
        # inference
        self.feature_engine.process(self.feature_graph_name, self.feature_input_tensors, self.feature_ouptut_tensors)
        # scale output data if output data type is int8 or uint8
        if self.feature_out_dtype == sail.BM_FLOAT32:
            output = self.feature_output.asnumpy()
        else:
            scale = self.feature_engine.get_output_scale(self.feature_graph_name, self.feature_output_name)
            output = self.feature_output.scale_to(scale)
        print('推理时间:{}'.format(time.time() - start_time))
        output = preprocessing.normalize(output)
        return output

    def run(self, exit):
        while True:
            if (exit.value == 1.0):
                print("pred exit")
                break
            img = self.queue.get(False)
            feature = self.get_feature(img)
            # time.sleep(0.02)
