# -*- coding:utf-8 -*-
# @author  : cbingcan
# @time    : 2021/8/24/024 9:53

from threading import Thread

import sophon.sail as sail

from capture.Capture import Capture
from detection.RetinaDetector import RetinaDetector
import time
import signal
import os

import threading

from threading import Thread
from upload import UploadProcess

class Pack():
    def __init__(self, url_list, detect_engine, idx, num, upload_flage=False):
        self.data_mutex = threading.Lock()
        self.url_list = url_list
        self.idx = idx
        self.detect_engine = detect_engine
        self.cap_thread_list = []
        self.cap_data_list = []
        self.detection_result_list= []
        self.dete_resu_mutex = threading.Lock()
        for url in url_list:
            cap = Capture(url['id'], url["id"], detect_engine, url, self.cap_data_list, self.data_mutex)          #cap_data_list是否需要互斥锁
            # cap = Capture(url['id'], url["id"], detect_engine, url, self.cap_data_list)          #cap_data_list是否需要互斥锁
            cap_thread = Thread(target=cap.run)
            self.cap_thread_list.append(cap_thread)


        up_handle = UploadProcess()
        up_handle.set_basic_info(detect_engine,self.detection_result_list,self.dete_resu_mutex,upload_flage)
        # up_handle.set_basic_info(detect_engine,self.detection_result_queue,self.dete_resu_mutex)
        self.upload_thread = Thread(target=up_handle.run)

        det = RetinaDetector(self.detect_engine, self.cap_data_list, self.idx, self.data_mutex,self.detection_result_list, self.dete_resu_mutex)
        # det = RetinaDetector(self.detect_engine, self.cap_data_list, self.idx)
        self.detect_thread = Thread(target=det.run, args=(self.idx, num))
        print('[Pack] init self.cap_thread_list {}*****************{}'.format(self.cap_thread_list,os.getpid()))

    def run(self):
        self.detect_thread.start()
        self.upload_thread.start()
        for cap_thread in self.cap_thread_list:
            #print('[Pack] run {} {}'.format(type(cap_thread), cap_thread))
            cap_thread.start()

        for cap_thread in self.cap_thread_list:
            cap_thread.join()
        self.detect_thread.join()
        self.upload_thread.join()
