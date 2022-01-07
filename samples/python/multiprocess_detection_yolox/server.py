# -*- coding:utf-8 -*-

import sys
from flask import Flask
from multiprocessing import Queue
from multiprocessing import Manager, Value

from multiprocessing import Process

import sophon.sail as sail

from pack import Pack
from utils import *
import time
import signal
import threading

import cv2
import os
import multiprocessing as mp

from websocket.utils import logger
from websocket.pull_frame import PullFrameProcess
from websocket.redis_websocket_srv import WebsocketProcess

log = logger.get_logger(__file__)



global num
num = Value('d', 0.0)

global exit
exit = Value('d', 0.0)

global global_start_t0

# 任务列表
service_list = []

def service_start():
    for service in service_list:
        service.start()

def service_join():
    for service in service_list:
        service.join()


def exit_handler(signum, frame):
    os._exit(0)

def term_sig_handler(signum, frame):
    print("catched signal: ", signum)
    global exit
    exit.value = 1.0

def init_pack(detect_engine, num):
    pack_pocess_list = []
    if len(VIDEO_LIST) == 0:
        print('[init_pack] no video error return')
        return
    pack_num = int((len(VIDEO_LIST)-1)/4) + 1
    print('[init_pack] pack_num {}'.format(pack_num))
    upload_flage = True
    for i in range(pack_num):
        print('[init_pack] start {}'.format(i))
        start = i*4
        end = start + 4
        if end > len(VIDEO_LIST):
            handle_url = VIDEO_LIST[start:]
        else:
            handle_url = VIDEO_LIST[start: end]
        print('[init_pack] handle url {}'.format(handle_url))

        pack = Pack(handle_url, detect_engine, i, num, upload_flage)
        upload_flage = False
        pack_process = Process(target=pack.run, args=())
        pack_pocess_list.append(pack_process)
    return pack_pocess_list

if __name__ == '__main__':

    """
    1、多进程获取视频流 写入queue
    2、多进程消费queue调用retinaface检测人脸、sort人脸跟踪、写入queue
    3、单进程从queue消费人脸图片进行人脸识别。
    4、将人脸识别结果上传到云端。
    """
    manager = Manager()

    detect_engine = sail.Engine(DETECTION_MODEL_PATH, DETECTION_TPU_ID, sail.SYSO) #检测引擎

    signal.signal(signal.SIGTERM, term_sig_handler)

    process_list = init_pack(detect_engine, num)


    web_srv = WebsocketProcess()
    web_srv.setport(WEBSOCKET_PORT)
    web_srv.daemon = True
    service_list.append(web_srv)

    # pull_srv = PullFrameProcess()
    # pull_srv.daemon = True
    # service_list.append(pull_srv)

    # 任务开启
    service_start()
 

    for i in range(len(process_list)):
        process_list[i].start()

    #global t
    t = time.time()
    while True:
        time.sleep(10)
        if num.value > 0 :
            print('detection per fps {} ms'.format((time.time() - t) * 1000/num.value))
        t = time.time()
        num.value = 0.0

    # 预留flask，后期增加视频流配置接口
    # app.run(host='0.0.0.0', port=SERVER_PORT, threaded=True)
    service_join()

    sys.exit(0)
