import cv2
import os
import time
from multiprocessing import Process
import base64
import numpy as np

from websocket.utils.redis_client import RedisClientInstance
from websocket.utils import logger

log = logger.get_logger(__file__)


# 摄像机流地址
# rtsp://admin:a12345678@192.168.3.113:554/Streaming/Channels/101

# RTSP_URL = "rtsp://admin:a12345678@192.168.3.113:554/Streaming/Channels/101"
RTSP_URL = "/data/video/zhuheqiao.mp4"

class PullFrameProcess(Process):

    def __init__(self):
        super(PullFrameProcess, self).__init__()
        self.url = RTSP_URL
        self.exit_flag = 0
        self.redis_client = None

    def __del__(self):
        pass

    def run(self):
        # redis连接
        self.store = RedisClientInstance.get_storage_instance()

        log.info("start pull rtsp")
        vc = cv2.VideoCapture(self.url)
        while self.exit_flag == 0:
            try:

                ret, frame_ost = vc.read()
                frame = cv2.resize(frame_ost,(352,288))
                _, image = cv2.imencode('.jpg', frame)
                image_id = os.urandom(4)
                self.store.single_set_string('image', np.array(image).tobytes())
                self.store.single_set_string('image_id', image_id)
                # log.info("store image id {}".format(image_id))
            except Exception as e:
                log.error("read frame error {}".format(e))
            # cv2.waitKey(1)
        vc.release()

    def stop(self):
        self.exit_flag = 1



