import sophon.sail as sail

from capture.Capture import Capture
from detection.RetinaDetector import RetinaDetector
import time
import signal
import os
import cv2
import numpy as np

import threading

import cv2
import os
import time
from multiprocessing import Process
import base64
import numpy as np

from websocket.utils.redis_client import RedisClientInstance
from websocket.utils import logger

log = logger.get_logger(__file__)

class UploadProcess(Process):
    def __init__(self):
        super(UploadProcess, self).__init__()
        self.exit_flag = 0
        self.redis_client = None
        self.upload_flage = False
        self.time_start = None
    
    def __del__(self):
        pass

    def set_basic_info(self, engine, result_queue, result_mutex, upload_flage=False):
        self.result_queue = result_queue
        self.result_mutex = result_mutex
        self.engine = engine
        self.handle = self.engine.get_handle()
        self.bmcv = sail.Bmcv(self.handle)
        self.upload_flage = upload_flage

    def run(self):
        print("***************************************************************")
        print("upload thread start run!")
        print("***************************************************************")
        self.store = RedisClientInstance.get_storage_instance()

        first_image_id = ""
        has_get_fisrt_id = False

        sleep_frame = 0.05
        while True:
            bm_image_ost = sail.BMImage()
            self.result_mutex.acquire()
            if len(self.result_queue) <= 0:
                self.result_mutex.release()
                time.sleep(0.04)
                continue
            else:
                data = self.result_queue.pop(0)
                im_tensor =  data["frame"]
                image_id = data["id"]
                det_box = data['detection']
                frame_number = data['frame_number']

                self.bmcv.vpp_resize(im_tensor, bm_image_ost, im_tensor.width(), im_tensor.height())
                self.result_mutex.release()
            
            if has_get_fisrt_id is False:
                has_get_fisrt_id = True
                first_image_id = image_id
            
            if  first_image_id != image_id:
                # print("first id: {},image_id: {}, skip....".format(first_image_id,image_id))
                continue

            if self.upload_flage:
                # print("upload: {}".format(image_id))
                t2 = time.time()
                for idx_temp, box in enumerate(det_box):
                    self.bmcv.rectangle(bm_image_ost, int(box[0]/6), int(box[1]/6), \
                                int((box[2]-box[0])/6), int((box[3]-box[1])/6), (255, 0, 0), 3)
                # self.bmcv.imwrite('{}/{}_result_{}.jpg'.format(self.save_path,image_id,t2), bm_image_ost)

                # result_tensor = self.bmcv.bm_image_to_tensor(bm_image_ost)

                bm_image_resize = self.bmcv.vpp_resize(bm_image_ost,int(bm_image_ost.width()/2), int(bm_image_ost.height()/2))
                result_tensor = self.bmcv.bm_image_to_tensor(bm_image_resize)

                result_numpy = result_tensor.asnumpy()
                np_array_temp = result_numpy[0]
                np_array_t = np.transpose(np_array_temp, [1, 2, 0])
                mat_array = np.array(np_array_t, dtype=np.uint8)

                # print("{}:::{}:::{}:::{}".format(result_numpy.shape,np_array_temp.shape,np_array_t.shape,mat_array.shape))
                # cv2.imwrite('{}/{}_result_{}.jpg'.format(self.save_path,image_id,t2), mat_array)

                _, image = cv2.imencode('.jpg', mat_array)
                self.store.single_set_string('image', np.array(image).tobytes())
                self.store.single_set_string('image_id', image_id)
                # log.info("store image id {}".format(image_id))
                time_end = time.time()
                if self.time_start is None:
                    self.time_start=time.time()
                else:
                    time_use = time_end-self.time_start
                    if time_use < sleep_frame:
                        time_sleep = sleep_frame - time_use
                        time.sleep(time_sleep)                  #视频匀速上传
                    self.time_start = time_end
                    

    def stop(self):
        self.exit_flag = 1

