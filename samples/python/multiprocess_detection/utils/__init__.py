# -*- coding:utf-8 -*-
# @author  : cbingcan
# @time    : 2021/8/24/024 11:00
SERVER_PORT= 8080

# input_img_h = 180
# input_img_w = 320
DETECTION_MODEL_PATH = 'models/compilation_cssd_32.bmodel'
DETECTION_SAVE_PATH = "/data/video/save_result"

DETECTION_TPU_ID = 0


VIDEO_LIST = [
    {"id": "35050000002000001371", "name": "001", "rtsp":"/data/video/fps25_num5.mp4"},
    {"id": "35050000002000001372", "name": "002", "rtsp":"/data/video/fps25_num5.mp4"},
    {"id": "35050000002000001373", "name": "003", "rtsp": "/data/video/fps25_num5.mp4"},
    {"id": "35050000002000001374", "name": "004", "rtsp": "/data/video/fps25_num5.mp4"},
    {"id": "35050000002000001392", "name": "013", "rtsp": "/data/video/fps25_num5.mp4"},
    {"id": "35050000002000001393", "name": "014", "rtsp": "/data/video/fps25_num5.mp4"},
    {"id": "35050000002000001394", "name": "015", "rtsp": "/data/video/fps25_num5.mp4"},
    {"id": "35050000002000001395", "name": "016", "rtsp": "/data/video/fps25_num5.mp4"}
]
