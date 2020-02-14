运行demo
________


获取bmodel和视频
^^^^^^^^^^^^^^^^

    运行”download.py“脚本可以下载所需的模型和数据。

        .. code-block:: shell
          
           python3 download.py yolov3_fp32.bmodel 
           python3 download.py yolov3_int8.bmodel
           python3 download.py det.h264

运行C++程序
^^^^^^^^^^^

    For case 0:
    
        .. code-block:: shell

           ./det_yolov3_0 --bmodel  ./yolov3_fp32.bmodel --input ./det.h264 --threads 2
           ./det_yolov3_0 --bmodel  ./yolov3_int8.bmodel --input ./det.h264 --threads 2


    For case 1:

        .. code-block:: shell

           ./det_yolov3_0 --bmodel  ./yolov3_fp32.bmodel --input ./det.h264 --threads 2
           ./det_yolov3_0 --bmodel  ./yolov3_int8.bmodel --input ./det.h264 --threads 2



运行python程序
^^^^^^^^^^^^^^

    For case 0:

        .. code-block:: shell

           python3 det_yolov3.py --bmodel ./yolov3_fp32.bmodel --input ./det.h264 --loops 1 --tpu_id 1
           python3 det_yolov3.py --bmodel ./yolov3_int8.bmodel --input ./det.h264 --loops 1 --tpu_id 1



