Usage
_____


Get model and data
^^^^^^^^^^^^^^^^^^

    To run this demo, we need both fp32 and int8 bmodels of a yolov3.
    We also need a video to be detected.
    We can get them through the script "download.py". 

        .. code-block:: shell
          
           python3 download.py yolov3_fp32.bmodel 
           python3 download.py yolov3_int8.bmodel
           python3 download.py det.h264

Run C++ cases
^^^^^^^^^^^^^

    For case 0:
    
        .. code-block:: shell

           ./det_yolov3_0 --bmodel  ./yolov3_fp32.bmodel --input ./det.h264 --threads 2
           ./det_yolov3_0 --bmodel  ./yolov3_int8.bmodel --input ./det.h264 --threads 2


    For case 1:

        .. code-block:: shell

           ./det_yolov3_0 --bmodel  ./yolov3_fp32.bmodel --input ./det.h264 --threads 2
           ./det_yolov3_0 --bmodel  ./yolov3_int8.bmodel --input ./det.h264 --threads 2



Run python cases
^^^^^^^^^^^^^^^^

    For case 0:

        .. code-block:: shell

           python3 det_yolov3.py --bmodel ./yolov3_fp32.bmodel --input ./det.h264 --loops 1 --tpu_id 1
           python3 det_yolov3.py --bmodel ./yolov3_int8.bmodel --input ./det.h264 --loops 1 --tpu_id 1



