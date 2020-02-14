Usage
_____


Get model and data
^^^^^^^^^^^^^^^^^^

    To run this demo, we need both fp32 and int8 bmodels of a ssd.
    We also need an image and a video to be detected.
    We can get them through the script "download.py". 

        .. code-block:: shell
          
           python3 download.py ssd_fp32.bmodel 
           python3 download.py ssd_int8.bmodel
           python3 download.py det.jpg
           python3 download.py det.h264

Run C++ cases
^^^^^^^^^^^^^

    For case 0:
    
        .. code-block:: shell

           # run fp32 bmodel with input of image
           ./det_ssd_0 --bmodel ./ssd_fp32.bmodel --input ./det.jpg --loops 1

           # run int8 bmodel with input of video
           ./det_ssd_0 --bmodel ./ssd_int8.bmodel --input ./det.h264 --loops 1


    For case 1:

        .. code-block:: shell

           # run fp32 bmodel with input of image
           ./det_ssd_1 --bmodel ./ssd_fp32.bmodel --input ./det.jpg --loops 1

           # run int8 bmodel with input of video
           ./det_ssd_1 --bmodel ./ssd_int8.bmodel --input ./det.h264 --loops 1


    For case 2:

        .. code-block:: shell

           # run int8 bmodel with input of video
           ./det_ssd_2 --bmodel ./ssd_int8.bmodel --input ./det.h264 --loops 1



    For case 3:

        .. code-block:: shell

           # run fp32 bmodel with input of image
           ./det_ssd_3 --bmodel ./ssd_fp32.bmodel --input ./det.jpg --loops 1

           # run int8 bmodel with input of video
           ./det_ssd_3 --bmodel ./ssd_int8.bmodel --input ./det.h264 --loops 1


    For case 4:

        .. code-block:: shell

           # run fp32 bmodel with input of image
           ./det_ssd_4 --bmodel ./ssd_fp32.bmodel --input ./det.jpg --loops 1

           # run int8 bmodel with input of video
           ./det_ssd_4 --bmodel ./ssd_int8.bmodel --input ./det.h264 --loops 1




Run python cases
^^^^^^^^^^^^^^^^

    For case 0:

        .. code-block:: shell

           # run fp32 bmodel with input of image
           python3 ./det_ssd_0.py --bmodel ./ssd_fp32.bmodel --input ./det.jpg --loops 1 --tpu_id 0 --compare verify_det_jpg_fp32_0.json

           # run int8 bmodel with input of video
           python3 ./det_ssd_0.py --bmodel ./ssd_int8.bmodel --input ./det.h264 --loops 1 --tpu_id 0 --compare verify_det_h264_int8_0.json


    For case 1:

        .. code-block:: shell

           # run fp32 bmodel with input of image
           python3 ./det_ssd_1.py --bmodel ./ssd_fp32.bmodel --input ./det.jpg --loops 1 --tpu_id 0 --compare verify_det_jpg_fp32_1.json

           # run int8 bmodel with input of video
           python3 ./det_ssd_1.py --bmodel ./ssd_int8.bmodel --input ./det.h264 --loops 1 --tpu_id 0 --compare verify_det_h264_int8_0.json


    For case 2:

        .. code-block:: shell

           # run int8 bmodel with input of video
           python3 ./det_ssd_2.py --bmodel ./ssd_int8.bmodel --input ./det.h264 --loops 1 --tpu_id 0 --compare verify_det_h264_int8_2.json



