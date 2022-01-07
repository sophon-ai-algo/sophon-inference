运行demo
________


获取bmodel、图像、视频
^^^^^^^^^^^^^^^^^^^^^^

    若您已经按照前述说明执行/workspace/examples/sail/sc5_tests/auto_test.sh，则相应的模型文件及测试图片应当已经下载完毕。
    如果没有这些文件，我们提供了”download.py“脚本下载所需的模型与数据。
        
        .. code-block:: shell
          
           python3 download.py ssd_fp32.bmodel --save_path ./data
           python3 download.py ssd_int8.bmodel --save_path ./data
           python3 download.py det.jpg --save_path ./data
           python3 download.py det.h264 --save_path ./data

运行C++程序
^^^^^^^^^^^

    请确保您已经按照前述说明执行/workspace/examples/sail/sc5_tests/auto_test.sh，该脚本将编译c++程序生成可执行文件。
    
    For case 0:
    
        .. code-block:: shell

           # run fp32 bmodel with input of image
           ./build/bin/det_ssd_0 \
           --bmodel ./data/ssd_fp32_191115.bmodel \
           --input ./data/det.jpg \
           --loops 1

           # run int8 bmodel with input of video
           ./build/bin/det_ssd_0 \
           --bmodel ./data/ssd_int8_191115.bmodel \
           --input ./data/det.h264 \
           --loops 1


    For case 1:

        .. code-block:: shell

           # run fp32 bmodel with input of image
           ./build/bin/det_ssd_1 \
           --bmodel ./data/ssd_fp32_191115.bmodel \
           --input ./data/det.jpg \
           --loops 1

           # run int8 bmodel with input of video
           ./build/bin/det_ssd_1 \
           --bmodel ./data/ssd_int8_191115.bmodel \
           --input ./data/det.h264 \
           --loops 1


    For case 2:

        .. code-block:: shell

           # run fp32 bmodel with input of image
           ./build/bin/det_ssd_2 \
           --bmodel ./data/ssd_fp32_191115.bmodel \
           --input ./data/det.jpg \
           --loops 1

           # run int8 bmodel with input of video
           ./build/bin/det_ssd_2 \
           --bmodel ./data/ssd_int8_191115.bmodel \
           --input ./data/det.h264 \
           --loops 1


运行python程序
^^^^^^^^^^^^^^

    For case 0:

        .. code-block:: shell

           # run fp32 bmodel with input of image
           python3 ./python/det_ssd/det_ssd_0.py \
           --bmodel ./data/ssd_fp32_191115.bmodel \
           --input ./data/det.jpg \
           --loops 1 \
           --tpu_id 0 \
           --compare ./python/det_ssd/verify_ssd_0_fp32_image.json

           # run int8 bmodel with input of video
           python3 ./python/det_ssd/det_ssd_0.py \
           --bmodel ./data/ssd_int8_191115.bmodel \
           --input ./data/det.h264 \
           --loops 1 \
           --tpu_id 0 \
           --compare ./python/det_ssd/verify_ssd_0_int8_video.json


    For case 1:

        .. code-block:: shell

           # run fp32 bmodel with input of image
           python3 ./python/det_ssd/det_ssd_1.py \
           --bmodel ./data/ssd_fp32_191115.bmodel \
           --input ./data/det.jpg \
           --loops 1 \
           --tpu_id 0 \
           --compare ./python/det_ssd/verify_ssd_1_fp32_image.json

           # run int8 bmodel with input of video
           python3 ./python/det_ssd/det_ssd_1.py \
           --bmodel ./data/ssd_int8_191115.bmodel \
           --input ./data/det.h264 \
           --loops 1 \
           --tpu_id 0 \
           --compare ./python/det_ssd/verify_ssd_1_int8_video.json


    For case 2:

        .. code-block:: shell

           # run fp32 bmodel with input of image
           python3 ./python/det_ssd/det_ssd_2.py \
           --bmodel ./data/ssd_fp32_191115.bmodel \
           --input ./data/det.jpg \
           --loops 1 \
           --tpu_id 0 \
           --compare ./python/det_ssd/verify_ssd_2_fp32_image.json

           # run int8 bmodel with input of video
           python3 ./python/det_ssd/det_ssd_2.py \
           --bmodel ./data/ssd_int8_191115.bmodel \
           --input ./data/det.h264 \
           --loops 1 \
           --tpu_id 0 \
           --compare ./python/det_ssd/verify_ssd_2_int8_video.json