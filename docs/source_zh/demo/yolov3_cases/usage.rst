运行demo
________


获取bmodel和视频
^^^^^^^^^^^^^^^^

    若您已经按照前述说明执行/workspace/examples/sail/sc5_tests/auto_test.sh，则相应的模型文件及测试图片应当已经下载完毕。
    如果没有这些文件，运行”download.py“脚本可以下载所需的模型和数据。

        .. code-block:: shell
          
            python3 download.py yolov3_fp32.bmodel --save_path ./data
            python3 download.py yolov3_int8.bmodel --save_path ./data
            python3 download.py det.h264 --save_path ./data

运行C++程序
^^^^^^^^^^^

    请确保您已经按照前述说明执行/workspace/examples/sail/sc5_tests/auto_test.sh，该脚本将编译c++程序生成可执行文件。
    
    For case 0:
    
        .. code-block:: shell

            # run fp32 bmodel
            ./build/bin/det_yolov3_0 \
            --bmodel ./data/yolov3_fp32_191115.bmodel \
            --input ./data/det.h264 \
            --threads 2

            # run int8 bmodel
            ./build/bin/det_yolov3_0 \
            --bmodel ./data/yolov3_int8_191115.bmodel \
            --input ./data/det.h264 \
            --threads 2


    For case 1:

        .. code-block:: shell

            # run fp32 bmodel
            ./build/bin/det_yolov3_1 \
            --bmodel ./data/yolov3_fp32_191115.bmodel \
            --input ./data/det.h264 \
            --threads 2

            # run int8 bmodel
            ./build/bin/det_yolov3_1 \
            --bmodel ./data/yolov3_int8_191115.bmodel \
            --input ./data/det.h264 \
            --threads 2



运行python程序
^^^^^^^^^^^^^^

    For case 0:

        .. code-block:: shell

            # run fp32 bmodel
            python3 ./python/det_yolov3/det_yolov3.py \
            --bmodel ./data/yolov3_fp32_191115.bmodel \
            --input ./data/det.h264 \
            --loops 1 \
            --tpu_id 1

            # run int8 bmodel
            python3 ./python/det_yolov3/det_yolov3.py \
            --bmodel ./data/yolov3_int8_191115.bmodel \
            --input ./data/det.h264 \
            --loops 1 \
            --tpu_id 1