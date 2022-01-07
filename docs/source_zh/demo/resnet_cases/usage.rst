运行demo
________


获取bmodel和图片
^^^^^^^^^^^^^^^^

    在运行 demo 之前，我们需要下载 rensnet50 的 fp32_bmodel 和 int8_bmodel。
    另外，也需要一张图片作为待处理的输入。
    若您已经按照前述说明执行/workspace/examples/sail/sc5_tests/auto_test.sh，则相应的模型文件及测试图片应当已经下载完毕。
    如果没有这些文件，可以使用 ”dowload.py“ 下载。
        .. code-block:: shell
          
           python3 download.py resnet50_fp32.bmodel --save_path ./data
           python3 download.py resnet50_int8.bmodel --save_path ./data
           python3 download.py cls.jpg --save_path ./data

    提示: 请从sdk/examples/sail/se5_tests/scripts/ 或 sdk/examples/sail/sc5_tests/目录下获取download.py。
    
运行C++程序
^^^^^^^^^^^

    请确保您已经按照前述说明执行/workspace/examples/sail/sc5_tests/auto_test.sh，该脚本将编译c++程序生成可执行文件。

    For case 0:
    
        .. code-block:: shell

           # run fp32 bmodel
           ./build/bin/cls_resnet_0 \
           --bmodel ./data/resnet50_fp32_191115.bmodel \
           --input ./data/cls.jpg

           # run int8 bmodel
           ./build/bin/cls_resnet_0 \
           --bmodel ./data/resnet50_int8_191115.bmodel \
           --input ./data/cls.jpg


    For case 1:

        .. code-block:: shell

           # run fp32 bmodel
           ./build/bin/cls_resnet_1 \
           --bmodel ./data/resnet50_fp32_191115.bmodel \
           --input ./data/cls.jpg \
           --threads 2

           # run int8 bmodel
           ./build/bin/cls_resnet_1 \
           --bmodel ./data/resnet50_int8_191115.bmodel \
           --input ./data/cls.jpg \
           --threads 2


    For case 2:

        .. code-block:: shell

           # run fp32 bmodel and int8 bmodel in two threads
           ./build/bin/cls_resnet_2 \
           --bmodel ./data/resnet50_fp32_191115.bmodel \
           --bmodel ./data/resnet50_int8_191115.bmodel \
           --input ./data/cls.jpg


    For case 3:

        .. code-block:: shell

           # run fp32 bmodel
           ./build/bin/cls_resnet_3 \
           --bmodel ./data/resnet50_fp32_191115.bmodel \
           --input ./data/cls.jpg \
           --tpu_id 0 \
           --tpu_id 1

           # run int8 bmodel
           ./build/bin/cls_resnet_3 \
           --bmodel ./data/resnet50_int8_191115.bmodel \
           --input ./data/cls.jpg \
           --tpu_id 0 \
           --tpu_id 1



运行python程序
^^^^^^^^^^^^^^

    For case 0:

        .. code-block:: shell

           # run fp32 bmodel
           python3 ./python/cls_resnet/cls_resnet_0.py \
           --bmodel ./data/resnet50_fp32_191115.bmodel \
           --input ./data/cls.jpg --loops 1

           # run int8 bmodel
           python3 ./python/cls_resnet/cls_resnet_0.py \
           --bmodel ./data/resnet50_int8_191115.bmodel \
           --input ./data/cls.jpg --loops 1




    For case 1:

        .. code-block:: shell

           # run fp32 bmodel
           python3 ./python/cls_resnet/cls_resnet_1.py \
           --bmodel ./data/resnet50_fp32_191115.bmodel \
           --input ./data/cls.jpg --threads 2

           # run int8 bmodel
           python3 ./python/cls_resnet/cls_resnet_1.py \
           --bmodel ./data/resnet50_int8_191115.bmodel \
           --input ./data/cls.jpg --threads 2



    For case 2:

        .. code-block:: shell

           # run fp32 bmodel and int8 bmodel in two threads
           python3 ./python/cls_resnet/cls_resnet_2.py \
           --bmodel ./data/resnet50_fp32_191115.bmodel \
           --bmodel ./data/resnet50_int8_191115.bmodel \
           --input ./data/cls.jpg


    For case 3:

        .. code-block:: shell

           # run fp32 bmodel
           python3 ./python/cls_resnet/cls_resnet_3.py \
           --bmodel ./data/resnet50_fp32_191115.bmodel \
           --input ./data/cls.jpg \
           --tpu_id 0 \
           --tpu_id 1

           # run int8 bmodel
           python3 ./python/cls_resnet/cls_resnet_3.py \
           --bmodel ./data/resnet50_int8_191115.bmodel \
           --input ./data/cls.jpg \
           --tpu_id 0 \
           --tpu_id 1

