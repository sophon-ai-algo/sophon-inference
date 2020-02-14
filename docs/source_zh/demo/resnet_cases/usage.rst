运行demo
________


获取bmodel和图片
^^^^^^^^^^^^^^^^

    在运行 demo 之前，我们需要下载 rensnet50 的 fp32_bmodel 和 int8_bmodel。
    另外，也需要一张图片作为待处理的输入。
    我们可以使用 ”dowload.py“ 下载。
        .. code-block:: shell
          
           python3 download.py resnet50_fp32.bmodel 
           python3 download.py resnet50_int8.bmodel
           python3 download.py cls.jpg

运行C++程序
^^^^^^^^^^^

    For case 0:
    
        .. code-block:: shell

           # run fp32 bmodel
           ./cls_resnet_0 --bmodel ./resnet50_fp32.bmodel --input ./cls.jpg

           # run int8 bmodel
           ./cls_resnet_0 --bmodel ./resnet50_fp32.bmodel --input ./cls.jpg


    For case 1:

        .. code-block:: shell

           # run fp32 bmodel
           ./cls_resnet_1 --bmodel ./resnet50_fp32.bmodel --input ./cls.jpg --threads 2

           # run int8 bmodel
           ./cls_resnet_1 --bmodel ./resnet50_int8.bmodel --input ./cls.jpg --threads 2


    For case 2:

        .. code-block:: shell

           # run fp32 bmodel and int8 bmodel in two threads
           ./cls_resnet_2 --bmodel ./resnet50_fp32.bmodel --bmodel ./resnet50_int8.bmodel --input ./cls.jpg


    For case 3:

        .. code-block:: shell

           # run fp32 bmodel
           ./cls_resnet_3 --bmodel ./resnet50_fp32.bmodel --input ./cls.jpg --tpu_id 0 --tpu_id 1

           # run int8 bmodel
           ./cls_resnet_3 --bmodel ./resnet50_int8.bmodel --input ./cls.jpg --tpu_id 0 --tpu_id 1



运行python程序
^^^^^^^^^^^^^^

    For case 0:

        .. code-block:: shell

           # run fp32 bmodel
           python3 ./cls_resnet_0.py --bmodel ./resnet50_fp32.bmodel --input ./cls.jpg --loops 1

           # run int8 bmodel
           python3 ./cls_resnet_0.py --bmodel ./resnet50_int8.bmodel --input ./cls.jpg --loops 1




    For case 1:

        .. code-block:: shell

           # run fp32 bmodel
           python3 ./cls_resnet_1.py --bmodel ./resnet50_fp32.bmodel --input ./cls.jpg --threads 2

           # run int8 bmodel
           python3 ./cls_resnet_1.py --bmodel ./resnet50_int8.bmodel --input ./cls.jpg --threads 2



    For case 2:

        .. code-block:: shell

           # run fp32 bmodel and int8 bmodel in two threads
           python3 ./cls_resnet_2.py --bmodel ./resnet50_fp32.bmodel --bmodel ./resnet50_int8.bmodel --input ./cls.jpg


    For case 3:

        .. code-block:: shell

           # run fp32 bmodel
           python3 ./cls_resnet_3.py --bmodel ./resnet50_fp32.bmodel --input ./cls.jpg --tpu_id 0 --tpu_id 1

           # run int8 bmodel
           python3 ./cls_resnet_3.py --bmodel ./resnet50_int8.bmodel --input ./cls.jpg --tpu_id 0 --tpu_id 1

