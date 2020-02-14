运行demo
________


获取bmodel和图片
^^^^^^^^^^^^^^^^

    使用 "download.py" 下载所需的模型和数据. 

        .. code-block:: shell
          
           python3 download.py mtcnn_fp32.bmodel 
           python3 download.py face.jpg

运行C++程序
^^^^^^^^^^^

    For case 0:
    
        .. code-block:: shell

           ./det_mtcnn --bmodel ./mtcnn_fp32.bmodel --input ./face.jpg


运行python程序
^^^^^^^^^^^^^^

    For case 0:

        .. code-block:: shell

           python3 ./det_mtcnn.py --bmodel ./mtcnn_fp32.bmodel --input ./face.jpg



