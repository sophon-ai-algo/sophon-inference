运行demo
________


获取bmodel和图片
^^^^^^^^^^^^^^^^

    若您已经按照前述说明执行/workspace/examples/sail/sc5_tests/auto_test.sh，则相应的模型文件及测试图片应当已经下载完毕。
    如果没有这些文件，使用 "download.py" 下载所需的模型和数据. 

        .. code-block:: shell
          
           python3 download.py mtcnn_fp32.bmodel --save_path ./data
           python3 download.py face.jpg --save_path ./data

运行C++程序
^^^^^^^^^^^

    请确保您已经按照前述说明执行/workspace/examples/sail/sc5_tests/auto_test.sh，该脚本将编译c++程序生成可执行文件。
    
    For case 0:
    
        .. code-block:: shell

           ./build/bin/det_mtcnn \
           --bmodel ./data/mtcnn_fp32_191115.bmodel \
           --input ./data/face.jpg


运行python程序
^^^^^^^^^^^^^^

    For case 0:

        .. code-block:: shell

           python3 ./python/det_mtcnn/det_mtcnn.py \
            --bmodel ./data/mtcnn_fp32_191115.bmodel \
            --input ./data/face.jpg