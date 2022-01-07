使用SAIL驱动TPU加载bmodel并进行推理
___________________________________

您可以使用如下命令下载测试图片cls.jpg：

    .. code-block:: shell

       wget \
       https://sophon-file.sophon.cn/sophon-prod-s3/model/19/12/05/cls.jpg.tgz
       tar -zxvf cls.jpg.tgz


然后，参考下面的程序使用SAIL进行推理：

    .. code-block:: python

        #!/usr/bin/env python3

        import cv2
        import numpy as np
        import sophon.sail as sail
    
        # initialize an Engine instance using bmodel.
        bmodel = sail.Engine("bmodel/compilation.bmodel", 0, sail.IOMode.SYSIO)
        # graph_name is just the net_name in conversion step.
        graph_name = bmodel.get_graph_names()[0]
        input_tensor_name = bmodel.get_input_names(graph_name)[0]
        # why transpose?
        # bmodel will always be NCHW layout,
        # so, if original tensorflow frozen model is formatted as NHWC,
        input_data = {input_tensor_name: \
            np.expand_dims(cv2.resize(cv2.imread("cls.jpg"), (224,224)), 0)}
        # do inference
        outputs = bmodel.process(graph_name, input_data)




