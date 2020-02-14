deploy bmodel on Sophon SC5 using sail
______________________________________

    .. code-block:: python

       #!/usr/bin/env python3

       import cv2
       import numpy as np
       import sophon.sail as sail

       bmodel = sail.Engine("bmodel/compilation.bmodel", 0, sail.IOMode.SYSIO)  # initialize an Engine instance using bmodel.
       graph_name = bmodel.get_graph_names()[0]                                 # graph_name is just the net_name in conversion step.
       input_tensor_name = bmodel.get_input_names(graph_name)[0]
       # why transpose?
       # bmodel will always be NCHW layout,
       # so, if original tensorflow frozen model is formatted as NHWC,
       # we should transpose original (1, 224, 224, 3) to (1, 3, 224, 224)
       input_data = {input_tensor_name: np.transpose(np.expand_dims(cv2.resize(cv2.imread("cls.jpg"), (224,224)), 0), [0,3,1,2]).copy()}
       outputs = bmodel.process(graph_name, input_data)                         # do inference




