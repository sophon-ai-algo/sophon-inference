Python代码解析
______________

Case 0: 使用 opencv 做解码和数据预处理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    如下，我们使用了一个 while 循环来模拟真实的目标检测业务：
    
        .. code-block:: python

           # ...
           # using opencv cap get video frame
           while cap.isOpened():

             # ...
             # get one frame
             ret, img = cap.read()

             # ... 
             # preforward
             data = preprocess(img, detected_size)
             
             # ...
             # set input data from host memory
             input_data = {input_name: np.array([data], dtype=np.float32)}
             
             # ...
             # forward
             output = net.process(graph_name, input_data)

             # ...
             # postforward include NMS
             bboxes, classes, probs = postprocess(output, img, detected_size, threshold)

             # ...








