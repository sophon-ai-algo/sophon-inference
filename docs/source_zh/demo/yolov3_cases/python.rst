Python代码解析
______________

Case 0: 使用 opencv 做解码和数据预处理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    如下，我们使用了一个 while 循环来模拟真实的目标检测业务：
    
        .. code-block:: python

           # ...

           while cap.isOpened():

             # ...

             ret, img = cap.read()

             # ... 

             data = preprocess(img, detected_size)
             
             # ...
             
             input_data = {input_name: np.array([data], dtype=np.float32)}
             
             # ...
             
             output = net.process(graph_name, input_data)

             # ...

             bboxes, classes, probs = postprocess(output, img, detected_size, threshold)

             # ...








