Python Codes Explanation
________________________

Case 0: decoding and preprocessing with opencv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    In this case, we detect objects in multiple videos with a bmodel converted by yolov3.
    We use public released opencv to decode videos and process images.
    In the function of inference, we first initialize instances of sail::Engine:
    
        .. code-block:: python

           # ...
           # set sail ele
           net = sail.Engine(bmodel_path, tpu_id, sail.IOMode.SYSIO)
           
           # ...


    Then, we use a while-loop to process each frame of the input video.
    The core of the pipeline are decoding, preprocessing, inference, postprocessing.
    
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








