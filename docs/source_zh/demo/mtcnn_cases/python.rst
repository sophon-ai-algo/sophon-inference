Python代码解析
______________

Case 0
^^^^^^

    在本例中，我们使用的 bmodel 是一个动态模型，它的输入张量尺寸是可变的。
    bmodel 中有三个具体模型：PNet、RNet、ONet。
    其中，PNet 的输入张量的高和宽可变，RNet 和 ONet 的输入张量的 batch 可变。   
    
        .. code-block:: python

           # init Engine to load bmodel and allocate input and output tensors
           engine = sail.Engine(bmodel_path, 0, sail.SYSIO)
           # init preprocessor and postprocessor
           preprocessor = PreProcessor([127.5, 127.5, 127.5], 0.0078125)
           postprocessor = PostProcessor([0.5, 0.3, 0.7])
           # read image
           image = cv2.imread(input_path).astype(np.float32)
           image = cv2.transpose(image)
           # run PNet, the first stage of MTCNN
           boxes = run_pnet(engine, preprocessor, postprocessor, image)
           if np.array(boxes).shape[0] > 0:
             # run RNet, the second stage of MTCNN
             boxes = run_rnet(preprocessor, postprocessor, boxes, image)
             if np.array(boxes).shape[0] > 0:
               # run ONet, the third stage of MTCNN
               boxes, points = run_onet(preprocessor, postprocessor, boxes, image)
           # print detected result
           for i, bbox, prob in zip(range(len(boxes)), boxes, probs):
             print("Face {} Box: {}, Score: {}".format(i, bbox, prob))
