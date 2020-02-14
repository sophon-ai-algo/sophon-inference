Python Codes Explanation
________________________

Case 0
^^^^^^

    In this case, we will experience the dynamic model, mtcnn, whose input shapes is variable.
    There are 3 graphs in the MTCNN model: PNet, RNet and ONet. 
    Input height and width may change for Pnet while input batch_szie may change for RNet and Onet.
    
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
