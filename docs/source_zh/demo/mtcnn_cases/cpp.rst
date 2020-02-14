C++代码解析
___________

Case 0
^^^^^^

    在本例中，我们使用的 bmodel 是一个动态模型，它的输入张量尺寸是可变的。
    bmodel 中有三个具体模型：PNet、RNet、ONet。
    其中，PNet 的输入张量的高和宽可变，RNet 和 ONet 的输入张量的 batch 可变。
    
        .. code-block:: cpp
        
           // init Engine to load bmodel and allocate input and output tensors
           sail::Engine engine(bmodel_path, tpu_id, sail::SYSIO);
           // init preprocessor and postprocessor
           PreProcessor preprocessor(127.5, 127.5, 127.5, 0.0078125);
           double threshold[3] = {0.5, 0.3, 0.7};
           PostProcessor postprocessor(threshold);
           auto reference = postprocessor.get_reference(compare_path);
           // read image
           cv::Mat frame = cv::imread(input_path);
           bool status = true;
           for (int i = 0; i < loops; ++i) {
             cv::Mat image = frame.t();
             // run PNet, the first stage of MTCNN
             auto boxes = run_pnet(engine, preprocessor, postprocessor, image);
             if (boxes.size() != 0) {
               // run RNet, the second stage of MTCNN
               boxes = run_rnet(engine, preprocessor, postprocessor, boxes, image);
               if (boxes.size() != 0) {
                 // run ONet, the third stage of MTCNN
                 boxes = run_onet(engine, preprocessor, postprocessor, boxes, image);
               }
             }
             // print_result
             if (postprocessor.compare(reference, boxes)) {
               print_result(boxes);
             } else {
                 status = false;
                 break;
             }
           }


