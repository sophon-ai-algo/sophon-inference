C++代码解析
___________

Case 0: 使用 opencv 做解码和数据预处理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    在该示例程序中，我们首先对 sail::Engine、PreProcessor、PostProcessor 初始化。
    其中，PreProcessor 中封装了 opencv 的 API。
    
        .. code-block:: cpp
           // ...
           // sail Engine ele
           sail::Engine engine(bmodel_path, tpu_id, sail::SYSIO);
          
           // ... 
           // preforward 
           PreProcessor preprocessor(416, 416);
           
           // ...
           // postforward
           PostProcessor postprocessor(0.5);
           
           // ...


    然后，我们使用一个 while 循环来模拟真实的目标检测业务。
    
        .. code-block:: cpp

           // ...
           // using opencv cap get video frame
           while (cap.read(frame)) {
           
             // ...
             // preforward
             preprocessor.processv2(input, frame);
             
             // ...
             // forward
             engine.process(graph_name);
             
             // ...
             // postforward include NMS            
             auto result = postprocessor.process(output, output_shape[2], height, width);



Case 1: 使用 bm-ffmpeg 解码，使用 bmcv 做预处理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


    我们定义了 FFMpegFrameProvider 和 PreProcessorBmcv类，它们分别封装了 sail::Decoder(bm-ffmpeg), sail::Bmcv(bmcv)。
    
        .. code-block:: cpp
    
           // ...
           PreProcessorBmcv preprocessor(bmcv, input_scale, 416, 416);
           PostProcessor postprocessor(0.5);

           // ...
           // bm-ffmpeg ele
           FFMpegFrameProvider frame_provider(bmcv, input_path, tpu_id);
           // sail bmcv ele
           sail::BMImage img0, img1;

           // ...
           // using bmcv-ele get frame 
           while (!frame_provider.get(img0)) {

             // ...
             // preforward
             preprocessor.process(img0, img1);
             
             // ...
             // forward
             engine.process(graph_name);

             // ...
             // postforward include NMS
             auto result = postprocessor.process(output, output_shape[2], height, width);




