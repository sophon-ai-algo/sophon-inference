C++代码解析
___________

Case 0: 使用 opencv 解码和数据预处理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    在该示例中，我们封装了一个函数，完成目标检测的任务，如下：

        .. code-block:: cpp

           /**
           * @brief Load a bmodel and do inference.
           *
           * @param bmodel_path  Path to bmodel
           * @param input_path   Path to input file
           * @param tpu_id       ID of TPU to use
           * @param loops        Number of loops to run
           * @param compare_path Path to correct result file
           * @return Program state
           *     @retval true  Success
           *     @retval false Failure
           */
           bool inference(
                const std::string& bmodel_path,
                const std::string& input_path,
                int                tpu_id,
                int                loops,
                const std::string& compare_path);
    


    在该函数中，我们通过循环处理同一张图像或者视频的连续帧来模拟真实的目标检测业务。

        .. code-block:: cpp
  
           // pipeline of inference
           for (int i = 0; i < loops; ++i) {
             // read an image from a image file or a video file
             cv::Mat frame;
             if (!decoder->read(frame)) {
               break;
             }
             // preprocess
             cv::Mat img1(input_shape[2], input_shape[3], is_fp32 ? CV_32FC3 : CV_8SC3);
             preprocessor.process(frame, img1);
             mat_to_tensor_(img1, in);
             // inference
             engine.process(graph_name, input_tensors, input_shapes, output_tensors);
             auto real_output_shape = engine.get_output_shape(graph_name, output_name);
             // postprocess
             float* output_data = reinterpret_cast<float*>(out.sys_data());
             std::vector<DetectRect> dets;
             postprocessor.process(dets, output_data, real_output_shape,
                                   frame.cols, frame.rows);
             // ...
           }


    在该示例程序中，我们使用了 opencv 进行图像/视频的解码和预处理，
    CvDecoder、PreProcess 类中均封装了 opencv 的相关 API。

        .. code-block:: cpp

           PreProcessor preprocessor(scale);

           // ...

           CvDecoder* decoder = CvDecoder::create(input_path);




Case 1: 使用 bm-ffmpeg 解码、使用 bmcv 做预处理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    在该示例程序中，我们使用 bm-ffmpeg 做解码，bmcv 库做预处理。
    我们已经在 SAIL 封装了 bm-ffmpeg 与 bmcv 的 API，因此用户无需关注其底层实现。

    如下，你可以将 sail::Decoder 看成是 cv::VideoCapture，将 sail::BMImage 看成 cv::Mat；

        .. code-block:: cpp

           // init decoder.
           // use bm-ffmpeg to decode video. default output format is compressed NV12
           sail::Decoder decoder(input_path, true, tpu_id);
           bool status = true;
           // pipeline of inference
           for (int i = 0; i < loops; ++i) {
             // read an image from a image file or a video file
             sail::BMImage img0 = decoder.read(handle);
            
             // do something...

           }




Case 2: case 1 的 4N 模式
^^^^^^^^^^^^^^^^^^^^^^^^^

    case 2 的流程与 case 1 几乎一致，但其 bmodel 的 batch 维度是 4。
    因此，需要 4 张图像或者 4 帧视频一起处理。

    当 bmodel 的 batch 为 4 的倍数时，可以发挥出 TPU 上 int8 算力的最大性能。


Case 3: 使用 bm-opencv 进行解码和预处理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    This case is suitale for SOC mode only.
    The form of calling bm-opencv in SOC mode is almost the same as calling opencv(public released) in PCIE mode.

Case 4: 使用 bm-opencv 解码、使用 bmcv 做预处理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    This case is suitale for SOC mode only.
    The form of calling bm-opencv in SOC mode is almost the same as calling opencv(public released) in PCIE mode.



