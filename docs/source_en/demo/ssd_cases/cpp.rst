C++ Codes Explanation
_____________________

Case 0: decoding and preprocessing with opencv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    In case 0, we exploit a bmodel converted from ssd300-vgg16 to detect objects from videos or images.
    We encapsulated an "inference" function to complete it.
    The definition of the function is:
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
    
    In this function, we first initialize a sail::Engine instance with specified device,
    and load a bmodel into this sail::Engine instance.

        .. code-block:: cpp

           // init Engine
           sail::Engine engine(tpu_id);
           // load bmodel without builtin input and output tensors
           engine.load(bmodel_path);


    Then, we read some parameters from this engine.
    Based on the information of inputs and outputs, we create tensors through sail::Tensor to hold the data.

        .. code-block:: cpp

           // get handle to create input and output tensors
           sail::Handle handle = engine.get_handle();
           // allocate input and output tensors with both system and device memory
           sail::Tensor in(handle, input_shape, input_dtype, true, true);
           sail::Tensor out(handle, output_shape, output_dtype, true, true);
           std::map<std::string, sail::Tensor*> input_tensors = {{input_name, &in}};
           std::map<std::string, sail::Tensor*> output_tensors = {{output_name, &out}};


    We also need to initialize instances from PreProcessor, PostProcessor and CvDecoder.
    
    For the instance of CvDecoder, we use it to decode videos or images.
    The CvDecoder is a virtual class defined in "cvdecoder.h", which is at the same folder of this demo.
    The factory method CvDecoder::create will create a decoder depends on the input path.

    The PreProcessor and PostProcessor are classes defined in "processor.h", which is at the same folder of this demo.
    Preprocessing contains some resizing or scaling to original input tensor,
    while postprocessing contains bbox transformation and non-max suppression.


    In the for-loop, there is a pipeline of the inference of detection:

        .. code-block:: cpp

           // read an image from a image file or a video file
           cv::Mat frame;
           if (!decoder->read(frame)) {
             break;
           }
           // preprocess
           cv::Mat img1(input_shape[2], input_shape[3], is_fp32 ? CV_32FC3 : CV_8SC3);
           preprocessor.process(frame, img1);
           mat_to_tensor(img1, in);
           // inference
           engine.process(graph_name, input_tensors, input_shapes, output_tensors);
           auto real_output_shape = engine.get_output_shape(graph_name, output_name);
           // postprocess
           float* output_data = reinterpret_cast<float*>(out.sys_data());
           std::vector<DetectRect> dets;
           postprocessor.process(dets, output_data, real_output_shape,
                                 frame.cols, frame.rows);



Case 1: decoding with bm-ffmpeg and preprocessing with bmcv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    In case 1, we use bm-ffmpeg and bmcv for decoding and preprocessing.
    But you don't need to concern about the implementation of bm-ffmpeg and bmcv.
    We have already encapsulated them into SAIL.

    For decoding, sail::Decoder is based on bm-ffmpeg to help you decode videos and images.
    Just treat sail::Decoder as cv::VideoCapture, while sail::BMImage as cv::Mat,
    you can easily understand the code below:

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


    And sail::Bmcv is used for preprocessing.
    Other codes are almost the same with case 0.


Case 2: decoding with bm-ffmpeg and preprocessing with bmcv, 4N-mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    The pipeline in case 2 is the same as that in case 1.
    But the batchsize in case 4 is 4.
    We want use this case to show you that,
    if you are using int8 computing units, batchsize is recommanded as 4 or multiples of 4.
    At this situation, you can use the TPU to its fullest.



Case 3: decoding and preprocessing with bm-opencv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    This case is suitale for SOC mode only.
    The form of calling bm-opencv in SOC mode is almost the same as calling opencv(public released) in PCIE mode.

Case 4: decoding with bm-opencv and preprocessing with bmcv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    This case is suitale for SOC mode only.
    The form of calling bm-opencv in SOC mode is almost the same as calling opencv(public released) in PCIE mode.



