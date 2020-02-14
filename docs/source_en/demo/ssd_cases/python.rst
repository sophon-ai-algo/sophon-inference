Python Codes Explanation
________________________

Case 0: decoding and preprocessing with opencv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    In case 0, we exploit a bmodel converted from ssd300-vgg16 to detect objects from videos or images.
    We encapsulated an "inference" function to complete it.
    The definition of the function is:
        .. code-block:: python

           def inference(bmodel_path, input_path, loops, tpu_id, compare_path):
             """ Load a bmodel and do inference.
             Args:
               bmodel_path: Path to bmodel
               input_path: Path to input file
               loops: Number of loops to run
               tpu_id: ID of TPU to use
               compare_path: Path to correct result file

             Returns:
               True for success and False for failure
             """

    In this function, we first initialize a sail::Engine instance with specified device,
    and load a bmodel into this sail::Engine instance.

        .. code-block:: python

           # init Engine and load bmodel
           engine = sail.Engine(bmodel_path, tpu_id, sail.IOMode.SYSIO)


    We also need to initialize instances from PreProcessor, PostProcessor and Decoder.
    
    In this case, the decoder we used is the VideoCapture of opencv, we use it to decode videos or images.

    The PreProcessor and PostProcessor are classes just defined in this script.
    Preprocessing contains some resizing or scaling to original input tensor,
    while postprocessing contains bbox transformation and non-max suppression.

        .. code-block:: python

           class PreProcessor:
             """ Preprocessing class.
             """

           class PostProcessor:
             """ Postprocessing class.

    In the for-loop, there is a pipeline of the inference of detection:

        .. code-block:: python

           # pipeline of inference
           for i in range(loops):
             # read an image from a image file or a video file
             ret, img0 = cap.read()
             if not ret:
               break
             h, w, _ = img0.shape
             # preprocess
             data = preprocessor.process(img0)
             # inference
             input_tensors = {input_name: np.array([data], dtype=np.float32)}
             output = engine.process(graph_name, input_tensors)
             # postprocess
             dets = postprocessor.process(output[output_name], w, h)
             # print result
             # ...


Case 1: decoding with bm-ffmpeg and preprocessing with bmcv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    In case 1, we use bm-ffmpeg and bmcv for decoding and preprocessing.
    But you don't need to concern about the implementation of bm-ffmpeg and bmcv.
    We have already encapsulated them into SAIL.

    For decoding, sail::Decoder is based on bm-ffmpeg to help you decode videos and images.
    Just treat sail::Decoder as cv::VideoCapture, while sail::BMImage as cv::Mat,
    you can easily understand the code below:

        .. code-block:: python

           # init decoder
           decoder = sail.Decoder(input_path, True, tpu_id)
           # pipeline of inference
           for i in range(loops):
             # read an image from a image file or a video file
             img0 = decoder.read(handle)
             # do somethig ...



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



