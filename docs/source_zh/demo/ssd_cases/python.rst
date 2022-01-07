Python代码解析
______________

Case 0: 使用 opencv 解码和数据预处理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    在该示例中，我们封装了一个函数，完成目标检测的任务，如下：

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

    在该函数中，我们通过循环处理视频的连续帧来模拟真实的目标检测业务。
    在本示例程序中，我们使用 opencv 进行视频解码和预处理。

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


Case 1: 使用 bm-ffmpeg 解码、使用 bmcv 做预处理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    在该示例程序中，我们使用 bm-ffmpeg 做解码，bmcv 库做预处理。
    我们已经在 SAIL 封装了 bm-ffmpeg 与 bmcv 的 API，因此用户无需关注其底层实现。

        .. code-block:: python

           # init decoder
           decoder = sail.Decoder(input_path, True, tpu_id)
           # pipeline of inference
           for i in range(loops):
             # read an image from a image file or a video file
             img0 = decoder.read(handle)
             # do somethig ...





Case 2: case 1 的 4N 模式
^^^^^^^^^^^^^^^^^^^^^^^^^

    case 2 的流程与 case 1 几乎一致，但其 bmodel 的 batch 维度是 4。
    因此，需要 4 张图像或者 4 帧视频一起处理。

    当 bmodel 的 batch 为 4 的倍数时，可以发挥出 TPU 上 int8 算力的最大性能。



Case 3: 使用 bm-opencv 做解码和数据预处理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Case 3 只适用于SOC模式。
    在SOC模式下调用bm-opencv和在PCIE模式下调用opencv的方法基本一致。

Case 4: 使用 bm-opencv 做解码，使用 bmcv 做预处理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Case 3 只适用于SOC模式。
    在SOC模式下调用bm-opencv和在PCIE模式下调用opencv的方法基本一致。



