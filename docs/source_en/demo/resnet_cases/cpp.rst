C++ Codes Explanation
_____________________

Case 0: simplest case
^^^^^^^^^^^^^^^^^^^^^

    In case 0, we encapsulated a function named "inference", as follows:
        .. code-block:: cpp

           bool inference(
           const std::string& bmodel_path,
           const std::string& input_path,
           int                tpu_id,
           int                loops,
           const std::string& compare_path);

    The bmodel_path is the path of the bmodel of resnet50 which converted from a caffemodel of official resnet50.
    We use this bmodel to initialize a sail::Engine instance, for futher inference.
    We can get parameters, like graph_name, input_name and so on, from the sail::Engine instance.
        .. code-block:: cpp

           sail::Engine engine(bmodel_path, tpu_id, sail::SYSIO);
           auto graph_name = engine.get_graph_names().front();
           auto input_name = engine.get_input_names(graph_name).front();
           auto output_name = engine.get_output_names(graph_name).front();
           auto input_shape = engine.get_input_shape(graph_name, input_name);
           auto output_shape = engine.get_output_shape(graph_name, output_name);
           auto in_dtype = engine.get_input_dtype(graph_name, input_name);
           auto out_dtype = engine.get_output_dtype(graph_name, output_name);


    Actually, you can get this information by using the "bm_model.bin" tool in BMNNSDK:

        .. code-block:: shell

           # fp32_bmodel
           bitmain@bitmain:~$ bm_model.bin --info resnet50_fp32_191115.bmodel
           # bmodel version: B.2.2
           # chip: BM1684
           # create time: Sat Nov 23 14:37:37 2019
           #
           # ==========================================
           # net: [ResNet-50_fp32]  index: [0]
           # ------------
           # stage: [0]  static
           # input: data, [1, 3, 224, 224], float32
           # output: fc1000, [1, 1000], float32

           # int8_bmodel
           # bitmain@bitmain:~$ bm_model.bin --info resnet50_int8_191115.bmodel
           # bmodel version: B.2.2
           # chip: BM1684
           # create time: Sat Nov 23 14:38:50 2019
           #
           # ==========================================
           # net: [ResNet-50_int8]  index: [0]
           # ------------
           # stage: [0]  static
           # input: data, [1, 3, 224, 224], int8
           # output: fc1000, [1, 1000], int8


    The input_path is the path of an arbitary image.
    We supplied ready-made bmodels and images, the script ${sophon-inference}/tools/download.py can help you get them.

    The tpu_id indicates which TPU you want to use.
    default value of tpu_id is 0, means using first TPU on your PC or Server.

    The loops determines how many times you will run the bmodel.
    Let's see what happened in the loop:
        .. code-block:: cpp

           for (int i = 0; i < loops; ++i) {
             // read image
             cv::Mat frame = cv::imread(input_path);
             // preprocess
             preprocessor.process(input, frame);
             // scale input data if input data type is int8 or uint8
             if (in_dtype != BM_FLOAT32) {
               engine.scale_input_tensor(graph_name, input_name, input);
             }
             // inference
             engine.process(graph_name);
             // scale output data if input data type is int8 or uint8
             if (out_dtype != BM_FLOAT32) {
               engine.scale_output_tensor(graph_name, output_name, output);
             }
             // postprocess
             auto result = postprocessor.process(output);
             // print result
             for (auto item : result) {
               spdlog::info("Top 5 of loop {}: [{}]", i, fmt::join(item, ", "));
               if(!postprocessor.compare(reference, item,
                   (out_dtype == BM_FLOAT32) ? "fp32" : "int8")) {
                 status = false;
                 break;
               }
             }
             if (!status) {
               break;
             }
           }

    As the codes shown, in each loop, we read an image from a string path to get a cv::Mat instance.
    Then, we do some preprocessing on the image data, like resizing.
    After preprocessing, we will scale the values of the data depends on its data type, 
    this procedure is required by the int8 mode, 
    which data should be converted from fp32 to int8 by a scale factor.
    Due to the pointer of the input tensor data was already stored in the SAIL::Engine instance, 
    we only need to use the "engine.process(graph_name)" to drive bmodel to do inference.
    And at last, postprocessing the output tensor which data pointer was also stored in the SAIL::Engine instance.
    Apparently, we can execute the inference pipeline(the loop) shown above, for many times, with feeding different images.

Case 1: multi-thread implementation of case 0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    In case 1, we will show the multi-thread programming mode of SAIL::Engine.
    Simplely, one bmodel was loaded by one SAIL::Engine instance, while input/output tensors are managed outside this SAIL::Engine instance in different threads.

    We loaded the bmodel into SAIL::Engine instance after constucor, not in the constructor:
        .. code-block:: cpp

           // init Engine
           sail::Engine engine(tpu_id);
           // load bmodel without builtin input and output tensors
           // each thread manage its input and output tensors
           int ret = engine.load(bmodel_path);
 
    In each thread, we seperately managed the input and output tensors.
    While in case 0, these tensors were managed automatically in the SAIL::Engine instance.
        .. code-block:: cpp

           // get handle to create input and output tensors
           sail::Handle handle = engine->get_handle();
           // allocate input and output tensors with both system and device memory
           sail::Tensor in(handle, input_shape, in_dtype, true, true);
           sail::Tensor out(handle, output_shape, out_dtype, true, true);
           std::map<std::string, sail::Tensor*> input_tensors = {{input_name, &in}};
           std::map<std::string, sail::Tensor*> output_tensors = {{output_name, &out}};


Case 2: multi-thread with multiple models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    In case 2, we will load different bmodels into a SAIL::Engine instance for each inference thread.
    The codes in case 2 is a little different with that in case 1.
    Just place the "SAIL::Engine.load(bmodel) function" into each thread is OK.
    In this case, we used a loading thread to finish it.
        .. code-block:: cpp

           /**
           * @brief Load a bmodel.
           *
           * @param thread_id   Thread id
           * @param engine      Pointer to an Engine instance
           * @param bmodel_path Path to bmodel
           */
           void thread_load(
                int                thread_id,
                sail::Engine*      engine,
                const std::string& bmodel_path) {
                  int ret = engine->load(bmodel_path);
                  if (ret == 0) {
                    auto graph_name = engine->get_graph_names().back();
                    spdlog::info("Thread {} load {} successfully.", thread_id, graph_name);
                  }
                }

    Other codes are almost the same with case 1.


Case 3: multi-thread with multiple TPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    In Case 3, we will exploit multiple TPUs to do inference.
    While the SAIL:Engine instance is bound to device, 
    we should initialize multiple SAIL::Engine instances for each TPU.
        .. code-block:: cpp

           // init Engine to load bmodel and allocate input and output tensors
           // one engine for one TPU
           std::vector<sail::Engine*> engines(thread_num, nullptr);
           for (int i = 0; i < thread_num; ++i) {
             engines[i] = new sail::Engine(bmodel_path, tpu_ids[i], sail::SYSIO);
           }

    Other codes are almost the same with case 1.




