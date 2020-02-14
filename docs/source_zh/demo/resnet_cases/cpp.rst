C++ 代码解析
____________

Case 0: 基础示例程序
^^^^^^^^^^^^^^^^^^^^

    在该示例中，我们封装了一个函数，完成图像分类的过程，如下：

        .. code-block:: cpp

           bool inference(
           const std::string& bmodel_path,
           const std::string& input_path,
           int                tpu_id,
           int                loops,
           const std::string& compare_path);


    在该函数中，我们通过循环处理同一张图像来模拟真实的图像分类业务。

        .. code-block:: cpp

           // pipeline of inference
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
             // ...
           }

    如上代码所示，在每次循环中，我们首先使用 opencv 的 ”imread“ 函数解码图片。
    然后对图像进行预处理，获取 bmodel 的输入张量，并根据 bmodel 的数据类型缩放输入张量，
    缩放比例是 bmodel 中的参数，已经事先被加载到 engine 中。
    接着驱动 TPU 进行推理。
    最后对 TPU 输出的张量进行后处理，获得最终的结果(top-5)。
    
    在该示例程序中，图片解码和预处理的实现中直接调用了 opencv 的相关函数，
    同时后处理比较简单，只计算了 top-5 分类结果。
    因此你可以直接参考相关代码，下面我们主要介绍模型推理部分。

    bmodel_path 是本例中用到的 resnet50 bmodel 的路径，fp32 或 int8 皆可。
    我们使用该 bmodel 初始化了一个 sail::Engine 的实例，
    该实例中保存了模型的基础信息，将作为我们后续模型推理的载体。
    如果你的机器上有多个 TPU，那么可以通过 tpu_id 指定使用哪个 TPU 进行推理，编号从0开始。

        .. code-block:: cpp

           sail::Engine engine(bmodel_path, tpu_id, sail::SYSIO);


    我们可以通过该实例提供的方法获取模型的属性，代码如下：

        .. code-block:: cpp

           sail::Engine engine(bmodel_path, tpu_id, sail::SYSIO);
           auto graph_name = engine.get_graph_names().front();
           auto input_name = engine.get_input_names(graph_name).front();
           auto output_name = engine.get_output_names(graph_name).front();
           auto input_shape = engine.get_input_shape(graph_name, input_name);
           auto output_shape = engine.get_output_shape(graph_name, output_name);
           auto in_dtype = engine.get_input_dtype(graph_name, input_name);
           auto out_dtype = engine.get_output_dtype(graph_name, output_name);

    graph_name 代表 bmodel 中我们要使用的具体模型的名称，在该示例程序中，bmodel 中只包含一个模型。
    input_name 代表我们使用的具体模型中的输入张量名称，在该模型中，只有一个输入张量。
    output_name 代表我们使用的具体模型中的输出张量名称，在该模型中，只有一个输出张量。
    input_shape 和 output_shape 代表指定张量的尺寸。
    in_dtype 和 out_dtype 代表指定张量的数据类型。

    实际上，你也可以使用 BMNNSDK 中的 bm_model.bin 工具获取上述信息，如下：

        .. code-block:: cpp

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

        
Case 1: 单模型多线程
^^^^^^^^^^^^^^^^^^^^

    在 case 0 中，我们使用了 bmodel 初始化了一个 sail::Engine 的实例，模型推理的过程是在主线程中进行的。
    在 case 1 中，我们将展示如果在多个线程中使用同一个 sail::Engine 实例做模型推理。
    
    我们定义了一个 ”thread_infer“ 函数来实现子线程中的模型推理逻辑，如下：

        .. code-block:: cpp

           void thread_infer(
             int                 thread_id,
             sail::Engine*       engine,
             const std::string&  input_path,
             int                 loops,
             const std::string&  compare_path,
             std::promise<bool>& status);


    在该函数中，我们也通过循环处理一张图片来模拟真实的图像分类业务，
    整体的代码逻辑与 case 0 类似，
    在这里不再重复介绍。

    case 1 与 case 0 主要的区别在于对 sail::Engine 实例的处理上，下面简写为 engine。

    首先，我们在主线程中创建 engine，使用了与 case 0 不同构造函数并使用 engine 的 ”load“ 函数加载 bmodel：

        .. code-block:: cpp

           sail::Engine engine(tpu_id);

           int ret = engine.load(bmodel_path);

    不同于 case 0，使用该构造函数创建 engine 时，engine 中不会为 bmodel 的输入与输入张量创建内存，
    而是需要用户额外提供输入与输出张量，即 sail::Tensor 的实例。

    因此，在子线程中，我们根据从 engine 实例中获取的模型信息创建对应张量：

        .. code-block:: cpp

           // get handle to create input and output tensors
           sail::Handle handle = engine->get_handle();
           // allocate input and output tensors with both system and device memory
           sail::Tensor in(handle, input_shape, in_dtype, true, true);
           sail::Tensor out(handle, output_shape, out_dtype, true, true);
           std::map<std::string, sail::Tensor*> input_tensors = {{input_name, &in}};
           std::map<std::string, sail::Tensor*> output_tensors = {{output_name, &out}};


    而在模型推理时，也需要指定对应的张量，选择下面的重载函数即可：

        .. code-block:: cpp

           engine->process(graph_name, input_tensors, output_tensors);



Case 2: 多线程多模型
^^^^^^^^^^^^^^^^^^^^

    case 2 是 case 1 在模型数量上的扩展。
    在 case 1 中，我们在每个线程中都是用同一个在主线程中加载的 bmodel 做推理。
    而在 case 2 中，我们将会在 engine 中加载多个 bmodel，如下是加载模型的子线程函数：
        
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

    其它代码与 case 1 基本一致.


Case 3: 多线程多 TPU 模式
^^^^^^^^^^^^^^^^^^^^^^^^^

    case 3 是 case 1 在 TPU 数量上的扩展。
    由于 engine 是与 TPU 一一对应的。
    因此我们将对每个线程创建一个指定 tpu_id 的 engine，如下：

        .. code-block:: cpp

           // init Engine to load bmodel and allocate input and output tensors
           // one engine for one TPU
           std::vector<sail::Engine*> engines(thread_num, nullptr);
           for (int i = 0; i < thread_num; ++i) {
             engines[i] = new sail::Engine(bmodel_path, tpu_ids[i], sail::SYSIO);
           }

    其它代码与 case 1 基本一致.



