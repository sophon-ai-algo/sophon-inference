Python 代码解析
_______________

Case 0: 基础示例程序
^^^^^^^^^^^^^^^^^^^^

    在该示例程序中，我们封装了一个函数，完成图像分类的过程，如下：

        .. code-block:: python

           def inference(bmodel_path, input_path, loops, tpu_id, compare_path):
           """ Do inference of a model in a thread.

           Args:
             bmodel_path: Path to bmodel
             input_path: Path to input image.
             loops: Number of loops to run
             compare_path: Path to correct result file
             status: Status of comparison

           Returns:
             True for success and False for failure.
           """

    在该函数中，我们通过循环处理同一张图像来模拟真实的图像分类业务。

        .. code-block:: python

           # pipeline of inference
           for i in range(loops):
             # read image and preprocess
             image = preprocess(input_path).astype(np.float32)
             # inference with fp32 input and output
             # data scale(input: fp32 to int8, output: int8 to fp32) is done inside
             # for int8 model
             output = engine.process(graph_name, {input_name:image})
             # postprocess
             result = postprocess(output[output_name])
             # print result
             # ...

    
    如上代码所示，在每次循环中，我们首先使用opencv 的”imread“函数解码图片。
    然后对图像进行预处理，获取bmodel 的输入张量。
    接着驱动TPU 进行推理。最后对TPU 输出的张量进行后处理，获得最终的结果(top-5)。

    在该示例程序中，图片解码和预处理的实现中直接调用了opencv 的相关函数，
    同时后处理比较简单，只计算了top-5 分类结果。
    因此你可以直接参考相关代码，下面我们主要介绍模型推理部分。

    bmodel_path 是本例中用到的resnet50 bmodel 的路径，fp32 或int8 皆可。
    我们使用该 bmodel 初始化了一个sail.Engine 的实例，
    该实例中保存了模型的基础信息，将作为我们后续模型推理的载体。
    如果你的机器上有多个TPU，那么可以通过tpu_id 指定使用哪个TPU进行推理，编号从0 开始。

        .. code-block:: python

           engine = sail.Engine(bmodel_path, tpu_id, sail.SYSIO)

    我们可以通过该实例提供的方法获取模型的属性，代码如下：

        .. code-block:: python

           graph_name = engine.get_graph_names()[0]
           input_name = engine.get_input_names(graph_name)[0]
           input_shape = engine.get_input_shape(graph_name, input_name)
           output_name = engine.get_output_names(graph_name)[0]
           output_shape = engine.get_output_shape(graph_name, output_name)
           out_dtype = engine.get_output_dtype(graph_name, output_name);

    graph_name 代表bmodel 中我们要使用的具体模型的名称，在该示例程序中，bmodel 中只包含一个模型。
    input_name 代表我们使用的具体模型中的输入张量名称，在该模型中，只有一个输入张量。
    output_name 代表我们使用的具体模型中的输出张量名称，在该模型中，只有一个输出张量。
    input_shape 和output_shape 代表指定张量的尺寸。in_dtype 和 out_dtype 代表指定张量的数据类型。

Case 1: 单模型多线程
^^^^^^^^^^^^^^^^^^^^

    在case 0 中，我们使用了bmodel 初始化了一个sail.Engine 的实例，
    模型推理的过程是在主线程中进行的。
    在case 1 中，我们将展示如果在多个线程中使用同一个sail.Engine 实例做模型推理。

    我们定义了一个”thread_infer“函数来实现子线程中的模型推理逻辑，如下：

        .. code-block:: python

           def thread_infer(thread_id, engine, input_path, loops, compare_path, status):
           """ Do inference of a model in a thread.

           Args:
             thread_id: ID of the thread
             engine: An sail.Engine instance
             input_path: Path to input image file
             loops: Number of loops to run
             compare_path: Path to correct result file
             status: Status of comparison

           Returns:
             None.
           """

    在该函数中，我们也通过循环处理一张图片来模拟真实的图像分类业务，
    整体的代码逻辑与 case 0 类似，在这里不再重复介绍。

    case 1 与case 0 主要的区别在于对sail.Engine 实例的处理上，下面简写为engine。

    首先，我们在主线程中创建 engine，使用了与 case 0 不同构造函数并使用 engine 的 ”load“ 函数加载 bmodel：

        .. code-block:: python

           engine = sail.Engine(ARGS.tpu_id)
           engine.load(ARGS.bmodel)

    不同于case 0，使用该构造函数创建engine 时，
    engine 中不会为bmodel 的输入与输入张量创建内存，
    而是需要用户额外提供输入与输出张量，即sail.Tensor 的实例。

    因此，在子线程中，我们根据从 engine 实例中获取的模型信息创建对应张量：

        .. code-block:: python
 
           # get handle to create input and output tensors
           handle = engine.get_handle()
           input = sail.Tensor(handle, input_shape, in_dtype, True, True)
           output = sail.Tensor(handle, output_shape, out_dtype, True, True)
           input_tensors = {input_name:input}
           ouptut_tensors = {output_name:output}

    而在模型推理时，也需要指定对应的张量，选择下面的重载函数即可：

        .. code-block:: python

           engine.process(graph_name, input_tensors, ouptut_tensors)

Case 2: 多线程多模型
^^^^^^^^^^^^^^^^^^^^

    case 2 是 case 1 在模型数量上的扩展。
    在case 1 中，我们在每个线程中都是用同一个在主线程中加载的 bmodel 做推理。
    而在case 2 中，我们将会在 engine 中加载多个bmodel，如下是加载模型的子线程函数：

        .. code-block:: python
   
           def thread_load(thread_id, engine, bmodel_path):
           """ Load a model in a thread.

           Args:
             thread_id: ID of the thread.
             engine: An sail.Engine instance.
             bmodel_path: Path to bmodel.

           Returns:
             None.
           """
           ret = engine.load(bmodel_path)
           if ret == 0:
             graph_name = engine.get_graph_names()[-1]
             print("Thread {} load {} successfully.".format(thread_id, graph_name))
           


    其它代码与 case 1 基本一致.
 
Case 3: 多线程多 TPU 模式
^^^^^^^^^^^^^^^^^^^^^^^^^

    case 3 是 case 1 在 TPU 数量上的扩展。由于 engine 是与TPU 一一对应的。
    因此我们将对每个线程创建一个指定 tpu_id 的 engine，如下：

        .. code-block:: python

           # init Engine to load bmodel and allocate input and output tensors
           # one engine for one TPU
           engines = list()
           thread_num = len(ARGS.tpu_id)
           for i in range(thread_num):
             engines.append(sail.Engine(ARGS.bmodel, ARGS.tpu_id[i], sail.SYSIO))

    其它代码与 case 1 基本一致.


