Python Codes Explanation
________________________

Case 0: simplest case
^^^^^^^^^^^^^^^^^^^^^

    In case 0, we drive a bmodel converted from resnet50 to classfy an image.
    Whole procedure is composed of four steps: initializing, preprocessing, inference, postprocessing,
    which corresponds to four function calls.

    Initializing:
        .. code-block:: python

           engine = sail.Engine(bmodel_path, tpu_id, sail.SYSIO)

    Preprocessing:
        .. code-block:: python

           image = preprocess(input_path).astype(np.float32)

    Inference:
        .. code-block:: python

           output = engine.process(graph_name, {input_name:image})

    Postprocessing:
        .. code-block:: python

           result = postprocess(output[output_name])


Case 1: multi-thread implementation of case 0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    In case 1, we will show the multi-thread programming mode of sail.Engine.
    Simplely, one bmodel was loaded by one sail.Engine instance, while input/output tensors are managed outside this sail.Engine instance in different threads.

    We loaded the bmodel into sail.Engine instance after constucor, not in the constructor:
        .. code-block:: python
 
           # init Engine
           engine = sail.Engine(ARGS.tpu_id)
           # load bmodel without builtin input and output tensors
           # each thread manage its input and output tensors
           engine.load(ARGS.bmodel)


    In each thread, we seperately managed the input and output tensors.
    While in case 0, these tensors were managed automatically in the sail.Engine instance.

        .. code-block:: python
 
           # get handle to create input and output tensors
           handle = engine.get_handle()
           input = sail.Tensor(handle, input_shape, in_dtype, True, True)
           output = sail.Tensor(handle, output_shape, out_dtype, True, True)
           input_tensors = {input_name:input}
           ouptut_tensors = {output_name:output}


Case 2: multi-thread with multiple models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    In case 2, multiple bmodels could be fed.
    The program can create multiple threads to load different bmodels.
    The loading function and its caller are as follows:

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
           

           # load bmodel without builtin input and output tensors
           # each thread manage its input and output tensors
           for i in range(thread_num):
             threads.append(threading.Thread(target=thread_load,
                                             args=(i, engine, ARGS.bmodel[i])))

    Other codes are almost the same as case 1.
 
Case 3: multi-thread with multiple TPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    In Case 3, we will exploit multiple TPUs to do inference. 
    While the SAIL:Engine instance is bound to device, 
    we should initialize multiple sail.Engine instances for each TPU.

        .. code-block:: python

           # init Engine to load bmodel and allocate input and output tensors
           # one engine for one TPU
           engines = list()
           thread_num = len(ARGS.tpu_id)
           for i in range(thread_num):
             engines.append(sail.Engine(ARGS.bmodel, ARGS.tpu_id[i], sail.SYSIO))

    Other codes are almost the same as case 1.
