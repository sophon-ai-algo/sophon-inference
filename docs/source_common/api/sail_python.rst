SAIL Python API
===============

SAIL use "pybind11" to wrap python interfaces, support python3.5.

Basic function
______________

.. code-block:: python

        def get_available_tpu_num():
            """ Get the number of available TPUs.

            Returns
            -------
            tpu_num : int
                Number of available TPUs
            """

Data type
_________

.. code-block:: python

        # Data type for float32
        sail.Dtype.BM_FLOAT32
        # Data type for int8
        sail.Dtype.BM_INT8
        # Data type for uint8
        sail.Dtype.BM_UINT8

PaddingAtrr
_________

.. code-block:: python

        def set_stx(stx):
            """ set offset stx.

            Parameters
            ----------
            stx : int
                Offset x information relative to the origin of dst image
            """

        def set_sty(sty):
            """ set offset sty.

            Parameters
            ----------
            sty : int
                Offset y information relative to the origin of dst image
            """

        def set_w(width):
            """ set widht.

            Parameters
            ----------
            width : int
                The width after resize
            """

        def set_h(height):
            """ set height.

            Parameters
            ----------
            height : int
                The height after resize
            """

        def set_r(r):
            """ set R.

            Parameters
            ----------
            r : int
                Pixel value information of R channel
            """

        def set_g(g):
            """ set G.

            Parameters
            ----------
            g : int
                Pixel value information of G channel
            """

        def set_g(b):
            """ set B.

            Parameters
            ----------
            b : int
                Pixel value information of B channel
            """

sail.Handle
___________

.. code-block:: python

    def __init__(tpu_id):
        """ Constructor handle instance

        Parameters
        ----------
        tpu_id : int
            create handle with tpu Id
        """

    def free():
        """ free handle
        """

sail.IOMode
___________

.. code-block:: python

        # Input tensors are in system memory while output tensors are in device memory
        sail.IOMode.SYSI
        # Input tensors are in device memory while output tensors are in system memory.
        sail.IOMode.SYSO
        # Both input and output tensors are in system memory.
        sail.IOMode.SYSIO
        # Both input and output tensors are in device memory.
        sail.IOMode.DEVIO

sail.Tensor
___________

**1). Tensor**

    .. code-block:: python

        def __init__(handle, data):
            """ Constructor allocates device memory of the tensor.

            Parameters
            ----------
            handle : sail.Handle
                Handle instance
            array_data : numpy.array
                Tensor ndarray data, dtype can be np.float32, np.int8 or np.uint8
            """

        def __init__(handle, shape, dtype, own_sys_data, own_dev_data):
            """ Constructor allocates system memory and device memory of the tensor.

            Parameters
            ----------
            handle : sail.Handle
                Handle instance
            shape : tuple
                Tensor shape
            dytpe : sail.Dtype
                Data type
            own_sys_data : bool
                Indicator of whether own system memory
            own_dev_data : bool
                Indicator of whether own device memory
            """

**2). shape**

    .. code-block:: python

        def shape():
            """ Get shape of the tensor.

            Returns
            -------
            tensor_shape : list
                Shape of the tensor
            """

**3). asnumpy**

    .. code-block:: python

        def asnumpy():
            """ Get system data of the tensor.

            Returns
            -------
            data : numpy.array
                System data of the tensor, dtype can be np.float32, np.int8
                or np.uint8 with respective to the dtype of the tensor.
            """

        def asnumpy(shape):
            """ Get system data of the tensor.

            Parameters
            ----------
            shape : tuple
                Tensor shape want to get

            Returns
            -------
            data : numpy.array
                System data of the tensor, dtype can be np.float32, np.int8
                or np.uint8 with respective to the dtype of the tensor.
            """

**4). update_data**

    .. code-block:: python

        def update_data(data):
            """ Update system data of the tensor. The data size should not exceed
                the tensor size, and the tensor shape will not be changed.

            Parameters
            -------
            data : numpy.array
                Data.
            """

**5). scale_from**

    .. code-block:: python

        def scale_from(data, scale):
            """ Scale data to tensor in system memory.

            Parameters
            -------
            data : numpy.array with dtype of float32
                Data.
            scale : float32
                Scale value.
            """

**6). scale_to**

    .. code-block:: python

        def scale_from(scale):
            """ Scale tensor to data in system memory.

            Parameters
            -------
            scale : float32
                Scale value.

            Returns
            -------
            data : numpy.array with dtype of float32
                Data.
            """

        def scale_from(scale, shape):
            """ Scale tensor to data in system memory.

            Parameters
            -------
            scale : float32
                Scale value.
            shape : tuple
                Tensor shape want to get

            Returns
            -------
            data : numpy.array with dtype of float32
                Data.
            """

**7). dtype**

    .. code-block:: python

        def dtype():
            """ Get data type of the tensor.

            Returns
            -------
            dtype : sail.Dtype
                Data type of the tensor
            """

**8). reshape**

    .. code-block:: python

        def reshape(shape):
            """ Reset shape of the tensor.

            Parameters
            -------
            shape : list
                New shape of the tensor
            """

**9). own_sys_data**

    .. code-block:: python

        def own_sys_data():
            """ Judge if the tensor owns data pointer in system memory.

            Returns
            -------
            judge_ret : bool
                True for owns data pointer in system memory.
            """

**10). own_dev_data**

    .. code-block:: python

        def own_dev_data():
            """ Judge if the tensor owns data in device memory.

            Returns
            -------
            judge_ret : bool
                True for owns data in device memory.
            """

**11). sync_s2d**

    .. code-block:: python

        def sync_s2d():
            """ Copy data from system memory to device memory.
            """

        def sync_s2d(size):
            """ Copy data from system memory to device memory with specified size.

            Parameters
            ----------
            size : int
                Byte size to be copied
            """

**12). sync_d2s**

    .. code-block:: python

        def sync_d2s():
            """ Copy data from device memory to system memory.
            """

        def sync_d2s(size):
            """ Copy data from device memory to system memory with specified size.

            Parameters
            ----------
            size : int
                Byte size to be copied
            """

sail.Engine
___________

**1). Engine**

    .. code-block:: python

        def __init__(tpu_id):
            """ Constructor does not load bmodel.

            Parameters
            ----------
            tpu_id : int
                TPU ID. You can use bm-smi to see available IDs
            """

        def __init__(handle):
            """ Constructor does not load bmodel.

            Parameters
            ----------
            hanle : Handle
               A Handle instance
            """

        def __init__(bmodel_path, tpu_id, mode):
            """ Constructor loads bmodel from file.

            Parameters
            ----------
            bmodel_path : str
                Path to bmodel
            tpu_id : int
                TPU ID. You can use bm-smi to see available IDs
            mode : sail.IOMode
                Specify the input/output tensors are in system memory
                or device memory
            """

        def __init__(bmodel_path, handle, mode):
            """ Constructor loads bmodel from file.

            Parameters
            ----------
            bmodel_path : str
                Path to bmodel
            hanle : Handle
               A Handle instance
            mode : sail.IOMode
                Specify the input/output tensors are in system memory
                or device memory
            """

        def __init__(bmodel_bytes, bmodel_size, tpu_id, mode):
            """ Constructor using default input shapes with bmodel which
            loaded in memory

            Parameters
            ----------
            bmodel_bytes : bytes
                Bytes of  bmodel in system memory
            bmodel_size : int
                Bmodel byte size
            tpu_id : int
                TPU ID. You can use bm-smi to see available IDs
            mode : sail.IOMode
                Specify the input/output tensors are in system memory
                or device memory
            """

        def __init__(bmodel_bytes, bmodel_size, handle, mode):
            """ Constructor using default input shapes with bmodel which
            loaded in memory

            Parameters
            ----------
            bmodel_bytes : bytes
                Bytes of  bmodel in system memory
            bmodel_size : int
                Bmodel byte size
            hanle : Handle
               A Handle instance
            mode : sail.IOMode
                Specify the input/output tensors are in system memory
                or device memory
            """

**2). get_handle**

    .. code-block:: python

        def get_handle():
            """ Get Handle instance.

            Returns
            -------
            handle: sail.Handle
               Handle instance
            """

**3). load**

    .. code-block:: python

        def load(bmodel_path):
            """ Load bmodel from file.

            Parameters
            ----------
            bmodel_path : str
                Path to bmodel
            """

        def load(bmodel_bytes, bmodel_size):
            """ Load bmodel from file.

            Parameters
            ----------
            bmodel_bytes : bytes
                Bytes of  bmodel in system memory
            bmodel_size : int
                Bmodel byte size
            """

**4). get_graph_names**

    .. code-block:: python

        def get_graph_names():
            """ Get all graph names in the loaded bmodels.

            Returns
            -------
            graph_names : list
                Graph names list in loaded context
            """

**5). set_io_mode**

    .. code-block:: python

        def set_io_mode(graph_name, mode):
            """ Set IOMode for a graph.

            Parameters
            ----------
            graph_name: str
                The specified graph name
            mode : sail.IOMode
                Specified io mode
            """

**6). get_input_names**

    .. code-block:: python

        def get_input_names(graph_name):
            """ Get all input tensor names of the specified graph.

            Parameters
            ----------
            graph_name : str
                Specified graph name

            Returns
            -------
            input_names : list
                All the input tensor names of the graph
            """

**7). get_output_names**

    .. code-block:: python

        def get_output_names(graph_name):
            """ Get all output tensor names of the specified graph.

            Parameters
            ----------
            graph_name : str
                Specified graph name

            Returns
            -------
            input_names : list
                All the output tensor names of the graph
            """

**8). get_max_input_shapes**

    .. code-block:: python

        def get_max_input_shapes(graph_name):
            """ Get max shapes of input tensors in a graph.
                For static models, the max shape is fixed and it should not be changed.
                For dynamic models, the tensor shape should be smaller than or equal to
                the max shape.

            Parameters
            ----------
            graph_name : str
                The specified graph name

            Returns
            -------
            max_shapes : dict {str : list}
                Max shape of the input tensors
            """

**9). get_input_shape**

    .. code-block:: python

        def get_input_shape(graph_name, tensor_name):
            """ Get the maximum dimension shape of an input tensor in a graph.
                There are cases that there are multiple input shapes in one input name, 
                This API only returns the maximum dimension one for the memory allocation 
                in order to get the best performance.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            tensor_name : str
                The specified input tensor name

            Returns
            -------
            tensor_shape : list
                The maxmim dimension shape of the tensor
            """

**10). get_max_output_shapes**

    .. code-block:: python

        def get_max_output_shapes(graph_name):
            """ Get max shapes of input tensors in a graph.
                For static models, the max shape is fixed and it should not be changed.
                For dynamic models, the tensor shape should be smaller than or equal to
                the max shape.

            Parameters
            ----------
            graph_name : str
                The specified graph name

            Returns
            -------
            max_shapes : dict {str : list}
                Max shape of the output tensors
            """

**11). get_output_shape**

    .. code-block:: python

        def get_output_shape(graph_name, tensor_name):
            """ Get the shape of an output tensor in a graph.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            tensor_name : str
                The specified output tensor name

            Returns
            -------
            tensor_shape : list
                The shape of the tensor
            """

**12). get_input_dtype**

    .. code-block:: python

        def get_input_dtype(graph_name, tensor_name)
            """ Get scale of an input tensor. Only used for int8 models.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            tensor_name : str
                The specified output tensor name

            Returns
            -------
            scale: sail.Dtype
                Data type of the input tensor
            """

**13). get_output_dtype**

    .. code-block:: python

        def get_output_dtype(graph_name, tensor_name)
            """ Get scale of an output tensor. Only used for int8 models.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            tensor_name : str
                The specified output tensor name

            Returns
            -------
            scale: sail.Dtype
                Data type of the output tensor
            """

**14). get_input_scale**

    .. code-block:: python

        def get_input_scale(graph_name, tensor_name)
            """ Get scale of an input tensor. Only used for int8 models.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            tensor_name : str
                The specified output tensor name

            Returns
            -------
            scale: float32
                Scale of the input tensor
            """

**15). get_output_scale**

    .. code-block:: python

        def get_output_scale(graph_name, tensor_name)
            """ Get scale of an output tensor. Only used for int8 models.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            tensor_name : str
                The specified output tensor name

            Returns
            -------
            scale: float32
                Scale of the output tensor
            """

**16). process**

    .. code-block:: python

        def process(graph_name, input_tensors):
            """ Inference with provided system data of input tensors.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            input_tensors : dict {str : numpy.array}
                Data of all input tensors in system memory

            Returns
            -------
            output_tensors : dict {str : numpy.array}
                Data of all output tensors in system memory
            """

        def process(graph_name, input_tensors, output_tensors):
            """ Inference with provided input and output tensors.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            input_tensors : dict {str : sail.Tensor}
                Input tensors managed by user
            output_tensors : dict {str : sail.Tensor}
                Output tensors managed by user
            """

        def process(graph_name, input_tensors, input_shapes, output_tensors):
            """ Inference with provided input tensors, input shapes and output tensors.

            Parameters
            ----------
            graph_name : str
                The specified graph name
            input_tensors : dict {str : sail.Tensor}
                Input tensors managed by user
            input_shapes : dict {str : list}
                Shapes of all input tensors
            output_tensors : dict {str : sail.Tensor}
                Output tensors managed by user
            """

sail.BMImage
____________

**1). BMImage**

    .. code-block:: python

        def __init__():
            """ Constructor.
            """

**2). width**

    .. code-block:: python

        def width():
            """ Get the img width.

            Returns
            ----------
            width : int
               The width of img
            """

**3). height**

    .. code-block:: python

        def height():
            """ Get the img height.

            Returns
            ----------
            height : int
               The height of img
            """

**4). format**

    .. code-block:: python

        def format():
            """ Get the img format.

            Returns
            ----------
            format : bm_image_format_ext
               The format of img
            """

sail.Decoder
____________

**1). Decoder**

    .. code-block:: python

        def __init__(file_path, compressed=True, tpu_id=0):
            """ Constructor.

            Parameters
            ----------
            file_path : str
               Path or rtsp url to the video/image file
            compressed : bool, default: True
               Whether the format of decoded output is compressed NV12.
            tpu_id: int, default: 0
               ID of TPU, there may be more than one TPU for PCIE mode.
            """

**2). is_opened**

    .. code-block:: python

        def is_opened():
            """ Judge if the source is opened successfully.

            Returns
            ----------
            judge_ret : bool
                True for success and False for failure
            """

**3). read**

    .. code-block:: python

        def read(handle, image):
            """ Read an image from the Decoder.

            Parameters
            ----------
            handle : sail.Handle
                Handle instance
            image : sail.BMImage
                BMImage instance
            Returns
            ----------
            judge_ret : int
                0 for success and others for failure
            """

sail.Bmcv
_________

**1). Bmcv**

    .. code-block:: python

        def __init__(handle):
            """ Constructor.

            Parameters
            ----------
            handle : sail.Handle
                Handle instance
            ""

**2). bm_image_to_tensor**

    .. code-block:: python

        def bm_image_to_tensor(image):
            """ Convert image to tensor.

            Parameters
            ----------
            image : sail.BMImage
                BMImage instance

            Returns
            -------
            tensor : sail.Tensor
                Tensor instance
            """

        def bm_image_to_tensor(image, tensor):
            """ Convert image to tensor.

            Parameters
            ----------
            image : sail.BMImage
                BMImage instance

            tensor : sail.Tensor
                Tensor instance
            """
**3). tensor_to_bm_image**

    .. code-block:: python

        def tensor_to_bm_image(tensor):
            """ Convert tensor to image.

            Parameters
            ----------
            tensor : sail.Tensor
                Tensor instance

            Returns
            -------
            image : sail.BMImage
                BMImage instance
            """

**4). crop_and_resize**

    .. code-block:: python

        def crop_and_resize(input, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h):
            """ Crop then resize an image.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            crop_x0 : int
                Start point x of the crop window
            crop_y0 : int
                Start point y of the crop window
            crop_w : int
                Width of the crop window
            crop_h : int
                Height of the crop window
            resize_w : int
                Target width
            resize_h : int
                Target height

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

**5). crop**

    .. code-block:: python

        def crop(input, crop_x0, crop_y0, crop_w, crop_h):
            """ Crop an image with given window.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            crop_x0 : int
                Start point x of the crop window
            crop_y0 : int
                Start point y of the crop window
            crop_w : int
                Width of the crop window
            crop_h : int
                Height of the crop window

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

**6). resize**

    .. code-block:: python

        def resize(input, resize_w, resize_h):
            """ Resize an image with interpolation of INTER_NEAREST.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            resize_w : int
                Target width
            resize_h : int
                Target height

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

**7). vpp_crop_and_resize**

    .. code-block:: python

        def vpp_crop_and_resize(input, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h):
            """ Crop then resize an image using vpp.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            crop_x0 : int
                Start point x of the crop window
            crop_y0 : int
                Start point y of the crop window
            crop_w : int
                Width of the crop window
            crop_h : int
                Height of the crop window
            resize_w : int
                Target width
            resize_h : int
                Target height

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

**8). vpp_crop_and_resize_padding**

    .. code-block:: python

        def vpp_crop_and_resize_padding(input, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, padding):
            """ Crop then resize an image using vpp.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            crop_x0 : int
                Start point x of the crop window
            crop_y0 : int
                Start point y of the crop window
            crop_w : int
                Width of the crop window
            crop_h : int
                Height of the crop window
            resize_w : int
                Target width
            resize_h : int
                Target height
            padding : PaddingAtrr
                padding info

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

**9). vpp_crop**

    .. code-block:: python

        def vpp_crop(input, crop_x0, crop_y0, crop_w, crop_h):
            """ Crop an image with given window using vpp.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            crop_x0 : int
                Start point x of the crop window
            crop_y0 : int
                Start point y of the crop window
            crop_w : int
                Width of the crop window
            crop_h : int
                Height of the crop window

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

**10). vpp_crop_padding**

    .. code-block:: python

        def vpp_crop_padding(input, crop_x0, crop_y0, crop_w, crop_h, padding):
            """ Crop an image with given window using vpp.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            crop_x0 : int
                Start point x of the crop window
            crop_y0 : int
                Start point y of the crop window
            crop_w : int
                Width of the crop window
            crop_h : int
                Height of the crop window
            padding : PaddingAtrr
                padding info

            Returns
            ----------
            output : sail.BMImage
                Output image
            """



**11). vpp_resize**

    .. code-block:: python

        def vpp_resize(input, resize_w, resize_h):
            """ Resize an image with interpolation of INTER_NEAREST using vpp.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            resize_w : int
                Target width
            resize_h : int
                Target height

            Returns
            ----------
            output : sail.BMImage
                Output image
            """
         def vpp_resize(input, output, resize_w, resize_h):
            """ Resize an image with interpolation of INTER_NEAREST using vpp.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            output : sail.BMImage
                Output image
            resize_w : int
                Target width
            resize_h : int
                Target height
            """

**12). vpp_resize_padding**

    .. code-block:: python

        def vpp_resize_padding(input, resize_w, resize_h, padding):
            """ Resize an image with interpolation of INTER_NEAREST using vpp.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            resize_w : int
                Target width
            resize_h : int
                Target height
            padding : PaddingAtrr
                padding info

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

**13). warp**

    .. code-block:: python

        def warp(input, matrix):
            """ Applies an affine transformation to an image.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            matrix: 2d list
                2x3 transformation matrix

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

**14). convert_to**

    .. code-block:: python

        def convert_to(input, alpha_beta):
            """ Applies a linear transformation to an image.

            Parameters
            ----------
            input : sail.BMImage
                Input image
            alpha_beta: tuple
                (a0, b0), (a1, b1), (a2, b2) factors

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

**15). yuv2bgr**

    .. code-block:: python

        def yuv2bgr(input):
            """ Convert an image from YUV to BGR.

            Parameters
            ----------
            input : sail.BMImage
                Input image

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

**16). vpp_convert**

    .. code-block:: python

        def vpp_convert(input):
            """ Convert an image to BGR PLANAR format using vpp.

            Parameters
            ----------
            input : sail.BMImage
                Input image

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

**17). convert**

    .. code-block:: python

        def convert(input):
            """ Convert an image to BGR PLANAR format.

            Parameters
            ----------
            input : sail.BMImage
                Input image

            Returns
            ----------
            output : sail.BMImage
                Output image
            """

**18). rectangle**

    .. code-block:: python

        def rectangle(image, x0, y0, w, h, color, thickness=1):
            """ Draw a rectangle on input image.

            Parameters
            ----------
            image : sail.BMImage
                Input image
            x0 : int
                Start point x of rectangle
            y0 : int
                Start point y of rectangle
            w : int
                Width of rectangle
            h : int
                Height of rectangle
            color : tuple
                Color of rectangle
            thickness : int
                Thickness of rectangle

            Returns
            ----------
            process_status : int
                0 for success and others for failure
            """

**19). imwrite**

    .. code-block:: python

        def imwrite(file_name, image):
            """ Save the image to the specified file.

            Parameters
            ----------
            file_name : str
                Name of the file
            output : sail.BMImage
                Image to be saved

            Returns
            ----------
            process_status : int
                0 for success and others for failure
            """

**20). get_handle**

    .. code-block:: python

        def get_handle():
            """ Get Handle instance.

            Returns
            -------
            handle: sail.Handle
               Handle instance
        """
