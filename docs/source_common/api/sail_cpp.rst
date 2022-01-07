SAIL C++ API
============

Basic function
______________

**1). get_available_tpu_num**
    .. code-block:: c

       /** @brief Get the number of available TPUs.
        *
        *  @return Number of available TPUs.
        */
       int get_available_tpu_num();

Data type
_________

**1). bm_data_type_t**
    .. code-block:: c

       enum bm_data_type_t {
         BM_FLOAT32,     // float32
         BM_FLOAT16,     // not supported for now
         BM_INT8,        // int8
         BM_UINT8        // unsigned int8
       };

PaddingAtrr
______

**1). PaddingAtrr**
    .. code-block:: c
       class PaddingAtrr {
       public:
           PaddingAtrr(){};
           ~PaddingAtrr(){};
           void set_stx(unsigned int stx);
           void set_sty(unsigned int sty);
           void set_w(unsigned int w);
           void set_h(unsigned int h);
           void set_r(unsigned int r);
           void set_g(unsigned int g);
           void set_b(unsigned int b);

           unsigned int    dst_crop_stx; // Offset x information relative to the origin of dst image
           unsigned int    dst_crop_sty; // Offset y information relative to the origin of dst image
           unsigned int    dst_crop_w;   // The width after resize
           unsigned int    dst_crop_h;   // The height after resize
           unsigned char   padding_r;    // Pixel value information of R channel
           unsigned char   padding_g;    // Pixel value information of G channel
           unsigned char   padding_b;    // Pixel value information of B channel
       };

Handle
______

**1). Handle Constructor**
    .. code-block:: c

       /**
        * @brief Constructor using existed bm_handle_t.
        *
        * @param handle A bm_handle_t
        */
       Handle(bm_handle_t handle);

       /**
        * @brief Constructor with device id.
        *
        * @param dev_id Device id
        */
       Handle(int dev_id);

**2). data**
    .. code-block:: c

       /**
        *  @brief Get inner bm_handle_t.
        *
        *  @return Inner bm_handle_t
        */
       bm_handle_t data();

Tensor
______

**1). Tensor Constructor**
    .. code-block:: c

       /**
        * @brief Common constructor.
        * @detail
        *  case 0: only allocate system memory
        *          (handle, shape, dtype, true, false)
        *  case 1: only allocate device memory
        *          (handle, shape, dtype, false, true)
        *  case 2: allocate system memory and device memory
        *          (handle, shape, dtype, true, true)
        *
        * @param handle       Handle instance
        * @param shape        Shape of the tensor
        * @param own_sys_data Indicator of whether own system memory.
        * @param own_dev_data Indicator of whether own device memory.
        */
       explicit Tensor(
           Handle                  handle,
           const std::vector<int>& shape,
           bm_data_type_t          dtype,
           bool                    own_sys_data,
           bool                    own_dev_data);

       /**
        *  @brief Copy constructor.
        *
        *  @param tensor A Tensor instance
        */
       Tensor(const Tensor& tensor);

**2). Tensor Assign Function**
    .. code-block:: c

       /**
        * @brief Assignment function.
        *
        * @param tensor A Tensor instance
        * @return A Tensor instance
        */
       Tensor& operator=(const Tensor& tensor);

**3). shape**
    .. code-block:: c

       /**
        * @brief Get shape of the tensor.
        *
        * @return Shape of the tensor
        */
       const std::vector<int>& shape() const;

**4). dtype**
    .. code-block:: c

       /**
        * @brief Get data type of the tensor.
        *
        * @return Data type of the tensor
        */
       void dtype();

**5). reshape**
    .. code-block:: c

       /**
        * @brief Reset shape of the tensor.
        *
        * @param shape Shape of the tensor
        */
       void reshape(const std::vector<int>& shape);

**6). own_sys_data**
    .. code-block:: c

       /**
        * @brief Judge if the tensor owns data in system memory.
        *
        * @return True for owns data in system memory.
        */
       bool own_sys_data();

**7). own_dev_data**
    .. code-block:: c

       /**
        * @brief Judge if the tensor owns data in device memory.
        *
        * @return True for owns data in device memory.
        */
       bool own_dev_data();

**8). sys_data**
    .. code-block:: c

       /**
        * @brief Get data pointer in system memory of the tensor.
        *
        * @return Data pointer in system memory of the tensor
        */
       void* sys_data();

**9). dev_data**
    .. code-block:: c

       /**
        * @brief Get pointer to device memory of the tensor.
        *
        * @return Pointer to device memory of the tensor
        */
       bm_device_mem_t* dev_data();

**10). reset_sys_data**
    .. code-block:: c

       /**
        * @brief Reset data pointer in system memory of the tensor.
        *
        * @param data  Data pointer in system memory of the tensor
        * @param shape Shape of the data
        */
       void reset_sys_data(
           void*              data,
           std::vector<int>& shape);

**11). reset_dev_data**
    .. code-block:: c

       /**
        * @brief Reset pointer to device memory of the tensor.
        *
        * @param data Pointer to device memory
        */
       void reset_dev_data(bm_device_mem_t* data);

**12). sync_s2d**
    .. code-block:: c

       /**
        * @brief Copy data from system memory to device memory.
        */
       void sync_s2d();

       /**
        * @brief Copy data from system memory to device memory with specified size.
        *
        * @param size Byte size to be copied
        */
       void sync_s2d(int size);

**13). sync_d2s**
    .. code-block:: c

       /**
        * @brief Copy data from device memory to system memory.
        */
       void sync_d2s();

       /**
        * @brief Copy data from device memory to system memory with specified size.
        *
        * @param size Byte size to be copied
        */
       void sync_d2s(int size);

**14). free**
    .. code-block:: c

       /**
        * @brief Free system and device memroy of the tensor.
        */
       void free();

IOMode
______

**1). IOMode**
    .. code-block:: c

        enum IOMode {
          /// Input tensors are in system memory while output tensors are
          /// in device memory.
          SYSI,
          /// Input tensors are in device memory while output tensors are
          /// in system memory.
          SYSO,
          /// Both input and output tensors are in system memory.
          SYSIO,
          /// Both input and output tensors are in device memory.
          DEVIO
        };

Engine
______

**1). Engine Constructor**
    .. code-block:: c

       /**
        * @brief Constructor does not load bmodel.
        *
        * @param tpu_id TPU ID. You can use bm-smi to see available IDs.
        */
        Engine(int tpu_id);

       /**
        * @brief Constructor does not load bmodel.
        *
        * @param handle Handle created elsewhere.
        */
       Engine(const Handle&   handle);

       /**
        * @brief Constructor loads bmodel from file.
        *
        * @param bmodel_path Path to bmodel
        * @param tpu_id      TPU ID. You can use bm-smi to see available IDs.
        * @param mode        Specify the input/output tensors are in system memory
        *                   or device memory
        */
       Engine(
           const std::string& bmodel_path,
           int                tpu_id,
           IOMode             mode);

       /**
        * @brief Constructor loads bmodel from file.
        *
        * @param bmodel_path Path to bmodel
        * @param handle      Handle created elsewhere.
        * @param mode        Specify the input/output tensors are in system memory
        *                    or device memory
        */
       Engine(
           const std::string& bmodel_path,
           const Handle&      handle,
           IOMode             mode);

       /**
        * @brief Constructor loads bmodel from system memory.
        *
        * @param bmodel_ptr  Pointer to bmodel in system memory
        * @param bmodel_size Byte size of bmodel in system memory
        * @param tpu_id      TPU ID. You can use bm-smi to see available IDs.
        * @param mode        Specify the input/output tensors are in system memory
        *                   or device memory
        */
        Engine(
            const void* bmodel_ptr,
            size_t      bmodel_size,
            int         tpu_id,
            IOMode      mode);

       /**
        * @brief Constructor loads bmodel from system memory.
        *
        * @param bmodel_ptr  Pointer to bmodel in system memory
        * @param bmodel_size Byte size of bmodel in system memory
        * @param handle      Handle created elsewhere.
        * @param mode        Specify the input/output tensors are in system memory
        *                    or device memory
        */
       Engine(
           const void*        bmodel_ptr,
           size_t             bmodel_size,
           const Handle&      handle,
           IOMode             mode);

       /**
        * @brief Copy constructor.
        *
        * @param other An other Engine instance.
        */
       Engine(const Engine& other);

**2). Engine Assign Function**
    .. code-block:: c

       /**
        * @brief Assignment function.
        *
        * @param other An other Engine instance.
        * @return Reference of a Engine instance.
        */
       Engine<Dtype>& operator=(const Engine& other);

**3). get_handle**
    .. code-block:: c

       /**
        * @brief Get Handle instance.
        *
        * @return Handle instance
        */
       Handle get_handle();

**4). load**
    .. code-block:: c

       /**
        * @brief Load bmodel from file.
        *
        * @param bmodel_path Path to bmodel
        * @return Program state
        *     @retval true  Success
        *     @retval false Failure
        */
       bool load(const std::string& bmodel_path);

       /**
        * @brief Load bmodel from system memory.
        *
        * @param bmodel_ptr  Pointer to bmodel in system memory
        * @param bmodel_size Byte size of bmodel in system memory
        * @return Program state
        *     @retval true  Success
        *    @retval false Failure
        */
       bool load(const void* bmodel_ptr, size_t bmodel_size);

**5). get_graph_names**
    .. code-block:: c

       /**
        * @brief Get all graph names in the loaded bomodels.
        *
        * @return All graph names
        */
       std::vector<std::string> get_graph_names();

**6). set_io_mode**
    .. code-block:: c

       /**
        * @brief Set IOMode for a graph.
        *
        * @param graph_name The specified graph name
        * @param mode The specified IOMode
        */
       void set_io_mode(
         const std::string& graph_name,
         IOMode             mode);

**7). get_input_names**
    .. code-block:: c

       /**
        * @brief Get all input tensor names of the specified graph.
        *
        * @param graph_name The specified graph name
        * @return All the input tensor names of the graph
        */
       std::vector<std::string> get_input_names(const std::string& graph_name);

**8). get_output_names**
    .. code-block:: c

       /**
        * @brief Get all output tensor names of the specified graph.
        *
        * @param graph_name The specified graph name
        * @return All the output tensor names of the graph
        */
       std::vector<std::string> get_output_names(const std::string& graph_name);

**9). get_max_input_shapes**
    .. code-block:: c

       /**
        * @brief Get max shapes of input tensors in a graph.
        *
        * For static models, the max shape is fixed and it should not be changed.
        * For dynamic models, the tensor shape should be smaller than or equal to
        * the max shape.
        *
        * @param graph_name The specified graph name
        * @return Max shape of input tensors
        */
       std::map<std::string, std::vector<int>> get_max_input_shapes(
           const std::string& graph_name);

**10). get_input_shape**
    .. code-block:: c

       /**
        * @brief Get the shape of an input tensor in a graph.
        *
        * @param graph_name  The specified graph name
        * @param tensor_name The specified tensor name
        * @return The shape of the tensor
        */
       std::vector<int> get_input_shape(
           const std::string& graph_name,
           const std::string& tensor_name);

**11). get_max_output_shapes**
    .. code-block:: c

       /**
        * @brief Get max shapes of output tensors in a graph.
        *
        * For static models, the max shape is fixed and it should not be changed.
        * For dynamic models, the tensor shape should be smaller than or equal to
        * the max shape.
        *
        * @param graph_name The specified graph name
        * @return Max shape of output tensors
        */
       std::map<std::string, std::vector<int>> get_max_output_shapes(
           const std::string& graph_name);

**12). get_output_shape**
    .. code-block:: c

       /**
        * @brief Get the shape of an output tensor in a graph.
        *
        * @param graph_name  The specified graph name
        * @param tensor_name The specified tensor name
        * @return The shape of the tensor
        */
       std::vector<int> get_output_shape(
           const std::string& graph_name,
           const std::string& tensor_name);

**13). get_input_dtype**
    .. code-block:: c

       /**
        * @brief Get data type of an input tensor. Refer to bmdef.h as following.
        *   typedef enum {
        *     BM_FLOAT32 = 0,
        *     BM_FLOAT16 = 1,
        *     BM_INT8 = 2,
        *     BM_UINT8 = 3,
        *     BM_INT16 = 4,
        *     BM_UINT16 = 5,
        *     BM_INT32 = 6,
        *     BM_UINT32 = 7
        *   } bm_data_type_t;
        *
        * @param graph_name  The specified graph name
        * @param tensor_name The specified tensor name
        * @return Data type of the input tensor
        */
       bm_data_type_t get_input_dtype(
           const std::string& graph_name,
           const std::string& tensor_name);

**14). get_output_dtype**
    .. code-block:: c

       /**
        * @brief Get data type of an output tensor. Refer to bmdef.h as following.
        *   typedef enum {
        *     BM_FLOAT32 = 0,
        *     BM_FLOAT16 = 1,
        *     BM_INT8 = 2,
        *     BM_UINT8 = 3,
        *     BM_INT16 = 4,
        *     BM_UINT16 = 5,
        *     BM_INT32 = 6,
        *     BM_UINT32 = 7
        *   } bm_data_type_t;
        *
        * @param graph_name  The specified graph name
        * @param tensor_name The specified tensor name
        * @return Data type of the input tensor
        */
       bm_data_type_t get_output_dtype(
           const std::string& graph_name,
           const std::string& tensor_name);

**15). get_input_scale**
    .. code-block:: c

       /**
        * @brief Get scale of an input tensor. Only used for int8 models.
        *
        * @param graph_name  The specified graph name
        * @param tensor_name The specified tensor name
        * @return Scale of the input tensor
        */
       float get_input_scale(
           const std::string& graph_name,
           const std::string& tensor_name);

**16). get_output_scale**
    .. code-block:: c

       /**
        * @brief Get scale of an output tensor. Only used for int8 models.
        *
        * @param graph_name  The specified graph name
        * @param tensor_name The specified tensor name
        * @return Scale of the output tensor
        */
       float get_output_scale(
           const std::string& graph_name,
           const std::string& tensor_name);

**17). reshape**
    .. code-block:: c

       /**
        * @brief Reshape input tensor for dynamic models.
        *
        * The input tensor shapes may change when running dynamic models.
        * New input shapes should be set before inference.
        *
        * @param graph_name   The specified graph name
        * @param input_shapes Specified shapes of all input tensors of the graph
        * @return 0 for success and 1 for failure
        */
       int reshape(
           const std::string&                       graph_name,
           std::map<std::string, std::vector<int>>& input_shapes);

**18). get_input_tensor**
    .. code-block:: c

       /**
        * @brief Get the specified input tensor.
        *
        * @param graph_name  The specified graph name
        * @param tensor_name The specified tensor name
        * @return The specified input tensor
        */
       Tensor* get_input_tensor(
           const std::string& graph_name,
           const std::string& tensor_name);

**19). get_output_tensor**
    .. code-block:: c

       /**
        * @brief Get the specified output tensor.
        *
        * @param graph_name  The specified graph name
        * @param tensor_name The specified tensor name
        * @return The specified output tensor
        */
       Tensor* get_output_tensor(
           const std::string& graph_name,
           const std::string& tensor_name);

**20). scale_input_tensor**
    .. code-block:: c

       /**
        * @brief Scale input tensor for int8 models.
        *
        * @param graph_name  The specified graph name
        * @param tensor_name The specified tensor name
        * @param data        Pointer to float data to be scaled
        */
       void scale_input_tensor(
           const std::string& graph_name,
           const std::string& tensor_name,
           float*             data);

**21). scale_output_tensor**
    .. code-block:: c

       /**
        * @brief Scale output tensor for int8 models.
        *
        * @param graph_name  The specified graph name
        * @param tensor_name The specified tensor name
        * @param data        Pointer to float data to be scaled
        */
       void scale_output_tensor(
           const std::string& graph_name,
           const std::string& tensor_name,
           float*             data);


**22). scale_fp32_to_int8**
    .. code-block:: c

       /**
        * @brief Scale data from float32 to int8. Only used for int8 models.
        *
        * @param src   Poniter to float32 data
        * @param dst   Poniter to int8 data
        * @param scale Value of scale
        * @param size  Size of data
        */
       void scale_fp32_to_int8(float* src, int8_t* dst, float scale, int size);

**23). scale_int8_to_fp32**
    .. code-block:: c

       /**
        * @brief Scale data from int8 to float32. Only used for int8 models.
        *
        * @param src   Poniter to int8 data
        * @param dst   Poniter to float32 data
        * @param scale Value of scale
        * @param size  Size of data
        */
       void scale_int8_to_fp32(int8_t* src, float* dst, float scale, int size);

**24). process**
    .. code-block:: c

       /**
        * @brief Inference with builtin input and output tensors.
        *
        * @param graph_name The specified graph name
        */
       void process(const std::string& graph_name);

       /**
        * @brief Inference with provided input tensors.
        *
        * @param graph_name    The specified graph name
        * @param input_shapes  Shapes of all input tensors
        * @param input_tensors Data pointers of all input tensors in system memory
        */
       void process(
           const std::string&                       graph_name,
           std::map<std::string, std::vector<int>>& input_shapes,
           std::map<std::string, void*>&            input_tensors);

       /**
        * @brief Inference with provided input and output tensors.
        *
        * @param graph_name The specified graph name
        * @param input      Input tensors
        * @param output     Output tensors
        */
       void process(
           const std::string&              graph_name,
           std::map<std::string, Tensor*>& input,
           std::map<std::string, Tensor*>& output);

       /**
        * @brief Inference with provided input and output tensors and input shapes.
        *
        * @param graph_name   The specified graph name
        * @param input        Input tensors
        * @param input_shapes Real input tensor shapes
        * @param output       Output tensors
        */
       void process(
           const std::string&                       graph_name,
           std::map<std::string, Tensor*>&          input,
           std::map<std::string, std::vector<int>>& input_shapes,
           std::map<std::string, Tensor*>&          output);

BMImage
_______

**1). BMImage Constructor**
    .. code-block:: c

       /**
        * @brief The default Constructor.
        */
       BMImage();

       /**
        * @brief The BMImage Constructor.
        *
        * @param handle A Handle instance
        * @param h      Image width
        * @param w      Image height
        * @param format Image format
        * @param dtype  Data type
        */
       BMImage(
           Handle&                  handle,
           int                      h,
           int                      w,
           bm_image_format_ext      format,
           bm_image_data_format_ext dtype);

**2). data**
    .. code-block:: c

       /**
        * @brief Get inner bm_image
        *
        * @return The inner bm_image
        */
       bm_image& data();

**3). width**
    .. code-block:: c

       /**
        * @brief Get the img width.
        *
        * @return the width of img
        */
       int width();

**4). height**
    .. code-block:: c

       /**
        * @brief Get the img height.
        *
        * @return the height of img
        */
       int height();

**5). format**
    .. code-block:: c

       /**
        * @brief Get the img format.
        *
        * @return the format of img
        */
       bm_image_format_ext format();

Decoder
_______

**1). Decoder Constructor**
    .. code-block:: c

       /**
        * @brief Constructor.
        *
        * @param file_path  Path or rtsp url to the video/image file.
        * @param compressed Whether the format of decoded output is compressed NV12.
        * @param tpu_id     ID of TPU, there may be more than one TPU for PCIE mode.
        */
       Decoder(
           const std::string& file_path,
           bool               compressed = true,
           int                tpu_id = 0);

**2). is_opened**
    .. code-block:: c

       /**
        * @brief Judge if the source is opened successfully.
        *
        * @return True if the source is opened successfully
        */
       bool is_opened();

**3). read**
    .. code-block:: c

       /**
        * @brief Read a bm_image from the Decoder.
        *
        * @param handle A bm_handle_t instance
        * @param image Reference of bm_image to be read to
        * @return 0 for success and 1 for failure
        */
       int read(Handle& handle, bm_image& image);

       /**
        * @brief Read a BMImage from the Decoder.
        *
        * @param handle A bm_handle_t instance
        * @param image Reference of BMImage to be read to
        * @return 0 for success and 1 for failure
        */
       int read(Handle& handle, BMImage& image);

Bmcv
_____

**1). Bmcv Constructor**
    .. code-block:: c

       /**
        * @brief Constructor.
        *
        * @param handle A Handle instance
        */
       explicit Bmcv(Handle handle);

**2). bm_image_to_tensor**
    .. code-block:: c

       /**
        * @brief Convert BMImage to tensor.
        *
        * @param img    Input image
        * @param tensor Output tensor
        */
       void bm_image_to_tensor(BMImage &img, Tensor &tensor);

       /**
        * @brief Convert BMImage to tensor.
        *
        * @param img Input image
        */
       Tensor bm_image_to_tensor(BMImage &img);

**3). tensor_to_bm_image**
    .. code-block:: c

       /**
        * @brief Convert tensor to BMImage.
        *
        * @param tensor   Input tensor
        * @param img      Output image
        */
       void tensor_to_bm_image(Tensor &tensor, BMImage &img);

       /**
        * @brief Convert tensor to BMImage.
        *
        * @param tensor   Input tensor
        */
       BMImage tensor_to_bm_image(Tensor &tensor);

**4). crop_and_resize**
    .. code-block:: c

       /**
        * @brief Crop then resize an image.
        *
        * @param input    Input image
        * @param output   Output image
        * @param crop_x0  Start point x of the crop window
        * @param crop_y0  Start point y of the crop window
        * @param crop_w   Width of the crop window
        * @param crop_h   Height of the crop window
        * @param resize_w Target width
        * @param resize_h Target height
        * @return 0 for success and other for failure
        */
       int crop_and_resize(
           BMImage                      &input,
           BMImage                      &output,
           int                          crop_x0,
           int                          crop_y0,
           int                          crop_w,
           int                          crop_h,
           int                          resize_w,
           int                          resize_h);

       /**
        * @brief Crop then resize an image.
        *
        * @param input    Input image
        * @param crop_x0  Start point x of the crop window
        * @param crop_y0  Start point y of the crop window
        * @param crop_w   Width of the crop window
        * @param crop_h   Height of the crop window
        * @param resize_w Target width
        * @param resize_h Target height
        * @return Output image
        */
       BMImage crop_and_resize(
           BMImage                      &input,
           int                          crop_x0,
           int                          crop_y0,
           int                          crop_w,
           int                          crop_h,
           int                          resize_w,
           int                          resize_h);

**5). crop**
    .. code-block:: c

       /**
        * @brief Crop an image with given window.
        *
        * @param input    Input image
        * @param output   Output image
        * @param crop_x0  Start point x of the crop window
        * @param crop_y0  Start point y of the crop window
        * @param crop_w   Width of the crop window
        * @param crop_h   Height of the crop window
        * @return 0 for success and other for failure
        */
       int crop(
           BMImage                      &input,
           BMImage                      &output,
           int                          crop_x0,
           int                          crop_y0,
           int                          crop_w,
           int                          crop_h);

       /**
        * @brief Crop an image with given window.
        *
        * @param input    Input image
        * @param crop_x0  Start point x of the crop window
        * @param crop_y0  Start point y of the crop window
        * @param crop_w   Width of the crop window
        * @param crop_h   Height of the crop window
        * @return Output image
        */
        BMImage crop(
           BMImage                      &input,
           int                          crop_x0,
           int                          crop_y0,
           int                          crop_w,
           int                          crop_h);

**6). resize**
    .. code-block:: c

       /**
        * @brief Resize an image with interpolation of INTER_NEAREST.
        *
        * @param input    Input image
        * @param output   Output image
        * @param resize_w Target width
        * @param resize_h Target height
        * @return 0 for success and other for failure
        */
       int resize(
           BMImage                      &input,
           BMImage                      &output,
           int                          resize_w,
           int                          resize_h);

       /**
        * @brief Resize an image with interpolation of INTER_NEAREST.
        *
        * @param input    Input image
        * @param resize_w Target width
        * @param resize_h Target height
        * @return Output image
        */
       BMImage resize(
           BMImage                      &input,
           int                          resize_w,
           int                          resize_h);

**7). vpp_crop_and_resize**
    .. code-block:: c

       /**
        * @brief Crop then resize an image using vpp.
        *
        * @param input    Input image
        * @param output   Output image
        * @param crop_x0  Start point x of the crop window
        * @param crop_y0  Start point y of the crop window
        * @param crop_w   Width of the crop window
        * @param crop_h   Height of the crop window
        * @param resize_w Target width
        * @param resize_h Target height
        * @return 0 for success and other for failure
        */
        int vpp_crop_and_resize(
            BMImage                      &input,
            BMImage                      &output,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h);

       /**
        * @brief Crop then resize an image using vpp.
        *
        * @param input    Input image
        * @param crop_x0  Start point x of the crop window
        * @param crop_y0  Start point y of the crop window
        * @param crop_w   Width of the crop window
        * @param crop_h   Height of the crop window
        * @param resize_w Target width
        * @param resize_h Target height
        * @return Output image
        */
        BMImage vpp_crop_and_resize(
            BMImage                      &input,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h);

**8). vpp_crop_and_resize_padding**
    .. code-block:: c

       /**
        * @brief Crop then resize an image using vpp.
        *
        * @param input       Input image
        * @param output      Output image
        * @param crop_x0     Start point x of the crop window
        * @param crop_y0     Start point y of the crop window
        * @param crop_w      Width of the crop window
        * @param crop_h      Height of the crop window
        * @param resize_w    Target width
        * @param resize_h    Target height
        * @param padding_in  PaddingAtrr info
        * @return 0 for success and other for failure
        */
        int vpp_crop_and_resize_padding(
            BMImage                      &input,
            BMImage                      &output,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h,
            PaddingAtrr                  &padding_in);

       /**
        * @brief Crop then resize an image using vpp.
        *
        * @param input       Input image
        * @param crop_x0     Start point x of the crop window
        * @param crop_y0     Start point y of the crop window
        * @param crop_w      Width of the crop window
        * @param crop_h      Height of the crop window
        * @param resize_w    Target width
        * @param resize_h    Target height
        * @param padding_in  PaddingAtrr info
        * @return Output image
        */
        BMImage vpp_crop_and_resize_padding(
            BMImage                      &input,
            int                          crop_x0,
            int                          crop_y0,
            int                          crop_w,
            int                          crop_h,
            int                          resize_w,
            int                          resize_h,
            PaddingAtrr                  &padding_in);

**9). vpp_crop**
    .. code-block:: c

       /**
        * @brief Crop an image with given window using vpp.
        *
        * @param input    Input image
        * @param output   Output image
        * @param crop_x0  Start point x of the crop window
        * @param crop_y0  Start point y of the crop window
        * @param crop_w   Width of the crop window
        * @param crop_h   Height of the crop window
        * @return 0 for success and other for failure
        */
       int vpp_crop(
           BMImage                      &input,
           BMImage                      &output,
           int                          crop_x0,
           int                          crop_y0,
           int                          crop_w,
           int                          crop_h);

       /**
        * @brief Crop an image with given window using vpp.
        *
        * @param input    Input image
        * @param crop_x0  Start point x of the crop window
        * @param crop_y0  Start point y of the crop window
        * @param crop_w   Width of the crop window
        * @param crop_h   Height of the crop window
        * @return Output image
        */
       BMImage vpp_crop(
           BMImage                      &input,
           int                          crop_x0,
           int                          crop_y0,
           int                          crop_w,
           int                          crop_h);

**10). vpp_crop_padding**
    .. code-block:: c

       /**
        * @brief Crop an image with given window using vpp.
        *
        * @param input       Input image
        * @param output      Output image
        * @param crop_x0     Start point x of the crop window
        * @param crop_y0     Start point y of the crop window
        * @param crop_w      Width of the crop window
        * @param crop_h      Height of the crop window
        * @param padding_in  PaddingAtrr info
        * @return 0 for success and other for failure
        */
       int vpp_crop_padding(
           BMImage                      &input,
           BMImage                      &output,
           int                          crop_x0,
           int                          crop_y0,
           int                          crop_w,
           int                          crop_h,
           PaddingAtrr                  &padding_in);

       /**
        * @brief Crop an image with given window using vpp.
        *
        * @param input    Input image
        * @param crop_x0  Start point x of the crop window
        * @param crop_y0  Start point y of the crop window
        * @param crop_w   Width of the crop window
        * @param crop_h   Height of the crop window
        * @param padding_in  PaddingAtrr info
        * @return Output image
        */
       BMImage vpp_crop_padding(
           BMImage                      &input,
           int                          crop_x0,
           int                          crop_y0,
           int                          crop_w,
           int                          crop_h,
           PaddingAtrr                  &padding_in);

**11). vpp_resize**
    .. code-block:: c

       /**
        * @brief Resize an image with interpolation of INTER_NEAREST using vpp.
        *
        * @param input    Input image
        * @param output   Output image
        * @param resize_w Target width
        * @param resize_h Target height
        * @return 0 for success and other for failure
        */
        int vpp_resize(
            BMImage                      &input,
            BMImage                      &output,
            int                          resize_w,
            int                          resize_h);

       /**
        * @brief Resize an image with interpolation of INTER_NEAREST using vpp.
        *
        * @param input    Input image
        * @param resize_w Target width
        * @param resize_h Target height
        * @return Output image
        */
       BMImage vpp_resize(
           BMImage                      &input,
           int                          resize_w,
           int                          resize_h);

**12). vpp_resize_padding**
    .. code-block:: c

       /**
        * @brief Resize an image with interpolation of INTER_NEAREST using vpp.
        *
        * @param input       Input image
        * @param output      Output image
        * @param resize_w    Target width
        * @param resize_h    Target height
        * @param padding_in  PaddingAtrr info
        * @return 0 for success and other for failure
        */
        int vpp_resize_padding(
            BMImage                      &input,
            BMImage                      &output,
            int                          resize_w,
            int                          resize_h,
            PaddingAtrr                  &padding_in);

       /**
        * @brief Resize an image with interpolation of INTER_NEAREST using vpp.
        *
        * @param input       Input image
        * @param resize_w    Target width
        * @param resize_h    Target height
        * @param padding_in  PaddingAtrr info
        * @return Output image
        */
       BMImage vpp_resize_padding(
           BMImage                      &input,
           int                          resize_w,
           int                          resize_h,
           PaddingAtrr                  &padding_in);

**13). warp**
    .. code-block:: c

       /**
        * @brief Applies an affine transformation to an image.
        *
        * @param input    Input image
        * @param output   Output image
        * @param matrix   2x3 transformation matrix
        * @return 0 for success and other for failure
        */
       int warp(
           BMImage                            &input,
           BMImage                            &output,
           const std::pair<
             std::tuple<float, float, float>,
             std::tuple<float, float, float>> &matrix);

       /**
        * @brief Applies an affine transformation to an image.
        *
        * @param input    Input image
        * @param matrix   2x3 transformation matrix
        * @return Output image
        */
       BMImage warp(
           BMImage                            &input,
           const std::pair<
             std::tuple<float, float, float>,
             std::tuple<float, float, float>> &matrix);

**14). convert_to**
    .. code-block:: c

       /**
        * @brief Applies a linear transformation to an image.
        *
        * @param input        Input image
        * @param output       Output image
        * @param alpha_beta   (a0, b0), (a1, b1), (a2, b2) factors
        * @return 0 for success and other for failure
        */
       int convert_to(
           BMImage                      &input,
           BMImage                      &output,
           const std::tuple<
             std::pair<float, float>,
             std::pair<float, float>,
             std::pair<float, float>>   &alpha_beta);

       /**
        * @brief Applies a linear transformation to an image.
        *
        * @param input        Input image
        * @param alpha_beta   (a0, b0), (a1, b1), (a2, b2) factors
        * @return Output image
        */
       BMImage convert_to(
           BMImage                      &input,
           const std::tuple<
             std::pair<float, float>,
             std::pair<float, float>,
             std::pair<float, float>>   &alpha_beta);

**15). yuv2bgr**
    .. code-block:: c

       /**
        * @brief Convert an image from YUV to BGR.
        *
        * @param input    Input image
        * @param output   Output image
        * @return 0 for success and other for failure
        */
       int yuv2bgr(
           BMImage                      &input,
           BMImage                      &output);

       /**
        * @brief Convert an image from YUV to BGR.
        *
        * @param input    Input image
        * @return Output image
        */
       BMImage yuv2bgr(BMImage  &input);

**16). vpp_convert**
    .. code-block:: c

       /**
        * @brief Convert an image to BGR PLANAR format using vpp.
        *
        * @param input    Input image
        * @param output   Output image
        * @return 0 for success and other for failure
        */
       int vpp_convert(
           BMImage  &input,
           BMImage  &output);

       /**
        * @brief Convert an image to BGR PLANAR format using vpp.
        *
        * @param input    Input image
        * @return Output image
        */
       BMImage vpp_convert(BMImage  &input);

**17). convert**
    .. code-block:: c

       /**
        * @brief Convert an image to BGR PLANAR format.
        *
        * @param input    Input image
        * @param output   Output image
        * @return 0 for success and other for failure
        */
       int convert(
           BMImage  &input,
           BMImage  &output);

       /**
        * @brief Convert an image to BGR PLANAR format.
        *
        * @param input    Input image
        * @return Output image
        */
       BMImage convert(BMImage  &input);

**18). rectangle**
    .. code-block:: c

       /**
        * @brief Draw a rectangle on input image.
        *
        * @param image      Input image
        * @param x0         Start point x of rectangle
        * @param y0         Start point y of rectangle
        * @param w          Width of rectangle
        * @param h          Height of rectangle
        * @param color      Color of rectangle
        * @param thickness  Thickness of rectangle
        * @return 0 for success and other for failure
        */
       int rectangle(
           BMImage                         &image,
           int                             x0,
           int                             y0,
           int                             w,
           int                             h,
           const std::tuple<int, int, int> &color,
           int                             thickness=1);

**19). imwrite**
    .. code-block:: c

       /**
        * @brief Save the image to the specified file.
        *
        * @param filename   Name of the file
        * @param image      Image to be saved
        * @return 0 for success and other for failure
        */
       int imwrite(
           const std::string &filename,
           BMImage           &image);

**20). get_handle**
    .. code-block:: c

       /**
        * @brief Get Handle instance.
        *
        * @return Handle instance
        */
       Handle get_handle();
