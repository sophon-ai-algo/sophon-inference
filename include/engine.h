/* Copyright 2016-2022 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

==============================================================================*/

/** @file     engine.h
 *  @brief    Header file of Engine
 *  @author   bitmain
 *  @version  2.0.3
 *  @date     2019-12-27
 */

#pragma once
#include <bmruntime_interface.h>
#include <bmlib_runtime.h>

#include "graph.h"
#include "tensor.h"

#ifdef PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#endif

/// Namespace containing all symbols from the sail library.
namespace sail {

/**
 * @brief The main class of running deep learning inference on TPU.
 *
 * It's a high level encapsulation of BMRuntime. It automatically manage
 * memory of input and output tensors for both static and dynamic models.
 * It can load more than one model and runs on more than one TPU.
 */
class DECL_EXPORT Engine {
 public:
  /**
   * @brief Constructor does not load bmodel.
   *
   * @param tpu_id TPU ID. You can use bm-smi to see available IDs.
   */
  Engine(int tpu_id);

  /**
   * @brief Constructor loads bmodel from file.
   *
   * @param bmodel_path Path to bmodel
   * @param tpu_id      TPU ID. You can use bm-smi to see available IDs
   * @param mode        Specify the input/output tensors are in system memory
   *                    or device memory
   */
  Engine(
      const std::string& bmodel_path,
      int                tpu_id,
      IOMode             mode);

  /**
   * @brief Constructor loads bmodel from system memory.
   *
   * @param bmodel_ptr  Pointer to bmodel in system memory
   * @param bmodel_size Byte size of bmodel in system memory
   * @param tpu_id      TPU ID. You can use bm-smi to see available IDs.
   * @param mode        Specify the input/output tensors are in system memory
   *                    or device memory
   */
  Engine(
      const void* bmodel_ptr,
      size_t      bmodel_size,
      int         tpu_id,
      IOMode      mode);

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
   * @param handle      Handle created elsewhere.
   * @param mode        Specify the input/output tensors are in system memory
   *                    or device memory
   */
  Engine(
      const void*        bmodel_ptr,
      size_t             bmodel_size,
      const Handle&      handle,
      IOMode             mode);

  ~Engine();

  /**
   * @brief Get Handle instance.
   *
   * @return Handle reference
   */
  Handle& get_handle();

  /**
   * @brief Get device id of this engine..
   *
   * @return Device id.
   */
  int get_device_id();

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
   *     @retval false Failure
   */
  bool load(const void* bmodel_ptr, size_t bmodel_size);

  /**
   * @brief Get all graph names in the loaded bomodels.
   *
   * @return All graph names
   */
  std::vector<std::string> get_graph_names();

  /**
   * @brief Set IOMode for a graph.
   *
   * @param graph_name The specified graph name
   * @param mode The specified IOMode
   */
  void set_io_mode(
    const std::string& graph_name,
    IOMode             mode);

  /**
   * @brief Jugde if the graph is dynamic.
   *
   * @param graph_name The specified graph name
   * @return 0 for dynamic and 1 for static
   */
  bool graph_is_dynamic(const std::string& graph_name);

  /**
   * @brief Get all input tensor names of the specified graph.
   *
   * @param graph_name The specified graph name
   * @return All the input tensor names of the graph
   */
  std::vector<std::string> get_input_names(const std::string& graph_name);

  /**
   * @brief Get all output tensor names of the specified graph.
   *
   * @param graph_name The specified graph name
   * @return All the output tensor names of the graph
   */
  std::vector<std::string> get_output_names(const std::string& graph_name);

  /**
   * @brief Get max shapes of input tensors in a graph.
   *
   * For static models, the max shape is fixed and it should not be changed.
   * For dynamic models, the tensor shape should be smaller than or equal to
   * the max shape.
   *
   * @param graph_name The specified graph name
   * @return Max shapes of input tensors
   */
  std::map<std::string, std::vector<int>> get_max_input_shapes(
      const std::string& graph_name);

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

  /**
   * @brief Get max shapes of output tensors in a graph.
   *
   * For static models, the max shape is fixed and it should not be changed.
   * For dynamic models, the tensor shape should be smaller than or equal to
   * the max shape.
   *
   * @param graph_name The specified graph name
   * @return Max shapes of output tensors
   */
  std::map<std::string, std::vector<int>> get_max_output_shapes(
      const std::string& graph_name);

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

  /**
   * @brief Scale data from float32 to int8. Only used for int8 models.
   *
   * @param src   Poniter to float32 data
   * @param dst   Poniter to int8 data
   * @param scale Value of scale
   * @param size  Size of data
   */
  void scale_fp32_to_int8(float* src, int8_t* dst, float scale, int size);

  /**
   * @brief Scale data from float32 to uint8. Only used for int8 models.
   *
   * @param src   Poniter to float32 data
   * @param dst   Poniter to uint8 data
   * @param scale Value of scale
   * @param size  Size of data
   */
  void scale_fp32_to_uint8(float* src, uint8_t* dst, float scale, int size);

  /**
   * @brief Scale data from int8 to float32. Only used for int8 models.
   *
   * @param src   Poniter to int8 data
   * @param dst   Poniter to float32 data
   * @param scale Value of scale
   * @param size  Size of data
   */
  void scale_int8_to_fp32(int8_t* src, float* dst, float scale, int size);

  /**
   * @brief Scale data from uint8 to float32. Only used for int8 models.
   *
   * @param src   Poniter to uint8 data
   * @param dst   Poniter to float32 data
   * @param scale Value of scale
   * @param size  Size of data
   */
  void scale_uint8_to_fp32(uint8_t* src, float* dst, float scale, int size);
  
  /**
   * @brief Scale data from int32 to float32. Only used for int32 models.
   *
   * @param src   Poniter to int32 data
   * @param dst   Poniter to float32 data
   * @param scale Value of scale
   * @param size  Size of data
   */
  void scale_int32_to_fp32(int32_t* src, float* dst, float scale, int size);
  /**
   * @brief Scale data from fp32 to int32. Only used for int32 models.
   * @param src Pointer to fp32 data
   * @param dst Point to int32 data
   * @param scale Value of scale
   * @param size Size of data
   */
  void scale_fp32_to_int32(float* src, int32_t* dst, float scale, int size);

  /**
   * @brief Create input tensors map, according to and bmodel.
   * @param graph_name   The specified graph name
   * @param create_mode Tensor Create mode
   *  case 0: only allocate system memory 
   *  case 1: only allocate device memory
   *  case other: according to engine IOMode
   */
  std::map<std::string, Tensor*> create_input_tensors_map(const std::string& graph_name, int create_mode = -1);
  
  /**
   * @brief Create output tensors map, according to and bmodel.
   * @param graph_name   The specified graph name 
   * @param create_mode Tensor Create mode 
   *  case 0: only allocate system memory
   *  case 1: only allocate device memory
   *  case other: according to engine IOMode
   */
  std::map<std::string, Tensor*> create_output_tensors_map(const std::string& graph_name, int create_mode = -1);
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
   * @param scale         Indicator of multiply scale for input tensors or not
   */
  void process(
      const std::string&                       graph_name,
      std::map<std::string, std::vector<int>>& input_shapes,
      std::map<std::string, void*>&            input_tensors);

  /**
   * @brief Inference with provided input and output tensors.
   *
   * @param input  Input tensors
   * @param output Output tensors
   */
  void process(
      const std::string&              graph_name,
      std::map<std::string, Tensor*>& input,
      std::map<std::string, Tensor*>& output);

  /**
   * @brief Inference with provided input and output tensors and input shapes.
   *
   * @param input        Input tensors
   * @param input_shapes Real input tensor shapes
   * @param output       Output tensors
   */
  void process(
      const std::string&                       graph_name,
      std::map<std::string, Tensor*>&          input,
      std::map<std::string, std::vector<int>>& input_shapes,
      std::map<std::string, Tensor*>&          output);

#ifdef PYTHON
  Engine(
      pybind11::bytes& bmodel_bytes,
      int              bmodel_size,
      int              tpu_id,
      IOMode           mode);

  Engine(
      pybind11::bytes& bmodel_bytes,
      int              bmodel_size,
      const Handle&    handle,
      IOMode           mode);

  bool load(
      pybind11::bytes& bmodel,
      int              bmodel_size);

  void process(
      const std::string&              graph_name,
      std::map<std::string, Tensor&>& input,
      std::map<std::string, Tensor&>& output);

  void process(
      const std::string&                       graph_name,
      std::map<std::string, Tensor&>&          input,
      std::map<std::string, std::vector<int>>& input_shapes,
      std::map<std::string, Tensor&>&          output);

  std::map<std::string, pybind11::array_t<float>> process(
      const std::string&                               graph_name,
      std::map<std::string, pybind11::array_t<float>>& input_tensors);
    
#endif

 private:

  class Engine_CC;
  class Engine_CC* const _impl;

  /**
   * @brief Forbidden copy constructor.
   * @brief Copy constructor.
   *
   * @param other An other Engine instance.
   */
  Engine(const Engine& other) = delete;

  /**
   * @brief Forbidden assignment function.
   * @brief Assignment function.
   *
   * @param other An other Engine instance.
   * @return Reference of a Engine instance.
   */
  Engine& operator=(const Engine& other) = delete;
};

}  // namespace sail
