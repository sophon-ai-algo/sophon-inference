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

/** @file     tensor.hpp
 *  @brief    Header file of Tensor
 *  @author   bitmain
 *  @version  2.0.3
 *  @date     2019-12-27
 */

#pragma once
#include <spdlog/spdlog.h>
#include <bmruntime_interface.h>
#include <bmlib_runtime.h>
#ifdef PYTHON
#include <type_traits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#endif

/// Namespace containing all symbols from the sail library.
namespace sail {

/* sail status */
typedef enum sail_status_t {
  SAIL_SUCCESS = 0,
  SAIL_ERR_DEVICE_INIT = 1,    /* device query failed */
  SAIL_ERR_TENSOR_INIT = 11,   /* sail tensor init failed */
  SAIL_ERR_TENSOR_INNER = 12,  /* sail tensor inner processing failed */
  SAIL_ERR_ENGINE_INIT = 21,   /* sail engine init failed */
  SAIL_ERR_ENGINE_INNER = 22,  /* sail engine inner attribute judge failed */
  SAIL_ERR_ENGINE_INPUT = 23,  /* sail engine input attribute judge failed */
  SAIL_ERR_ENGINE_OUTPUT = 24, /* sail engine output attribute judge failed */
  SAIL_ERR_ENGINE_INFER = 25,  /* sail engine inference failed */
  SAIL_ERR_BMCV_INIT = 31,     /* sail bmcv init failed */
  SAIL_ERR_BMCV_TRANS = 32,    /* sail bmcv data type transform failed */
  SAIL_ERR_BMCV_FUNC = 33,     /* sail bmcv process failed */
  SAIL_ERR_DECODER_INIT = 41,  /* sail decoder init failed */
  SAIL_ERR_DECODER_READ = 42,  /* sail decoder get frame failed */
} sail_status_t;

/**
 * @brief Get the number of available TPUs.
 *
 * @return Number of available TPUs.
 */
int DECL_EXPORT get_available_tpu_num();

/**
 * @brief Get current time of system.
 *
 * @return current time.
 */
double DECL_EXPORT get_current_time_us();

#ifdef _WIN32
    int DECL_EXPORT setenv(const char* name, const char* value, int overwrite);
#endif

int DECL_EXPORT set_print_flag(bool print_flag);

int DECL_EXPORT set_dump_io_flag(bool dump_io_flag);

void DECL_EXPORT get_sail_version(char* sail_version);

bool DECL_EXPORT get_print_flag();

#define PRINT_function_Time_ms(func_name, start_time, end_time) printf("Function[%s]-[%s] time use: %.4f ms \n",__FUNCTION__,func_name, abs(end_time-start_time)/1000.0);

#define PRINT_TIME_MS(func_name, start_time) if(get_print_flag()){ \
          PRINT_function_Time_ms(func_name, start_time, get_current_time_us());}


class DECL_EXPORT Handle {
 public:
  /**
   * @brief Default constructor.
   */
  explicit Handle();

  /**
   * This Function is not recommended, will be removed in future 
   *
   * @brief Constructor using existed bm_handle_t.
   *
   * @param handle A bm_handle_t
   */
  explicit Handle(bm_handle_t handle);

  /**
   * @brief Constructor with device id.
   *
   * @param dev_id Device id
   */
  explicit Handle(int dev_id);

  /**
   * @brief Copy constructor.
   *
   * @param other An other Handle instance.
   */
  Handle(const Handle& other);

  /**
   * @brief Assignment function.
   *
   * @param other An other Handle instance.
   * @return Reference of a Handle instance.
   */
  Handle& operator=(const Handle& other);

  /**
   * @brief Destructor.
   */
  ~Handle();

  /**
   * @brief Get inner bm_handle_t.
   *
   * @return Inner bm_handle_t
   */
  bm_handle_t data();

  /**
   * @brief Get device id of this handle.
   *
   * @return Device id.
   */
  int get_device_id();

  /**
   * @brief Get serial number
   * 
   * @return serial number
   */

  std::string get_sn();

 private:
  //   /**
  //  * @brief Forbidden copy constructor.
  //  */
  // Handle(const Handle& other) = delete;

  // /**
  //  * @brief Forbidden assignment function.
  //  */
  // Handle& operator=(const Handle& other) = delete;

  class Handle_CC;
  class Handle_CC* const _impl;
};

/**
 * @brief Indicator of where to store input and output tensors.
 */
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

/**
 * @brief A class manages the system and device memory of a tensor.
 *
 * A tensor may only stores in sytem memory, or only stores in device memory,
 * or stores in both system memory and device memory. This class handles all
 * the conditions.
 */
class DECL_EXPORT Tensor {
 public:
  /**
   * @brief Common constructor.\n
   * @detail
   *  case 0: only allocate system memory
   *          (handle, shape, dtype, true, false) \n
   *  case 1: only allocate device memory
   *          (handle, shape, dtype, false, true) \n
   *  case 2: allocate system memory and device memory
   *          (handle, shape, dtype, true, true) \n
   *
   * @param handle       Handle instance
   * @param shape        Shape of the tensor
   * @param dtype        Data type
   * @param own_sys_data Indicator of whether own system memory
   * @param own_dev_data Indicator of whether own device memory
   */
  explicit Tensor(
      const Handle&           handle,
      const std::vector<int>& shape={},
      bm_data_type_t          dtype=BM_FLOAT32,
      bool                    own_sys_data=false,
      bool                    own_dev_data=false);

  /**
   * @brief Constructor of only system data.\n
   *
   * @param shape Shape of the tensor
   * @param dtype Data type
   */
  explicit Tensor(
      const std::vector<int>& shape={},
      bm_data_type_t          dtype=BM_FLOAT32);

  /**
   * @brief Copy constructor.
   *
   * @param tensor A Tensor instance
   */
  Tensor(const Tensor& tensor);
  Tensor(Tensor&& tensor);

  /**
   * @brief Assignment function.
   *
   * @param tensor A Tensor instance
   * @return A Tensor instance
   */
  Tensor& operator=(const Tensor& tensor);
  Tensor& operator=(Tensor&&  tensor);

  virtual ~Tensor();

  /**
   * @brief Scale data to tensor in system memory.
   *
   * @param src   Data of type float32 to be scaled from
   * @param scale Scale value
   */
  void scale_from(float* src, float scale);

  /**
   * @brief Scale data to tensor in system memory.
   *
   * @param src   Data of type float32 to be scaled from
   * @param scale Scale value
   * @param size  Size of data
   */
  void scale_from(float* src, float scale, int size);

  /**
   * @brief Scale int32 type data to tensor in system memory.
   *
   * @param src   Data of type int32 to be scaled from
   * @param scale Scale value
   * @param size  Size of data
   */
  void scale_from_int32(int32_t* src, float scale, int size);

  /**
   * @brief Scale tensor to data in system memory.
   *
   * @param dst   Data of type float32 to scaled to
   * @param scale Scale value
   */
  void scale_to(float* dst, float scale);

  /**
   * @brief Scale tensor to data in system memory.
   *
   * @param dst   Data of type float32 to scaled to
   * @param scale Scale value.
   * @param size  Size of data to scale to
   */
  void scale_to(float* dst, float scale, int size);

#ifdef PYTHON
  /**
   * @brief Constructor allocates device memory of the tensor(py).
   *
   * @param handle Handle instance
   * @param data   Ndarray data
   */
    explicit Tensor(Handle handle, pybind11::array_t<float>&   data);
    explicit Tensor(Handle handle, pybind11::array_t<int8_t>&  data);
    explicit Tensor(Handle handle, pybind11::array_t<uint8_t>& data);
    explicit Tensor(Handle handle, pybind11::array_t<int32_t>& data);

    explicit Tensor(Handle handle, pybind11::array_t<float>&   data, bool own_sys_data);
    explicit Tensor(Handle handle, pybind11::array_t<int8_t>&  data, bool own_sys_data);
    explicit Tensor(Handle handle, pybind11::array_t<uint8_t>& data, bool own_sys_data);
    explicit Tensor(Handle handle, pybind11::array_t<int32_t>& data, bool own_sys_data);
  /**
   * @brief Get ndarray in system memory of the tensor.
   *
   * @return Ndarray data
   */
  pybind11::object asnumpy();

  /**
   * @brief Get ndarray in system memory of the tensor with specified shape.
   *
   * @return Ndarray data with specified shape.
   */
  pybind11::object asnumpy(const std::vector<int>& shape);

  /**
   *
   */
  pybind11::array_t<long> pysys_data();

  /**
   * @brief Scale data to tensor in system memory.
   *
   * @param data  Data of type float32 to be scaled from.
   * @param scale Scale value.
   */
  void scale_from(pybind11::array_t<float>& data, float scale);

  void scale_from(pybind11::array_t<int32_t>& data, float scale);
  /**
   * @brief Scale tensor to data in system memory.
   *
   * @param scale Scale value.
   * @return Ndarray data of type float32 to scale to.
   */
  pybind11::array_t<float> scale_to(float scale);

  /**
   * @brief Scale tensor to data in system memory.
   *
   * @param scale Scale value.
   * @param shape Shape of output data to scale to.
   * @return Ndarray data of type float32 to scale to.
   */
  pybind11::array_t<float> scale_to(float scale, const std::vector<int>& shape);

  /*pybind11::array_t<int32_t> scale_to(float scale);
  pybind11::array_t<int32_t> scale_to(
    float                   scale,
    const std::vector<int>& shape);*/

  /**
   * @brief Update system data of the tensor. The data size should not exceed
   *        the tensor size, and the tensor shape will not be changed.
   *
   * @param data Ndarray data with the same data type of the tensor.
   */
  template <typename T>
  void update_data(pybind11::array_t<T>& data) {
    return update_data(data.request(),sizeof(T));
  }
#endif

  /**
   * @brief Get Handle instance.
   *
   * @return Handle reference
   */
  Handle& get_handle();

  /**
   * @brief Get shape of the tensor.
   *
   * @return Shape of the tensor
   */
  const std::vector<int>& shape() const;

  /**
   * @brief Get data type of the tensor.
   *
   * @return Data type of the tensor
   */
  bm_data_type_t dtype() const;

  /**
   * @brief Reset data type and shape of the tensor.
   *
   * @param shape Shape of the tensor
   * @param dtype Data type of the tensor
   */
  void reset(const std::vector<int>& shape, bm_data_type_t dtype);

  /**
   * @brief Reset shape of the tensor.
   *
   * @param shape Shape of the tensor
   */
  void reshape(const std::vector<int>& shape);

  /**
   * @brief Judge if the tensor owns data in system memory.
   *
   * @return True for owns data in system memory.
   */
  bool& own_sys_data();

  /**
   * @brief Judge if the tensor owns data in device memory.
   *
   * @return True for owns data in device memory.
   */
  bool& own_dev_data();

  /**
   * @brief Get data pointer in system memory of the tensor.
   *
   * @return Data pointer in system memory of the tensor
   */
  void* sys_data();

  /**
   * @brief Get pointer to device memory of the tensor.
   *
   * @return Instance of device memory structure of the tensor
   */
  bm_device_mem_t dev_data();

  /**
   * @brief Reset data pointer in system memory of the tensor.
   *
   * @param data  Data pointer in system memory of the tensor
   * @param shape Shape of the data
   */
  void reset_sys_data(
      void*             data,
      std::vector<int>& shape);

  /**
   * @brief Reset pointer to device memory of the tensor.
   *
   * @param data Instance of device memory structure
   */
  void reset_dev_data(bm_device_mem_t data);

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

  /**
   * @brief Copy data from another tensor to this tensor.
   *
   * @param src Another tensor pointer.
   */
  void sync_from(Tensor* src);

  /**
   * @brief Copy data from this tensor to another tensor.
   *
   * @param src Another tensor pointer.
   */
  void sync_to(Tensor* dst);

  /**
   * @brief Free system and device memroy of the tensor.
   */
  void free();

 private:

  class Tensor_CC;
  class Tensor_CC* const _impl;

#ifdef PYTHON
  void update_data(const pybind11::buffer_info& buf, int type_size);
#endif
};

}  // namespace sail
