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
int get_available_tpu_num();

class Handle {
 public:
  /**
   * @brief Default constructor.
   */
  explicit Handle();

  /**
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

 private:
  /**
   * @brief Free inner bm_handle_t.
   */
  void free();

  bool own_handle_;
  bool allocated_;
  bm_handle_t handle_;
  int dev_id_;
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
class Tensor {
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
  template <typename T>
  explicit Tensor(
      Handle                handle,
      pybind11::array_t<T>& data)
      : handle_(handle), dtype_(BM_FLOAT32), own_sys_data_(false),
        own_dev_data_(true), sys_data_(nullptr), dev_data_({}) {
    pybind11::buffer_info buf = data.request();
    if (buf.ndim < 1) {
      spdlog::error("Invalid tensor shape!");
      exit(SAIL_ERR_TENSOR_INIT);
    }
    if (std::is_same<T, float>::value) {
      dtype_ = BM_FLOAT32;
    } else if (std::is_same<T, int8_t>::value) {
      dtype_ = BM_INT8;
    } else if (std::is_same<T, uint8_t>::value) {
      dtype_ = BM_UINT8;
    }
    shape_.clear();
    for (auto it : buf.shape) {
      shape_.push_back(static_cast<int>(it));
    }
    // alloc dev_mem
    int data_size = std::accumulate(shape_.begin(), shape_.end(),
                    sizeof(T), std::multiplies<int>());
    bm_malloc_device_byte(handle_.data(), &dev_data_, data_size);
#ifdef USE_PCIE
//    sys_data_ = buf.ptr;
    memcpy(sys_data_, buf.ptr, data_size);
#else
    bm_mem_mmap_device_mem(handle_.data(), &dev_data_, (unsigned long long*)&sys_data_);
    memcpy(sys_data_, buf.ptr, data_size);
#endif
  }

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
   * @brief Scale data to tensor in system memory.
   *
   * @param data  Data of type float32 to be scaled from.
   * @param scale Scale value.
   */
  void scale_from(pybind11::array_t<float>& data, float scale);

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

  /**
   * @brief Update system data of the tensor. The data size should not exceed
   *        the tensor size, and the tensor shape will not be changed.
   *
   * @param data Ndarray data with the same data type of the tensor.
   */
  template <typename T>
  void update_data(pybind11::array_t<T>& data) {
    pybind11::buffer_info buf = data.request();
    if (buf.ndim < 1) {
      spdlog::error("Invalid tensor shape!");
      exit(SAIL_ERR_TENSOR_INNER);
    }
    std::vector<int> shape;
    for (auto it : buf.shape) {
      shape.push_back(static_cast<int>(it));
    }
    size_t type_size = 1;
    if (dtype_ == BM_FLOAT32) {
      type_size = sizeof(float);
    } else if (dtype_ == BM_INT8) {
      type_size = sizeof(int8_t);
    } else if (dtype_ == BM_UINT8) {
      type_size = sizeof(uint8_t);
    }
    int old_size = std::accumulate(shape_.begin(), shape_.end(),
  		 type_size, std::multiplies<int>());
    int new_size = std::accumulate(shape.begin(), shape.end(),
  		 sizeof(T), std::multiplies<int>());
    if (new_size > old_size) {
      spdlog::error("Data size exceeds tensor size!");
      exit(SAIL_ERR_TENSOR_INNER);
    }
#ifdef USE_PCIE
//    if (own_sys_data_) {
//      std::free(sys_data_);
//      own_sys_data_ = false;
//    }
//    sys_data_ = buf.ptr;
    memcpy(sys_data_, buf.ptr, new_size);

#else
    memcpy(sys_data_, buf.ptr, new_size);
#endif
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
  /**
   * @brief Judge if a tensor shape is valid.
   *
   * @param shape Shape of a tensor
   * @return True for valid and flase for invalid..
   */
  bool shape_is_valid(const std::vector<int>& shape);

  /// Handle instance.
  Handle handle_;

  /// Data type
  bm_data_type_t dtype_;

  /// Shape of the tensor.
  std::vector<int> shape_;

  /// Indicator of whether own the data pointer in system memory.
  bool own_sys_data_;

  /// Indicator of whether own the device memory struct.
  bool own_dev_data_;

  /// Data pointer in system memory of the tensor.
  void* sys_data_;

  /// Instance of device memory structure.
  bm_device_mem_t dev_data_;
};

}  // namespace sail
