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

#include <cstring>
#include <numeric>
#include <functional>
#include "bmlib_runtime.h"
#include "tensor.h"
#include "internal.h"


namespace sail {

    int get_available_tpu_num() {
        int count = 0;
        bm_dev_getcount(&count);
        return count;
    }

    Handle::Handle() : own_handle_(true), allocated_(false), dev_id_(-1) {}

    Handle::Handle(bm_handle_t handle) : handle_(handle), own_handle_(false),
                                         allocated_(true), dev_id_(-2) {}

    Handle::Handle(int dev_id) : own_handle_(true), allocated_(true), dev_id_(-1) {
        if (bm_dev_query(dev_id)) {
            printf("Error: Invalid tpu id: %d!\n", dev_id);
            exit(SAIL_ERR_DEVICE_INIT);
        }
        bm_dev_request(&handle_, dev_id);
        dev_id_ = dev_id;
    }

    Handle::Handle(const Handle &other)
            : own_handle_(false), handle_(other.handle_),
              dev_id_(other.dev_id_), allocated_(other.allocated_) {
    }

    Handle &Handle::operator=(const Handle &other) {
        if (this != &other) {
            free();
            own_handle_ = false;
            handle_ = other.handle_;
            allocated_ = other.allocated_;
            dev_id_ = other.dev_id_;
        }
        return *this;
    }

    Handle::~Handle() {
        free();
    }

    bm_handle_t Handle::data() {
        return handle_;
    }

    int Handle::get_device_id() {
        return dev_id_;
    }

    void Handle::free() {
        if (own_handle_) {
            if (allocated_) {
                bm_dev_free(handle_);
            }
            own_handle_ = false;
        }
        handle_ = nullptr;
        dev_id_ = -1;
    }

    inline int get_type_size(bm_data_type_t dtype) {
        int type_size = 0;
        switch (dtype) {
            case BM_FLOAT32:
                return sizeof(float);
            case BM_INT8:
                return sizeof(int8_t);
            case BM_UINT8:
                return sizeof(uint8_t);
            case BM_INT16:
                return sizeof(int16_t);
            case BM_UINT16:
                return sizeof(uint16_t);
            case BM_INT32:
                return sizeof(int32_t);
            case BM_UINT32:
                return sizeof(uint32_t);
            default:
                return 0;
        }
    }

    Tensor::Tensor(
            const Handle &handle,
            const std::vector<int> &shape,
            bm_data_type_t dtype,
            bool own_sys_data,
            bool own_dev_data)
            : handle_(handle), shape_(shape), dtype_(dtype),
              own_sys_data_(own_sys_data), own_dev_data_(own_dev_data),
              sys_data_(nullptr), dev_data_({}), data_size_(0) {
        int ret = 0;
        if (shape_is_valid(shape)) {
            int type_size = get_type_size(dtype);
            data_size_ = std::accumulate(shape_.begin(), shape_.end(),
                                         type_size, std::multiplies<int>());
            if (own_dev_data_) {
                ret = bm_malloc_device_byte_heap_mask(handle_.data(), &dev_data_, 7, data_size_);
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("bm_malloc_device_type() err={}, size={}", ret, data_size_);
                }
            }
            if (own_sys_data_) {
#ifndef IS_SOC_MODE
                sys_data_ = malloc(data_size_);
#else
                if (own_dev_data_) {
                  bm_mem_mmap_device_mem(handle_.data(), &dev_data_,
                                         (unsigned long long*)&sys_data_);
                  own_sys_data_is_mmap_ = true;
                } else {
                  sys_data_ = malloc(data_size_);
                }
#endif
            }
            //} else {
            //  spdlog::error("tensor shape is not valid!");
            //  exit(SAIL_ERR_TENSOR_INIT);
        }
    }

    Tensor::Tensor(
            const std::vector<int> &shape,
            bm_data_type_t dtype)
            : shape_(shape), dtype_(dtype), own_sys_data_(true),
              own_dev_data_(false), sys_data_(nullptr), dev_data_({}), data_size_(0),
              own_sys_data_is_mmap_(false) {
        int type_size = get_type_size(dtype);
        data_size_ = std::accumulate(shape_.begin(), shape_.end(),
                                     type_size, std::multiplies<int>());
        if (data_size_ > 0) {
            sys_data_ = malloc(data_size_);
        }
    }

    Tensor::Tensor(const Tensor &other) {
        handle_ = other.handle_;
        dtype_ = other.dtype_;
        shape_ = other.shape_;
        own_sys_data_ = other.own_sys_data_;
        own_dev_data_ = other.own_dev_data_;
        own_sys_data_is_mmap_ = other.own_sys_data_is_mmap_;
        int type_size = get_type_size(dtype_);
        data_size_ = std::accumulate(shape_.begin(), shape_.end(),
                                     type_size, std::multiplies<int>());
        if (own_dev_data_) {
            int ret = bm_malloc_device_byte_heap_mask(handle_.data(), &dev_data_, 7, data_size_);
            if (BM_SUCCESS != ret) {
                SPDLOG_ERROR("bm_malloc_device_byte_heap_mask() err={}", ret);
                exit(SAIL_ERR_BMCV_INIT);
            }
        }

#ifndef IS_SOC_MODE
        if (own_sys_data_) {
            sys_data_ = malloc(data_size_);
            memcpy(sys_data_, other.sys_data_, data_size_);
            bm_memcpy_s2d(handle_.data(), dev_data_, sys_data_);
        } else {
            void *tmp = malloc(data_size_);
            bm_memcpy_d2s(handle_.data(), tmp, other.dev_data_);
            bm_memcpy_s2d(handle_.data(), dev_data_, tmp);
            std::free(tmp);
        }
#else
        if (own_sys_data_) {
          if (own_dev_data_) {
            bm_mem_mmap_device_mem(handle_.data(), &dev_data_,
                                   (unsigned long long*)&sys_data_);
            own_sys_data_is_mmap_ = true;
          } else {
            sys_data_ = malloc(data_size_);
          }
          memcpy(sys_data_, other.sys_data_, data_size_);
        } else {
          void* tmp = malloc(data_size_);
          bm_memcpy_d2s(handle_.data(), tmp, other.dev_data_);
          bm_memcpy_s2d(handle_.data(), dev_data_, tmp);
          std::free(tmp);
        }
#endif
    }

    Tensor::Tensor(Tensor &&other){
        *this = std::move(other);
    }

    Tensor &Tensor::operator=(const Tensor &other) {
        if (this != &other) {
            free();
            handle_ = other.handle_;
            dtype_ = other.dtype_;
            shape_ = other.shape_;
            own_sys_data_ = other.own_sys_data_;
            own_dev_data_ = other.own_dev_data_;
            own_sys_data_is_mmap_ = other.own_sys_data_is_mmap_;
            int type_size = get_type_size(dtype_);
            data_size_ = std::accumulate(shape_.begin(), shape_.end(),
                                         type_size, std::multiplies<int>());
            if (own_dev_data_) {
                int ret = bm_malloc_device_byte_heap_mask(handle_.data(), &dev_data_, 7, data_size_);
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("bm_malloc_device_byte_heap_mask() err={}", ret);
                    exit(SAIL_ERR_BMCV_INIT);
                }
            }
#ifndef IS_SOC_MODE
            if (own_sys_data_) {
                sys_data_ = malloc(data_size_);
                memcpy(sys_data_, other.sys_data_, data_size_);
                if (own_dev_data_) {
                    bm_memcpy_s2d(handle_.data(), dev_data_, sys_data_);
                }
            } else {
                void *tmp = malloc(data_size_);
                bm_memcpy_d2s(handle_.data(), tmp, other.dev_data_);
                bm_memcpy_s2d(handle_.data(), dev_data_, tmp);
                std::free(tmp);
            }
#else
            if (own_sys_data_) {
              if (own_dev_data_) {
                bm_mem_mmap_device_mem(handle_.data(), &dev_data_,
                                       (unsigned long long*)&sys_data_);
                own_sys_data_is_mmap_ = true;
              } else {
                sys_data_ = malloc(data_size_);
              }
              memcpy(sys_data_, other.sys_data_, data_size_);
            } else {
              void* tmp = malloc(data_size_);
              bm_memcpy_d2s(handle_.data(), tmp, other.dev_data_);
              bm_memcpy_s2d(handle_.data(), dev_data_, tmp);
              std::free(tmp);
            }
#endif
        }
        return *this;
    }

    Tensor &Tensor::operator=(Tensor &&other) {
        if (this != &other) {
            std::swap(handle_, other.handle_);
            std::swap(dtype_, other.dtype_);
            std::swap(shape_, other.shape_);
            std::swap(own_sys_data_, other.own_sys_data_);
            std::swap(own_dev_data_, other.own_dev_data_);
            std::swap(sys_data_, other.sys_data_);
            std::swap(dev_data_, other.dev_data_);
            std::swap(data_size_, other.data_size_);
            std::swap(own_sys_data_is_mmap_, other.own_sys_data_is_mmap_);
        }
        return *this;
    }

    void Tensor::free() {
        if (own_sys_data_ && sys_data_) {
            if (own_sys_data_is_mmap_) {
                bm_mem_unmap_device_mem(handle_.data(), sys_data_, data_size_);
            } else {
                std::free(sys_data_);
            }
            sys_data_ = nullptr;
        }

        if (own_dev_data_) {
            if (dev_data_.u.device.device_addr != 0 && dev_data_.size > 0) bm_free_device(handle_.data(), dev_data_);
            dev_data_ = {};
        } else {
            if (sys_data_ != nullptr && dev_data_.size > 0) {
                if (own_sys_data_is_mmap_) {
                    bm_mem_unmap_device_mem(handle_.data(), sys_data_, dev_data_.size);
                    sys_data_ = nullptr;
                }
            }
        }

    }

    Tensor::~Tensor() {
        free();
    }

    Handle &Tensor::get_handle() {
        return handle_;
    }

    const std::vector<int> &Tensor::shape() const {
        return shape_;
    }

    bm_data_type_t Tensor::dtype() const {
        return dtype_;
    }

    void Tensor::reset(const std::vector<int> &shape, bm_data_type_t dtype) {
        if (!shape_is_valid(shape)) {
            spdlog::error("Invalid tensor shape!");
            exit(SAIL_ERR_TENSOR_INNER);
        }

        int ret = 0;
        int size_shape = std::accumulate(shape.begin(), shape.end(),
                                         1, std::multiplies<int>());

        int data_size = size_shape * get_type_size(dtype);

        if (dev_data_.size < data_size) {
            if (own_dev_data_) {
                bm_free_device(handle_.data(), dev_data_);
                ret = bm_malloc_device_byte_heap_mask(handle_.data(), &dev_data_, 7, data_size);
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("bm_malloc_device_byte err={}, size=%d", ret, data_size);
                }
            }
            if (own_sys_data_) {
#ifndef IS_SOC_MODE
                std::free(sys_data_);
                sys_data_ = malloc(data_size);
#else
                if (own_dev_data_) {
                  bm_mem_mmap_device_mem(handle_.data(), &dev_data_,
                                         (unsigned long long*)&sys_data_);
                  own_sys_data_is_mmap_ = true;
                } else {
                  std::free(sys_data_);
                  sys_data_ = malloc(data_size);
                }
#endif
            }
        }
        dtype_ = dtype;
        shape_ = shape;
        data_size_ = data_size;
    }

    void Tensor::reshape(const std::vector<int> &shape) {
        reset(shape, dtype_);
    }

    void Tensor::reset_sys_data(void *data, std::vector<int> &shape) {
        reshape(shape);
        if (own_dev_data_) {
            if (sys_data_ != nullptr) {
                memcpy(sys_data_, data, data_size_);
            } else {
                spdlog::error("Cannot reset_sys_data when own_dev_data is true.");
                exit(SAIL_ERR_TENSOR_INNER);
            }
        } else if (own_sys_data_) {
            if (sys_data_ != nullptr) {
                memcpy(sys_data_, data, data_size_);
            }
        } else {
            sys_data_ = data;
        }
    }

    void Tensor::reset_dev_data(bm_device_mem_t data) {
        if (own_dev_data_) {
            if (sys_data_ != nullptr && dev_data_.size > 0 && own_sys_data_is_mmap_) {
                bm_mem_unmap_device_mem(handle_.data(), sys_data_, dev_data_.size);
                printf("%s:%d\n", __FILE__, __LINE__);
                sys_data_ = nullptr;
            }

            bm_free_device(handle_.data(), dev_data_);
            dev_data_ = data;
            own_dev_data_ = false;
            // device memory changed, mmap will change too
#ifdef IS_SOC_MODE
            bm_mem_mmap_device_mem(handle_.data(), &dev_data_, (unsigned long long*)&sys_data_);
            own_sys_data_is_mmap_ = true;
#endif
        } else {
            if (sys_data_ != nullptr) {
                if (own_sys_data_is_mmap_) {
#ifdef IS_SOC_MODE
                    bm_mem_unmap_device_mem(handle_.data(), sys_data_, dev_data_.size);
                    sys_data_ = nullptr;
#endif
                }
                dev_data_ = data;
#ifdef IS_SOC_MODE
                bm_mem_mmap_device_mem(handle_.data(), &dev_data_, (unsigned long long*)&sys_data_);
                own_sys_data_is_mmap_ = true;
#endif
            } else {
                dev_data_ = data;
#ifdef IS_SOC_MODE
                bm_mem_mmap_device_mem(handle_.data(), &dev_data_, (unsigned long long*)&sys_data_);
                own_sys_data_is_mmap_ = true;
#endif
            }
        }
        sync_d2s();

    }

    bool &Tensor::own_sys_data() {
        return own_sys_data_;
    }

    bool &Tensor::own_dev_data() {
        return own_dev_data_;
    }

    bool Tensor::shape_is_valid(const std::vector<int> &shape) {
        if (shape.empty()) {
            return false;
        }
        if (std::any_of(shape.begin(), shape.end(), [](int i) { return i <= 0; })) {
            return false;
        }
        return true;
    }

    bm_device_mem_t Tensor::dev_data() {
        return dev_data_;
    }

    void *Tensor::sys_data() {
        return sys_data_;
    }

    void Tensor::sync_s2d() {
        if (sys_data_) {
            if (own_sys_data_is_mmap_) {
                bm_mem_flush_device_mem(handle_.data(), &dev_data_);
            } else {
                bm_memcpy_s2d(handle_.data(), dev_data_, sys_data_);
            }
        }
    }

    void Tensor::sync_s2d(int size) {
        if (sys_data_) {
            if (own_sys_data_is_mmap_) {
                bm_mem_flush_partial_device_mem(handle_.data(), &dev_data_, 0, size);
            } else {
                int ret = bm_memcpy_s2d_partial(handle_.data(), dev_data_, sys_data_, size);
                if (ret != BM_SUCCESS) {
                    spdlog::error("bm_memcpy_s2d_partial(sys_data=%p, dev_data.addr=%p, size=%d)\n",
                                  sys_data_, (void *) dev_data_.u.device.device_addr, size);
                }
            }
        }
    }

    void Tensor::sync_d2s() {
        sync_d2s(data_size_);
    }

    void Tensor::sync_d2s(int size) {
        if (own_dev_data_) {
            if (sys_data_) {
                if (!own_sys_data_is_mmap_) {
                    if (bm_memcpy_d2s_partial(handle_.data(), sys_data_, dev_data_, size) != 0) {
                        SPDLOG_ERROR("bm_memcpy_d2s_partial() err");
                    }
                } else {
                    bm_mem_invalidate_device_mem(handle_.data(), &dev_data_);
                }
            }
        } else {
            if (sys_data_ != nullptr) {
                if (!own_sys_data_is_mmap_) {
                    if (BM_SUCCESS != bm_memcpy_d2s_partial(handle_.data(), sys_data_, dev_data_, size)) {
                        SPDLOG_ERROR("bm_memcpy_d2s_partial() err");
                    }
                } else {
                    bm_mem_invalidate_device_mem(handle_.data(), &dev_data_);
                }
            }
        }
    }

    void Tensor::sync_from(Tensor *src) {
        if (dtype_ != src->dtype()) {
            spdlog::error("sync_from: data type not match!");
            exit(SAIL_ERR_TENSOR_INNER);
        }
        int size = std::accumulate(shape_.begin(), shape_.end(),
                                   1, std::multiplies<int>());
        auto src_shape = src->shape();
        int src_size = std::accumulate(src_shape.begin(), src_shape.end(),
                                       1, std::multiplies<int>());
        if (size != src_size) {
            spdlog::error("sync_from: tensor size not match!");
            exit(SAIL_ERR_TENSOR_INNER);
        }
        auto src_handle = src->get_handle().data();
        auto dtype_size = get_type_size(dtype_);
        void *src_sys_data = src->sys_data();
        bool src_own_dev_data = src->own_dev_data();
        bm_device_mem_t src_dev_data = src->dev_data();
        if (sys_data_) {
            if (src_own_dev_data) {
                bm_memcpy_d2s(src_handle, sys_data_, src_dev_data);
            } else if (src_sys_data) {
                memcpy(sys_data_, src_sys_data, size * dtype_size);
            }
            if (own_dev_data_) {
                bm_memcpy_s2d(handle_.data(), dev_data_, sys_data_);
            }
        } else if (own_dev_data_) {
            if (src_sys_data) {
                bm_memcpy_s2d(handle_.data(), dev_data_, src_sys_data);
            } else if (src_own_dev_data) {
                void *tmp = malloc(size * dtype_size);
                bm_memcpy_d2s(handle_.data(), tmp, src_dev_data);
                bm_memcpy_s2d(handle_.data(), dev_data_, tmp);
                std::free(tmp);
            }
        }
    }

    void Tensor::sync_to(Tensor *dst) {
        if (dtype_ != dst->dtype()) {
            spdlog::error("dst_from: data type not match!");
            exit(SAIL_ERR_TENSOR_INNER);
        }
        int size = std::accumulate(shape_.begin(), shape_.end(),
                                   1, std::multiplies<int>());
        auto dst_shape = dst->shape();
        int dst_size = std::accumulate(dst_shape.begin(), dst_shape.end(),
                                       1, std::multiplies<int>());
        if (size != dst_size) {
            spdlog::error("dst_from: tensor size not match!");
            exit(SAIL_ERR_TENSOR_INNER);
        }
        auto dst_handle = dst->get_handle().data();
        auto dtype_size = get_type_size(dtype_);
        void *dst_sys_data = dst->sys_data();
        bool dst_own_dev_data = dst->own_dev_data();
        bm_device_mem_t dst_dev_data = dst->dev_data();
        if (dst_sys_data) {
            if (own_dev_data_) {
                bm_memcpy_d2s(handle_.data(), dst_sys_data, dev_data_);
            } else if (sys_data_) {
                memcpy(dst_sys_data, sys_data_, size * dtype_size);
            }
            if (dst_own_dev_data) {
                bm_memcpy_s2d(dst_handle, dst_dev_data, sys_data_);
            }
        } else if (dst_own_dev_data) {
            if (sys_data_) {
                bm_memcpy_s2d(dst_handle, dst_dev_data, sys_data_);
            } else if (own_dev_data_) {
                void *tmp = malloc(size * dtype_size);
                bm_memcpy_d2s(handle_.data(), tmp, dev_data_);
                bm_memcpy_s2d(handle_.data(), dst_dev_data, tmp);
                std::free(tmp);
            }
        }
    }

    void Tensor::scale_from(float *src, float scale) {
        int size = std::accumulate(shape_.begin(), shape_.end(),
                                   1, std::multiplies<int>());
        scale_from(src, scale, size);
    }

#if USE_ASM_SSE

    void Tensor::scale_from(float *src, float scale, int size) {
        if (nullptr == sys_data_) {
            spdlog::error("When call scale_from, own_sys_data must be true");
            exit(EXIT_FAILURE);
        }
        AnyScale_SSE(src, BM_FLOAT32, sys_data_, dtype_, scale, size);
    }

    void Tensor::scale_from_int32(int32_t *src, float scale, int size) {
        if (nullptr == sys_data_) {
            spdlog::error("When call scale_from_int32, own_sys_data must be true");
            exit(EXIT_FAILURE);
        }
        AnyScale_SSE(src, BM_INT32, sys_data_, dtype_, scale, size);
    }

    void Tensor::scale_to(float *dst, float scale, int size) {

        if (nullptr == sys_data_) {
            SPDLOG_ERROR("When call scale_to, own_sys_data must be true");
            exit(EXIT_FAILURE);
        }

        AnyScale_SSE(sys_data_, dtype_, dst, BM_FLOAT32, scale, size);
    }

#else // don't use asm language.

    void Tensor::scale_from(float *src, float scale, int size) {
        AnyScale(src, BM_FLOAT32, sys_data_, dtype_, scale, size);
    }

    void Tensor::scale_to(float *dst, float scale, int size) {
        AnyScale(sys_data_, dtype_, dst, BM_FLOAT32, scale, size);
    }


    void Tensor::scale_from_int32(int32_t* src, float scale, int size) {
      if (nullptr == sys_data_) {
        spdlog::error("When call scale_from_int32, own_sys_data must be true");
        exit(EXIT_FAILURE);
      }
      AnyScale(src, BM_INT32, sys_data_, dtype_, scale, size);
    }
#endif

    void Tensor::scale_to(float *dst, float scale) {
        int size = std::accumulate(shape_.begin(), shape_.end(),
                                   1, std::multiplies<int>());
        scale_to(dst, scale, size);
    }

#ifdef PYTHON

    void Tensor::scale_from(pybind11::array_t<float> &data, float scale) {
        auto buf = data.request();
        int size = 1;
        for (auto it : buf.shape) {
            size *= static_cast<int>(it);
        }
        int tensor_size = std::accumulate(shape_.begin(), shape_.end(),
                                          1, std::multiplies<int>());
        if (size > tensor_size) {
            SPDLOG_ERROR("data size exceeds tensor size!");
            exit(SAIL_ERR_TENSOR_INNER);
        }

        float* src = reinterpret_cast<float*>(buf.ptr);

        scale_from(src, scale, size);
    }

    void Tensor::scale_from(pybind11::array_t<int32_t> &data, float scale) {
        auto buf = data.request();
        int size = 1;
        for (auto it : buf.shape) {
            size *= static_cast<int>(it);
        }
        int tensor_size = std::accumulate(shape_.begin(), shape_.end(),
                                          1, std::multiplies<int>());
        if (size > tensor_size) {
            SPDLOG_ERROR("data size exceeds tensor size!");
            exit(SAIL_ERR_TENSOR_INNER);
        }

        int32_t* src = reinterpret_cast<int32_t*>(buf.ptr);
        scale_from_int32(src, scale, size);
    }

    pybind11::array_t<float> Tensor::scale_to(float scale) {
        std::vector<ssize_t> shape;
        for (auto v : shape_) {
            shape.push_back(static_cast<ssize_t>(v));
        }
        auto ndarray = pybind11::array_t<float>(shape);
        float *dst = ndarray.mutable_data();
        scale_to(dst, scale);
        return std::move(ndarray);
    }

    pybind11::array_t<float> Tensor::scale_to(
            float scale,
            const std::vector<int> &shape) {
        int tensor_size = std::accumulate(shape_.begin(), shape_.end(),
                                          1, std::multiplies<int>());
        int size = std::accumulate(shape.begin(), shape.end(),
                                   1, std::multiplies<int>());
        std::vector<ssize_t> array_shape;
        for (auto v : shape) {
            array_shape.push_back(static_cast<ssize_t>(v));
        }
        auto ndarray = pybind11::array_t<float>(array_shape);
        if (size > tensor_size) {
            SPDLOG_ERROR("data size exceeds tensor size!");
            exit(SAIL_ERR_TENSOR_INNER);
        }

        float *dst = ndarray.mutable_data();
        scale_to(dst, scale, size);

        return std::move(ndarray);
    }

    pybind11::object Tensor::asnumpy() {
        std::unique_ptr<uint8_t[]> ptr;
        void *data = sys_data_;
        if (sys_data_ == nullptr) {
            if (dev_data_.u.device.device_addr == 0) {
                SPDLOG_ERROR("asnumpy: sys_data=null and dev_data is null!");
                exit(SAIL_ERR_TENSOR_INNER);
            }
            ptr.reset(new uint8_t[data_size_]);
            data = ptr.get();
            if (BM_SUCCESS != bm_memcpy_d2s_partial(handle_.data(), data, dev_data_, data_size_)) {
                SPDLOG_ERROR("bm_memcpy_d2s_partial() err");
            }
        }

        // fill numpy array
        pybind11::ssize_t item_size = 1;
        std::string format;
        if (dtype_ == BM_FLOAT32) {
            item_size = sizeof(float);
            format = pybind11::format_descriptor<float>::format();
        } else if (dtype_ == BM_INT8) {
            item_size = sizeof(int8_t);
            format = pybind11::format_descriptor<int8_t>::format();
        } else if (dtype_ == BM_UINT8) {
            item_size = sizeof(uint8_t);
            format = pybind11::format_descriptor<uint8_t>::format();
        } else if (dtype_ == BM_INT32) {
            item_size = sizeof(int32_t);
            format = pybind11::format_descriptor<int32_t>::format();
        }

        pybind11::ssize_t ndim = shape_.size();
        std::vector<pybind11::ssize_t> shape;
        for (auto it : shape_) {
            shape.push_back(it);
        }
        std::vector<pybind11::ssize_t> stride;
        for (size_t i = 1; i < shape_.size(); i++) {
            pybind11::ssize_t inner_stride = std::accumulate(shape.begin() + i,
                                                             shape.end(), item_size,
                                                             std::multiplies<pybind11::ssize_t>());
            stride.push_back(inner_stride);
        }
        stride.push_back(item_size);

        pybind11::buffer_info output_buf(data, item_size, format,
                                         ndim, shape, stride);
        if (dtype_ == BM_FLOAT32) {
            return std::move(pybind11::array_t<float>(output_buf));
        } else if (dtype_ == BM_INT8) {
            return std::move(pybind11::array_t<int8_t>(output_buf));
        } else if (dtype_ == BM_UINT8) {
            return std::move(pybind11::array_t<uint8_t>(output_buf));
        } else if (dtype_ == BM_INT32) {
            return std::move(pybind11::array_t<int32_t>(output_buf));
        } else {
            return pybind11::cast<pybind11::none>(Py_None);;
        }
    }

    pybind11::object Tensor::asnumpy(const std::vector<int> &shape) {
        std::unique_ptr<uint8_t[]> ptr;
        void *data = sys_data_;
        if (sys_data_ == nullptr) {
            if (dev_data_.u.device.device_addr == 0) {
                SPDLOG_ERROR("asnumpy: sys_data=null and dev_data is null!");
                exit(SAIL_ERR_TENSOR_INNER);
            }
            ptr.reset(new uint8_t[data_size_]);
            data = ptr.get();
            if (BM_SUCCESS != bm_memcpy_d2s_partial(handle_.data(), data, dev_data_, data_size_)) {
                SPDLOG_ERROR("bm_memcpy_d2s_partial() err");
            }
        }
        // fill numpy array
        pybind11::ssize_t item_size = 1;
        std::string format;
        if (dtype_ == BM_FLOAT32) {
            item_size = sizeof(float);
            format = pybind11::format_descriptor<float>::format();
        } else if (dtype_ == BM_INT8) {
            item_size = sizeof(int8_t);
            format = pybind11::format_descriptor<int8_t>::format();
        } else if (dtype_ == BM_UINT8) {
            item_size = sizeof(uint8_t);
            format = pybind11::format_descriptor<uint8_t>::format();
        } else if (dtype_ == BM_INT32) {
            item_size = sizeof(int32_t);
            format = pybind11::format_descriptor<int32_t>::format();
        }

        pybind11::ssize_t ndim = shape.size();
        std::vector<pybind11::ssize_t> stride;
        for (size_t i = 1; i < shape.size(); i++) {
            pybind11::ssize_t inner_stride = std::accumulate(shape.begin() + i,
                                                             shape.end(), item_size,
                                                             std::multiplies<pybind11::ssize_t>());
            stride.push_back(inner_stride);
        }
        stride.push_back(item_size);


        pybind11::buffer_info output_buf(data, item_size, format,
                                         ndim, shape, stride);
        if (dtype_ == BM_FLOAT32) {
            return std::move(pybind11::array_t<float>(output_buf));
        } else if (dtype_ == BM_INT8) {
            return std::move(pybind11::array_t<int8_t>(output_buf));
        } else if (dtype_ == BM_UINT8) {
            return std::move(pybind11::array_t<uint8_t>(output_buf));
        } else if (dtype_ == BM_INT32) {
            return std::move(pybind11::array_t<int32_t>(output_buf));
        } else {
            return pybind11::cast<pybind11::none>(Py_None);;
        }
    }

    pybind11::array_t<long> Tensor::pysys_data() {
        //printf("pysys_data sys_data_=0x%x\n",sys_data_);

        std::vector<ssize_t> array_shape;
        array_shape.push_back(static_cast<ssize_t>(1));
        auto ndarray = pybind11::array_t<long>(array_shape);
        long *dst = ndarray.mutable_data();
        //long ldata = (long)(sys_data_);
        dst[0] = (long) sys_data_;
        //memcpy(dst, &ldata, 1 * sizeof(int));
        //printf("pysys_data sysd=0x%x\n",dst[0]);
        return std::move(ndarray);
    }

#endif

}  // namespace sail
