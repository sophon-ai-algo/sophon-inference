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
#include "tensor.h"

// using simd intrinsic accelarate scale op
#if defined(__amd64__) || defined(__x86_64__)
#include <x86intrin.h>
#elif defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
#endif

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
  // centos not print the message, change to printf
//    spdlog::error("Invalid tpu id: {}!", dev_id);
    printf("Error: Invalid tpu id: %d!\n", dev_id);
    exit(SAIL_ERR_DEVICE_INIT);
  }
  bm_dev_request(&handle_, dev_id);
  dev_id_ = dev_id;
}

Handle::Handle(const Handle& other)
    : own_handle_(false), handle_(other.handle_),
      dev_id_(other.dev_id_), allocated_(other.allocated_) {
}

Handle& Handle::operator=(const Handle& other) {
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
  if (dtype == BM_FLOAT32) {
    type_size = sizeof(float);
  } else if (dtype == BM_INT8) {
    type_size = sizeof(int8_t);
  } else if (dtype == BM_UINT8) {
    type_size = sizeof(uint8_t);
  }
  return type_size;
}

Tensor::Tensor(
    const Handle&           handle,
    const std::vector<int>& shape,
    bm_data_type_t          dtype,
    bool                    own_sys_data,
    bool                    own_dev_data)
    : handle_(handle), shape_(shape), dtype_(dtype),
      own_sys_data_(own_sys_data), own_dev_data_(own_dev_data),
      sys_data_(nullptr), dev_data_({}) {
  if (shape_is_valid(shape)) {
    int type_size = get_type_size(dtype);
    int data_size = std::accumulate(shape_.begin(), shape_.end(),
                                    type_size, std::multiplies<int>());
    if (own_dev_data_) {
      bm_malloc_device_byte(handle_.data(), &dev_data_, data_size);
    }
    if (own_sys_data_) {
#ifdef USE_PCIE
      sys_data_ = malloc(data_size);
#else
      if (own_dev_data_) {
        bm_mem_mmap_device_mem(handle_.data(), &dev_data_,
                               (unsigned long long*)&sys_data_);
      } else {
        sys_data_ = malloc(data_size);
      }
#endif
    }
  } else {
    spdlog::error("tensor shape is not valid!");
    exit(SAIL_ERR_TENSOR_INIT);
  }
}

Tensor::Tensor(
    const std::vector<int>& shape,
    bm_data_type_t          dtype)
    : shape_(shape), dtype_(dtype), own_sys_data_(true),
      own_dev_data_(false), sys_data_(nullptr), dev_data_({}) {
  int type_size = get_type_size(dtype);
  int data_size = std::accumulate(shape_.begin(), shape_.end(),
                                  type_size, std::multiplies<int>());
  if (data_size > 0) {
    sys_data_ = malloc(data_size);
  }
}

Tensor::Tensor(const Tensor& other) {
  handle_       = other.handle_;
  dtype_        = other.dtype_;
  shape_        = other.shape_;
  own_sys_data_ = other.own_sys_data_;
  own_dev_data_ = other.own_dev_data_;
  int type_size = get_type_size(dtype_);
  int data_size = std::accumulate(shape_.begin(), shape_.end(),
                                  type_size, std::multiplies<int>());
  if (own_dev_data_) {
    bm_malloc_device_byte(handle_.data(), &dev_data_, data_size);
  }
#ifdef USE_PCIE
  if (own_sys_data_) {
    sys_data_ = malloc(data_size);
    memcpy(sys_data_, other.sys_data_, data_size);
    bm_memcpy_s2d(handle_.data(), dev_data_, sys_data_);
  } else {
    void* tmp = malloc(data_size);
    bm_memcpy_d2s(handle_.data(), tmp, other.dev_data_);
    bm_memcpy_s2d(handle_.data(), dev_data_, tmp);
    std::free(tmp);
  }
#else
  if (own_sys_data_) {
    if (own_dev_data_) {
      bm_mem_mmap_device_mem(handle_.data(), &dev_data_,
                             (unsigned long long*)&sys_data_);
    } else {
      sys_data_ = malloc(data_size);
    }
    memcpy(sys_data_, other.sys_data_, data_size);
  } else {
    void* tmp = malloc(data_size);
    bm_memcpy_d2s(handle_.data(), tmp, other.dev_data_);
    bm_memcpy_s2d(handle_.data(), dev_data_, tmp);
    std::free(tmp);
  }
#endif
}

Tensor::Tensor(Tensor&& other) {
  *this = std::move(other);
}

Tensor& Tensor::operator=(const Tensor& other) {
  if (this != &other) {
    free();
    handle_       = other.handle_;
    dtype_        = other.dtype_;
    shape_        = other.shape_;
    own_sys_data_ = other.own_sys_data_;
    own_dev_data_ = other.own_dev_data_;
    int type_size = get_type_size(dtype_);
    int data_size = std::accumulate(shape_.begin(), shape_.end(),
                                    type_size, std::multiplies<int>());
    if (own_dev_data_) {
      bm_malloc_device_byte(handle_.data(), &dev_data_, data_size);
    }
#ifdef USE_PCIE
    if (own_sys_data_) {
      sys_data_ = malloc(data_size);
      memcpy(sys_data_, other.sys_data_, data_size);
      if (own_dev_data_) {
        bm_memcpy_s2d(handle_.data(), dev_data_, sys_data_);
      }
    } else {
      void* tmp = malloc(data_size);
      bm_memcpy_d2s(handle_.data(), tmp, other.dev_data_);
      bm_memcpy_s2d(handle_.data(), dev_data_, tmp);
      std::free(tmp);
    }
#else
    if (own_sys_data_) {
      if (own_dev_data_) {
        bm_mem_mmap_device_mem(handle_.data(), &dev_data_,
                               (unsigned long long*)&sys_data_);
      } else {
        sys_data_ = malloc(data_size);
      }
      memcpy(sys_data_, other.sys_data_, data_size);
    } else {
      void* tmp = malloc(data_size);
      bm_memcpy_d2s(handle_.data(), tmp, other.dev_data_);
      bm_memcpy_s2d(handle_.data(), dev_data_, tmp);
      std::free(tmp);
    }
#endif
  }
  return *this;
}

Tensor& Tensor::operator=(Tensor&& other) {
  if (this != &other) {
    std::swap(handle_, other.handle_);
    std::swap(dtype_, other.dtype_);
    std::swap(shape_, other.shape_);
    std::swap(own_sys_data_, other.own_sys_data_);
    std::swap(own_dev_data_, other.own_dev_data_);
    std::swap(sys_data_, other.sys_data_);
    std::swap(dev_data_, other.dev_data_);
  }
  return *this;
}

void Tensor::free() {
  if (own_sys_data_ && sys_data_) {
#ifdef USE_PCIE
    std::free(sys_data_);
#else
    int type_size = get_type_size(dtype_);
    int data_size = std::accumulate(shape_.begin(), shape_.end(),
                                    type_size, std::multiplies<int>());
    bm_mem_unmap_device_mem(handle_.data(), sys_data_, data_size);
#endif
    sys_data_ = nullptr;
  }
  if (own_dev_data_) {
    bm_free_device(handle_.data(), dev_data_);
    dev_data_ = {};
  }
}

Tensor::~Tensor() {
  free();
}

Handle& Tensor::get_handle() {
  return handle_;
}

const std::vector<int>& Tensor::shape() const {
  return shape_;
}

bm_data_type_t Tensor::dtype() const {
  return dtype_;
}

void Tensor::reset(const std::vector<int>& shape, bm_data_type_t dtype) {
  if (!shape_is_valid(shape)) {
    spdlog::error("Invalid tensor shape!");
    exit(SAIL_ERR_TENSOR_INNER);
  }

  int size_shape_ = std::accumulate(shape_.begin(), shape_.end(),
                                    1, std::multiplies<int>());
  int size_shape  = std::accumulate(shape.begin(), shape.end(),
                                    1, std::multiplies<int>());
  int data_size_  = size_shape_ * get_type_size(dtype_);
  int data_size   = size_shape  * get_type_size(dtype);

  if (data_size_ != data_size) {
    if (own_dev_data_) {
      bm_free_device(handle_.data(), dev_data_);
      bm_malloc_device_byte(handle_.data(), &dev_data_, data_size);
    }
    if (own_sys_data_) {
#ifdef USE_PCIE
      std::free(sys_data_);
      sys_data_ = malloc(data_size);
#else
      if (own_dev_data_) {
        bm_mem_mmap_device_mem(handle_.data(), &dev_data_,
                               (unsigned long long*)&sys_data_);
      } else {
        std::free(sys_data_);
        sys_data_ = malloc(data_size);
      }
#endif
    }
  }
  dtype_ = dtype;
  shape_ = shape;
}

void Tensor::reshape(const std::vector<int>& shape) {
  reset(shape, dtype_);
}

void Tensor::reset_sys_data(void* data, std::vector<int>& shape) {
#ifdef USE_PCIE
  if (own_sys_data_) {
    std::free(sys_data_);
    own_sys_data_ = false;
  }
  sys_data_ = data;
#else
  int type_size = get_type_size(dtype_);
  int size = std::accumulate(shape.begin(), shape.end(),
                             type_size, std::multiplies<int>());
  memcpy(sys_data_, data, size);
#endif
}

void Tensor::reset_dev_data(bm_device_mem_t data) {
  if (own_dev_data_) {
    spdlog::error("Cannot reset_dev_data when own_dev_data is true.");
    exit(SAIL_ERR_TENSOR_INNER);
  } else {
    dev_data_ = data;
  }
}

bool& Tensor::own_sys_data() {
  return own_sys_data_;
}

bool& Tensor::own_dev_data() {
  return own_dev_data_;
}

bool Tensor::shape_is_valid(const std::vector<int>& shape) {
  if (shape.empty()) {
    return false;
  }
  if (std::any_of(shape.begin(), shape.end(), [](int i){return i <= 0;})) {
    return false;
  }
  return true;
}

bm_device_mem_t Tensor::dev_data() {
    return dev_data_;
}

void* Tensor::sys_data() {
  return sys_data_;
}

void Tensor::sync_s2d() {
  if (sys_data_) {
#ifdef USE_PCIE
    bm_memcpy_s2d(handle_.data(), dev_data_, sys_data_);
#else
    bm_mem_flush_device_mem(handle_.data(), &dev_data_);
#endif
  }
}

void Tensor::sync_s2d(int size) {
  if (sys_data_) {
#ifdef USE_PCIE
    bm_memcpy_s2d_partial(handle_.data(), dev_data_, sys_data_, size);
#else
    bm_mem_flush_partial_device_mem(handle_.data(), &dev_data_, 0, size);
#endif
  }
}

void Tensor::sync_d2s() {
  if (sys_data_) {
#ifdef USE_PCIE
    bm_memcpy_d2s(handle_.data(), sys_data_, dev_data_);
#else
    bm_mem_invalidate_device_mem(handle_.data(), &dev_data_);
#endif
  }
}

void Tensor::sync_d2s(int size) {
  if (sys_data_) {
#ifdef USE_PCIE
    bm_memcpy_d2s_partial(handle_.data(), sys_data_, dev_data_, size);
#else
    bm_mem_invalidate_partial_device_mem(handle_.data(), &dev_data_, 0, size);
#endif
  }
}

void Tensor::sync_from(Tensor* src) {
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
  void* src_sys_data = src->sys_data();
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
      void* tmp = malloc(size * dtype_size);
      bm_memcpy_d2s(handle_.data(), tmp, src_dev_data);
      bm_memcpy_s2d(handle_.data(), dev_data_, tmp);
      std::free(tmp);
    }
  }
}

void Tensor::sync_to(Tensor* dst) {
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
  void* dst_sys_data = dst->sys_data();
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
      void* tmp = malloc(size * dtype_size);
      bm_memcpy_d2s(handle_.data(), tmp, dev_data_);
      bm_memcpy_s2d(handle_.data(), dst_dev_data, tmp);
      std::free(tmp);
    }
  }
}

void Tensor::scale_from(float* src, float scale) {
  int size = std::accumulate(shape_.begin(), shape_.end(),
                             1, std::multiplies<int>());
  scale_from(src, scale, size);
}

void Tensor::scale_from(float* src, float scale, int size) {
  if (dtype_ == BM_INT8) {
    int8_t* dst = reinterpret_cast<int8_t*>(sys_data_);
#if defined(__amd64__) || defined(__x86_64__)
    __m128  vec4fp32;
    __m128i vec4int;
    __m128  vec4scales = _mm_set1_ps(scale);
    for (int i = 0; i < size; i+=4) {
      vec4fp32 = _mm_load_ps(src+i);
      vec4fp32 = _mm_mul_ps(vec4fp32, vec4scales);
      vec4int = _mm_cvtps_epi32(vec4fp32);
      vec4int = _mm_packs_epi32(vec4int, vec4int);
      vec4int = _mm_packs_epi16(vec4int, vec4int);
      *reinterpret_cast<int*>(dst+i) = _mm_cvtsi128_si32(vec4int);
    }
#elif defined(__arm__) || defined(__aarch64__)
    int8x8_t  vec_s8x8; // target 8 x int8
    int32x4_t vec_s32x4_l, vec_s32x4_h;
    int32x4_t min_val = vdupq_n_s32(-128);
    int32x4_t max_val = vdupq_n_s32(127);
    float32x4_t vec_f32x4_l, vec_f32x4_h;
    for (int i = 0; i < size; i+=8) {
      vec_f32x4_l = vld1q_f32(src+i);
      vec_f32x4_h = vld1q_f32(src+i+4);
      vec_f32x4_l = vmulq_n_f32(vec_f32x4_l, scale);
      vec_f32x4_h = vmulq_n_f32(vec_f32x4_h, scale);
      vec_s32x4_l = vcvtnq_s32_f32(vec_f32x4_l);
      vec_s32x4_h = vcvtnq_s32_f32(vec_f32x4_h);
      vec_s32x4_l = vminq_s32(vmaxq_s32(vec_s32x4_l, min_val), max_val);
      vec_s32x4_h = vminq_s32(vmaxq_s32(vec_s32x4_h, min_val), max_val);
      vec_s8x8 = vmovn_s16(vcombine_s16(vmovn_s32(vec_s32x4_l), vmovn_s32(vec_s32x4_h)));
      vst1_s8(dst+i, vec_s8x8);
    }
#endif
  } else if (dtype_ == BM_UINT8) {
    uint8_t* dst = reinterpret_cast<uint8_t*>(sys_data_);
#if defined(__amd64__) || defined(__x86_64__)
    __m128 vec4fp32;
    __m128i vec4int;
    __m128 vec4scales = _mm_set1_ps(scale);
    for (int i = 0; i < size; i+=4) {
      vec4fp32 = _mm_load_ps(src+i);
      vec4fp32 = _mm_mul_ps(vec4fp32, vec4scales);
      vec4int = _mm_cvtps_epi32(vec4fp32);
      vec4int = _mm_packus_epi32(vec4int, vec4int);
      vec4int = _mm_packus_epi16(vec4int, vec4int);
      *reinterpret_cast<int*>(dst+i) = _mm_cvtsi128_si32(vec4int);
    }
#elif defined(__arm__) || defined(__aarch64__)
    uint8x8_t  vec_u8x8; // target 8 x int8
    uint32x4_t vec_u32x4_l, vec_u32x4_h;
    uint32x4_t min_val = vdupq_n_u32(0);
    uint32x4_t max_val = vdupq_n_u32(255);
    float32x4_t vec_f32x4_l, vec_f32x4_h;
    for (int i = 0; i < size; i+=8) {
      vec_f32x4_l = vld1q_f32(src+i);
      vec_f32x4_h = vld1q_f32(src+i+4);
      vec_f32x4_l = vmulq_n_f32(vec_f32x4_l, scale);
      vec_f32x4_h = vmulq_n_f32(vec_f32x4_h, scale);
      vec_u32x4_l = vcvtnq_u32_f32(vec_f32x4_l);
      vec_u32x4_h = vcvtnq_u32_f32(vec_f32x4_h);
      vec_u32x4_l = vminq_u32(vmaxq_u32(vec_u32x4_l, min_val), max_val);
      vec_u32x4_h = vminq_u32(vmaxq_u32(vec_u32x4_h, min_val), max_val);
      vec_u8x8 = vmovn_u16(vcombine_u16(vmovn_u32(vec_u32x4_l), vmovn_u32(vec_u32x4_h)));
      vst1_u8(dst+i, vec_u8x8);
    }
#endif
  }
}

void Tensor::scale_to(float* dst, float scale, int size) {
  if (dtype_ == BM_INT8) {
    int8_t* src = reinterpret_cast<int8_t*>(sys_data_);
#if defined(__amd64__) || defined(__x86_64__)
    __m128  vec4scales = _mm_set1_ps(scale);
    __m128i vec4int;
    __m128  vec4fp32;
    for (int i = 0; i < size; i+=4) {
      vec4int = _mm_cvtsi32_si128(*reinterpret_cast<int*>(src+i));
      vec4int = _mm_unpacklo_epi8(vec4int,  _mm_cmplt_epi8(vec4int, _mm_setzero_si128()));
      vec4int = _mm_unpacklo_epi16(vec4int, _mm_cmplt_epi8(vec4int, _mm_setzero_si128()));
      vec4fp32 = _mm_cvtepi32_ps(vec4int);
      vec4fp32 = _mm_mul_ps(vec4fp32, vec4scales);
      _mm_store_ps(dst+i, vec4fp32);
    }
#elif defined(__arm__) || defined(__aarch64__)
    int8x8_t    vec_s8x8;
    int32x4_t   vec_s32x4;
    float32x4_t vec_f32x4;
    for (int i = 0; i < size; i+=4) {
      vec_s8x8  = vld1_s8(src+i);
      vec_s8x8  = vget_low_s8(vcombine_s8(vec_s8x8, vcreate_s8(0)));
      vec_s32x4 = vmovl_s16(vget_low_s16(vmovl_s8(vec_s8x8)));
      vec_f32x4 = vcvtq_f32_s32(vec_s32x4);
      vec_f32x4 = vmulq_n_f32(vec_f32x4, scale);
      vst1q_f32(dst+i, vec_f32x4);
    }
#endif
  } else if (dtype_ == BM_UINT8) {
    uint8_t* src = reinterpret_cast<uint8_t*>(sys_data_);
#if defined(__amd64__) || defined(__x86_64__)
    __m128  vec4scales = _mm_set1_ps(scale);
    __m128i vec4int;
    __m128  vec4fp32;
    for (int i = 0; i < size; i+=4) {
      vec4int = _mm_cvtsi32_si128(*reinterpret_cast<int*>(src+i));
      vec4int = _mm_unpacklo_epi8(vec4int, _mm_setzero_si128());
      vec4int = _mm_unpacklo_epi16(vec4int, _mm_setzero_si128());
      vec4fp32 = _mm_cvtepi32_ps(vec4int);
      vec4fp32 = _mm_mul_ps(vec4fp32, vec4scales);
      _mm_store_ps(dst+i, vec4fp32);
    }
#elif defined(__arm__) || defined(__aarch64__)
    uint8x8_t   vec_u8x8;
    uint32x4_t  vec_u32x4;
    float32x4_t vec_f32x4;
    for (int i = 0; i < size; i+=4) {
      vec_u8x8  = vld1_u8(src+i);
      vec_u8x8  = vget_low_u8(vcombine_u8(vec_u8x8, vcreate_u8(0)));
      vec_u32x4 = vmovl_u16(vget_low_u16(vmovl_u8(vec_u8x8)));
      vec_f32x4 = vcvtq_f32_u32(vec_u32x4);
      vec_f32x4 = vmulq_n_f32(vec_f32x4, scale);
      vst1q_f32(dst+i, vec_f32x4);
    }
#endif
  }
}

void Tensor::scale_to(float* dst, float scale) {
  int size = std::accumulate(shape_.begin(), shape_.end(),
                             1, std::multiplies<int>());
  scale_to(dst, scale, size);
}

#ifdef PYTHON
void Tensor::scale_from(pybind11::array_t<float>& data, float scale) {
  auto buf = data.request();
  int size = 1;
  for (auto it : buf.shape) {
    size *= static_cast<int>(it);
  }
  int tensor_size = std::accumulate(shape_.begin(), shape_.end(),
                                    1, std::multiplies<int>());
  if (size > tensor_size) {
    spdlog::error("data size exceeds tensor size!");
    exit(SAIL_ERR_TENSOR_INNER);
  }
//  float* src = reinterpret_cast<float*>(buf.ptr);
  float* src = new float[size];
  memcpy(src, buf.ptr, size * sizeof(float));
  if (dtype_ == BM_FLOAT32) {
    float* dst = reinterpret_cast<float*>(sys_data_);
    // for (int i = 0; i < size; ++i) {
    //   dst[i] = src[i] * scale;
    // }
  } else {
    scale_from(src, scale, size);
  }
  delete [] src;
}

pybind11::array_t<float> Tensor::scale_to(float scale) {
  std::vector<ssize_t> shape;
  for (auto v : shape_) {
    shape.push_back(static_cast<ssize_t>(v));
  }
  auto ndarray = pybind11::array_t<float>(shape);
  float* dst = ndarray.mutable_data();
  int size = std::accumulate(shape_.begin(), shape_.end(),
                             1, std::multiplies<int>());
  if (dtype_ == BM_FLOAT32) {
    float* src = reinterpret_cast<float*>(sys_data_);
    // for (int i = 0; i < size; ++i) {
    //   dst[i] = src[i] * scale;
    // }
  } else {
    scale_to(dst, scale);
  }
  return std::move(ndarray);
}

pybind11::array_t<float> Tensor::scale_to(
    float                   scale,
    const std::vector<int>& shape)  {
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
    spdlog::error("data size exceeds tensor size!");
    exit(SAIL_ERR_TENSOR_INNER);
  }
  float* dst = ndarray.mutable_data();
  if (dtype_ == BM_FLOAT32) {
    float* src = reinterpret_cast<float*>(sys_data_);
    // for (int i = 0; i < size; ++i) {
    //   dst[i] = src[i] * scale;
    // }
  } else {
    scale_to(dst, scale, size);
  }
  return std::move(ndarray);
}

pybind11::object Tensor::asnumpy() {
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
  }
  pybind11::ssize_t ndim = shape_.size();
  std::vector<pybind11::ssize_t> shape;
  for (auto it : shape_) {
    shape.push_back(it);
  }
  std::vector<pybind11::ssize_t> stride;
  for (size_t i = 1; i < shape_.size(); i++) {
    pybind11::ssize_t inner_stride = std::accumulate(shape.begin() + i,
        shape.end(), item_size, std::multiplies<pybind11::ssize_t>());
    stride.push_back(inner_stride);
  }
  stride.push_back(item_size);
  pybind11::buffer_info output_buf(sys_data_, item_size, format,
                                   ndim, shape, stride);
  if (dtype_ == BM_FLOAT32) {
    return std::move(pybind11::array_t<float>(output_buf));
  } else if (dtype_ == BM_INT8) {
    return std::move(pybind11::array_t<int8_t>(output_buf));
  } else if (dtype_ == BM_UINT8) {
    return std::move(pybind11::array_t<uint8_t>(output_buf));
  } else {
    return pybind11::cast<pybind11::none>(Py_None);;
  }
}

pybind11::object Tensor::asnumpy(const std::vector<int>& shape) {
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
  }
  pybind11::ssize_t ndim = shape.size();
  std::vector<pybind11::ssize_t> stride;
  for (size_t i = 1; i < shape.size(); i++) {
    pybind11::ssize_t inner_stride = std::accumulate(shape.begin() + i,
        shape.end(), item_size, std::multiplies<pybind11::ssize_t>());
    stride.push_back(inner_stride);
  }
  stride.push_back(item_size);
  pybind11::buffer_info output_buf(sys_data_, item_size, format,
                                   ndim, shape, stride);
  if (dtype_ == BM_FLOAT32) {
    return std::move(pybind11::array_t<float>(output_buf));
  } else if (dtype_ == BM_INT8) {
    return std::move(pybind11::array_t<int8_t>(output_buf));
  } else if (dtype_ == BM_UINT8) {
    return std::move(pybind11::array_t<uint8_t>(output_buf));
  } else {
    return pybind11::cast<pybind11::none>(Py_None);;
  }
}
#endif

}  // namespace sail
