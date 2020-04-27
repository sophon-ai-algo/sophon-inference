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

#include <numeric>
#include <cmath>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
#include "engine.h"
#ifdef PYTHON
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#endif

// using simd intrinsic accelarate scale op
#if defined(__amd64__) || defined(__x86_64__)
#include <x86intrin.h>
#elif defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace sail {

Engine::Engine(int tpu_id)
    : io_mode_(DEVIO), handle_(tpu_id), p_bmrt_(nullptr) {
  init_bmrt();
}

Engine::Engine(
    const std::string& bmodel_path,
    int                tpu_id,
    IOMode             mode)
    : io_mode_(mode), handle_(tpu_id), p_bmrt_(nullptr) {
  if (init_bmrt()) {
    return;
  }
  if (!load(bmodel_path)) {
    return;
  }
  if (alloc_tensors()) {
    return;
  }
}

Engine::Engine(
    const void* bmodel_ptr,
    size_t      bmodel_size,
    int         tpu_id,
    IOMode      mode)
    : io_mode_(mode), handle_(tpu_id), p_bmrt_(nullptr) {
  if (init_bmrt()) {
    return;
  }
  if (!load(bmodel_ptr, bmodel_size)) {
    return;
  }
  if (alloc_tensors()) {
    return;
  }
}

Engine::Engine(const Handle& handle)
    : io_mode_(DEVIO), handle_(handle), p_bmrt_(nullptr) {
  init_bmrt();
}

Engine::Engine(
    const std::string& bmodel_path,
    const Handle&      handle,
    IOMode             mode)
    : io_mode_(mode), handle_(handle), p_bmrt_(nullptr) {
  if (init_bmrt()) {
    return;
  }
  if (!load(bmodel_path)) {
    return;
  }
  if (alloc_tensors()) {
    return;
  }
}

Engine::Engine(
    const void*        bmodel_ptr,
    size_t             bmodel_size,
    const Handle&      handle,
    IOMode             mode)
    : io_mode_(mode), handle_(handle), p_bmrt_(nullptr) {
  if (init_bmrt()) {
    return;
  }
  if (!load(bmodel_ptr, bmodel_size)) {
    return;
  }
  if (alloc_tensors()) {
    return;
  }
}

Engine::Engine(const Engine& other) :
    io_mode_          (other.io_mode_),
    handle_           (other.handle_),
    p_bmrt_           (other.p_bmrt_),
    graphs_           (other.graphs_),
    input_dtypes_     (other.input_dtypes_),
    output_dtypes_    (other.output_dtypes_),
    input_scales_     (other.input_scales_),
    output_scales_    (other.output_scales_),
    input_shapes_     (other.input_shapes_),
    output_shapes_    (other.output_shapes_),
    max_input_shapes_ (other.max_input_shapes_),
    max_output_shapes_(other.max_output_shapes_) {
}

Engine& Engine::operator=(const Engine& other) {
  if (this != &other) {
    free();
    io_mode_           = other.io_mode_;
    handle_            = other.handle_;
    p_bmrt_            = other.p_bmrt_;
    graphs_            = other.graphs_;
    input_dtypes_      = other.input_dtypes_;
    output_dtypes_     = other.output_dtypes_;
    input_scales_      = other.input_scales_;
    output_scales_     = other.output_scales_;
    input_shapes_      = other.input_shapes_;
    output_shapes_     = other.output_shapes_;
    max_input_shapes_  = other.max_input_shapes_;
    max_output_shapes_ = other.max_output_shapes_;
  }
  return *this;
}

Engine::~Engine() {
  free();
}

Handle& Engine::get_handle() {
  return handle_;
}

int Engine::get_device_id() {
  return handle_.get_device_id();
}

bool Engine::load(const std::string& bmodel_path) {
  std::vector<std::string> previous_graph_names = get_graph_names();
  // check bmodel is exist or not
  struct stat buffer;
  if (stat(bmodel_path.c_str(), &buffer) != 0) {
    spdlog::error("bmodel {} does not exist", bmodel_path);
    exit(SAIL_ERR_ENGINE_INIT);
  }
  if (!bmrt_load_bmodel(p_bmrt_, bmodel_path.c_str())) {
    spdlog::error("Load {} failed", bmodel_path);
    exit(SAIL_ERR_ENGINE_INIT);
  }
  std::vector<std::string> current_graph_names = get_graph_names();
  std::vector<std::string> update_graph_names(
      current_graph_names.begin() + previous_graph_names.size(),
      current_graph_names.end());
  update_status(update_graph_names);
  return true;
}

bool Engine::load(const void* bmodel_ptr, size_t bmodel_size) {
  // check bmodel is exist or not
  if (bmodel_size <= 0) {
    spdlog::error("bmodel path does not exist");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  std::vector<std::string> previous_graph_names = get_graph_names();
  if (!bmrt_load_bmodel_data(p_bmrt_, bmodel_ptr, bmodel_size)) {
    spdlog::error("Load bmodel failed");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  std::vector<std::string> current_graph_names = get_graph_names();
  std::vector<std::string> update_graph_names(
      current_graph_names.begin() + previous_graph_names.size(),
      current_graph_names.end());
  update_status(update_graph_names);
  return true;
}

std::vector<std::string> Engine::get_graph_names() {
  std::vector<std::string> graph_names;
  int graph_num = bmrt_get_network_number(p_bmrt_);
  const char** names;
  bmrt_get_network_names(p_bmrt_, &names);
  for (int i = 0; i < graph_num; ++i) {
    graph_names.push_back(names[i]);
  }
  std::free(names);
  return graph_names;
}

void Engine::set_io_mode(const std::string& graph_name, IOMode mode) {
  graphs_[graph_name]->set_io_mode(mode);
}

std::vector<std::string> Engine::get_input_names(
    const std::string& graph_name) {
  std::vector<std::string> tensor_names;
  const bm_net_info_t* p_info = bmrt_get_network_info(p_bmrt_,
                                                      graph_name.c_str());
  for (int i = 0; i < p_info->input_num; ++i) {
    tensor_names.push_back(p_info->input_names[i]);
  }
  return tensor_names;
}

std::vector<std::string> Engine::get_output_names(
    const std::string &graph_name) {
  std::vector<std::string> tensor_names;
  const bm_net_info_t* p_info = bmrt_get_network_info(p_bmrt_,
                                                      graph_name.c_str());
  for (int i = 0; i < p_info->output_num; ++i) {
    tensor_names.push_back(p_info->output_names[i]);
  }
  return tensor_names;
}

std::map<std::string, std::vector<int>> Engine::get_max_input_shapes(
    const std::string& graph_name) {
  auto it_find = max_input_shapes_.find(graph_name);
  if (it_find != max_input_shapes_.end()) {
    return max_input_shapes_[graph_name];
  } else {
    std::map<std::string, std::vector<int>> input_shapes;
    const bm_net_info_t* p_info = bmrt_get_network_info(p_bmrt_,
                                                        graph_name.c_str());
    std::map<std::string, int> input_map;
    for (int i = 0; i < p_info->input_num; ++i) {
      input_map[p_info->input_names[i]] = i;
    }
    for (auto item : input_map) {
      auto name = item.first;
      auto idx = item.second;
      bm_shape_t init_shape = p_info->stages[0].input_shapes[idx];
      std::vector<int> shape(init_shape.dims,
                             init_shape.dims + init_shape.num_dims);
      for (int i = 1; i < p_info->stage_num; ++i) {
        bm_shape_t in_shape = p_info->stages[i].input_shapes[idx];
        for (int j = 0; j < in_shape.num_dims; ++j) {
          if (in_shape.dims[j] > shape[j]) {
            shape[j] = in_shape.dims[j];
          }
        }
      }
      input_shapes[name] = shape;
    }
    return input_shapes;
  }
}

std::vector<int> Engine::get_input_shape(
    const std::string& graph_name,
    const std::string& tensor_name) {
  return input_shapes_[graph_name][tensor_name];
}

std::map<std::string, std::vector<int>>
Engine::get_max_output_shapes(const std::string& graph_name) {
  auto it_find = max_output_shapes_.find(graph_name);
  if (it_find != max_output_shapes_.end()) {
    return max_output_shapes_[graph_name];
  } else {
    std::map<std::string, std::vector<int>> output_shapes;
    std::vector<std::string> output_names;
    const bm_net_info_t* p_info = bmrt_get_network_info(p_bmrt_,
                                                        graph_name.c_str());
    for (int i = 0; i < p_info->output_num; ++i) {
      output_names.push_back(p_info->output_names[i]);
    }
    std::map<std::string, int> output_map;
    for (int i = 0; i < p_info->output_num; ++i) {
      output_map[p_info->output_names[i]] = i;
    }
    for (auto name : output_names) {
      int idx = output_map[name];
      bm_shape_t init_shape = p_info->stages[0].output_shapes[idx];
      std::vector<int> shape(init_shape.dims,
                             init_shape.dims + init_shape.num_dims);
      for (int i = 1; i < p_info->stage_num; ++i) {
        bm_shape_t out_shape = p_info->stages[i].output_shapes[idx];
        for (int j = 0; j < out_shape.num_dims; ++j) {
          if (out_shape.dims[j] > shape[j]) {
            shape[j] = out_shape.dims[j];
          }
        }
      }
      output_shapes[name] = shape;
    }
    return output_shapes;
  }
}

std::vector<int> Engine::get_output_shape(
    const std::string& graph_name,
    const std::string& tensor_name) {
  return output_shapes_[graph_name][tensor_name];
}

bm_data_type_t Engine::get_input_dtype(
    const std::string& graph_name,
    const std::string& tensor_name) {
  return input_dtypes_[graph_name][tensor_name];
}

bm_data_type_t Engine::get_output_dtype(
    const std::string& graph_name,
    const std::string& tensor_name) {
  return output_dtypes_[graph_name][tensor_name];
}

float Engine::get_input_scale(
    const std::string& graph_name,
    const std::string& tensor_name) {
  return input_scales_[graph_name][tensor_name];
}

float Engine::get_output_scale(
    const std::string& graph_name,
    const std::string& tensor_name) {
  return output_scales_[graph_name][tensor_name];
}

int Engine::reshape(
    const std::string&                       graph_name,
    std::map<std::string, std::vector<int>>& input_shapes) {
  int ret = is_input_shape_valid(graph_name, input_shapes);
  if (ret) {
    spdlog::error("Reshape failed.");
    exit(SAIL_ERR_ENGINE_INNER);
  }
  input_shapes_[graph_name] = input_shapes;
  graphs_[graph_name]->reshape();
  return 0;
}

Tensor* Engine::get_input_tensor(
    const std::string& graph_name,
    const std::string& tensor_name) {
  return graphs_[graph_name]->get_input_tensor(tensor_name);
}

Tensor* Engine::get_output_tensor(
    const std::string& graph_name,
    const std::string& tensor_name) {
  return graphs_[graph_name]->get_output_tensor(tensor_name);
}

void Engine::scale_input_tensor(
    const std::string& graph_name,
    const std::string& tensor_name,
    float*             data) {
  void* tensor =
      graphs_[graph_name]->get_input_tensor(tensor_name)->sys_data();
  std::vector<int> shape = get_input_shape(graph_name, tensor_name);
  int size = std::accumulate(shape.begin(), shape.end(),
                             1, std::multiplies<int>());
  if (input_dtypes_[graph_name][tensor_name] == BM_FLOAT32) {
    float* target = reinterpret_cast<float*>(tensor);
    memcpy(target, data, size * sizeof(float));
  } else if (input_dtypes_[graph_name][tensor_name] == BM_INT8) {
    float scale = get_input_scale(graph_name, tensor_name);
    int8_t* target = reinterpret_cast<int8_t*>(tensor);
    scale_fp32_to_int8(data, target, scale, size);
  } else if (input_dtypes_[graph_name][tensor_name] == BM_UINT8) {
    float scale = get_input_scale(graph_name, tensor_name);
    uint8_t* target = reinterpret_cast<uint8_t*>(tensor);
    scale_fp32_to_uint8(data, target, scale, size);
  }
}

void Engine::scale_output_tensor(
    const std::string& graph_name,
    const std::string& tensor_name,
    float*             data) {
  void* tensor =
      graphs_[graph_name]->get_output_tensor(tensor_name)->sys_data();
  std::vector<int> shape = get_output_shape(graph_name, tensor_name);
  int size = std::accumulate(shape.begin(), shape.end(),
                             1, std::multiplies<int>());
  if (output_dtypes_[graph_name][tensor_name] == BM_FLOAT32) {
    float* target = reinterpret_cast<float*>(tensor);
    memcpy(data, target, size * sizeof(float));
  } else if (output_dtypes_[graph_name][tensor_name] == BM_INT8) {
    float scale = get_output_scale(graph_name, tensor_name);
    int8_t* target = reinterpret_cast<int8_t*>(tensor);
    scale_int8_to_fp32(target, data, scale, size);
  } else if (output_dtypes_[graph_name][tensor_name] == BM_UINT8) {
    float scale = get_output_scale(graph_name, tensor_name);
    uint8_t* target = reinterpret_cast<uint8_t*>(tensor);
    scale_uint8_to_fp32(target, data, scale, size);
  }
}

// x86 sse simd
#if defined(__amd64__) || defined(__x86_64__)
void Engine::scale_fp32_to_int8(float* src, int8_t* dst, float scale, int size) {
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
}

void Engine::scale_fp32_to_uint8(float* src, uint8_t* dst, float scale, int size) {
  __m128  vec4fp32;
  __m128i vec4int;
  __m128  vec4scales = _mm_set1_ps(scale);
  for (int i = 0; i < size; i+=4) {
    vec4fp32 = _mm_load_ps(src+i);
    vec4fp32 = _mm_mul_ps(vec4fp32, vec4scales);
    vec4int = _mm_cvtps_epi32(vec4fp32);
    vec4int = _mm_packus_epi32(vec4int, vec4int);
    vec4int = _mm_packus_epi16(vec4int, vec4int);
    *reinterpret_cast<int*>(dst+i) = _mm_cvtsi128_si32(vec4int);
  }
}

void Engine::scale_int8_to_fp32(int8_t* src, float* dst, float scale, int size) {
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
}

void Engine::scale_uint8_to_fp32(uint8_t* src, float* dst, float scale, int size) {
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
}
// arm using simd neon TODO, now only accelate by omp
#elif defined(__arm__) || defined(__aarch64__)
void Engine::scale_fp32_to_int8(float* src, int8_t* dst, float scale, int size) {
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
}

void Engine::scale_fp32_to_uint8(float* src, uint8_t* dst, float scale, int size) {
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
}

void Engine::scale_int8_to_fp32(int8_t* src, float* dst, float scale, int size) {
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
}

void Engine::scale_uint8_to_fp32(uint8_t* src, float* dst, float scale, int size) {
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
}
#endif

void Engine::process(const std::string& graph_name) {
  graphs_[graph_name]->inference();
}

void Engine::process(
    const std::string&                       graph_name,
    std::map<std::string, std::vector<int>>& input_shapes,
    std::map<std::string, void*>&            input_tensors) {
  reshape(graph_name, input_shapes);
  for (auto& item : input_tensors) {
    if (input_dtypes_[graph_name][item.first] == BM_FLOAT32) {
      graphs_[graph_name]->reset_input_tensor(item.first, item.second);
    }
  }
  graphs_[graph_name]->inference();
}

void Engine::process(
    const std::string&                       graph_name,
    std::map<std::string, Tensor*>&          input_tensors,
    std::map<std::string, std::vector<int>>& input_shapes,
    std::map<std::string, Tensor*>&          output_tensors) {
  output_shapes_[graph_name] = graphs_[graph_name]->inference(
      input_tensors, input_shapes, output_tensors);
}

void Engine::process(
    const std::string&              graph_name,
    std::map<std::string, Tensor*>& input_tensors,
    std::map<std::string, Tensor*>& output_tensors) {
  output_shapes_[graph_name] = graphs_[graph_name]->inference(
      input_tensors, output_tensors);
}

int Engine::init_bmrt() {
  p_bmrt_ = bmrt_create(handle_.data());
  if (!p_bmrt_) {
    spdlog::error("Init bmruntime failed.");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  return 0;
}

int Engine::is_input_shape_valid(
    const std::string&                             graph_name,
    const std::map<std::string, std::vector<int>>& input_shapes) {
  std::vector<std::string> input_names;
  const bm_net_info_t* p_info = bmrt_get_network_info(p_bmrt_,
                                                      graph_name.c_str());
  std::map<std::string, int> input_map;
  for (int i = 0; i < p_info->input_num; ++i) {
    input_names.push_back(p_info->input_names[i]);
    input_map[p_info->input_names[i]] = i;
  }
  for (auto& item : input_shapes) {
    auto tensor_name = item.first;
    auto it = std::find(input_names.begin(), input_names.end(), tensor_name);
    if (it == input_names.end()) {
      spdlog::error("Bad input tensor '{}' for {}.", tensor_name, graph_name);
      exit(SAIL_ERR_ENGINE_INNER);
    }
    std::vector<int>& max_shape = max_input_shapes_[graph_name][tensor_name];
    if (item.second.size() != max_shape.size()) {
      spdlog::error("Bad dimension of input tensor {} for {}, {}(N) vs {}(Y)",
                    tensor_name, graph_name, item.second.size(),
                    max_shape.size());
      exit(SAIL_ERR_ENGINE_INNER);
    }
    if (p_info->is_dynamic) {
      std::string msg("Invalid value at dim {} for input tensor '{}': {}");
      for (size_t i = 0; i < max_shape.size(); ++i) {
        if (item.second[i] > max_shape[i] || item.second[i] <= 0) {
          spdlog::error(msg.c_str(), i, tensor_name, item.second[i]);
          exit(SAIL_ERR_ENGINE_INNER);
        }
      }
    } else {
      int idx = input_map[tensor_name];
      bool flag = false;
      for (int i = 0; i < p_info->stage_num; ++i) {
        bm_shape_t in_shape = p_info->stages[i].input_shapes[idx];
        bool flag_stage = true;
        for (size_t i = 0; i < max_shape.size(); ++i) {
          if (in_shape.dims[i] != item.second[i]) {
            flag_stage = false;
            break;
          }
        }
        if (flag_stage) {
          flag = true;
          break;
        }
      }
      if (!flag) {
        spdlog::error("Invalid shape for input tensor '{}': [{}]",
                      tensor_name, fmt::join(item.second, ", "));
        exit(SAIL_ERR_ENGINE_INNER);
      }
    }
  }
  return 0;
}

int Engine::alloc_tensors() {
  /** For some models, output shapes are not fixed and are comfirmed after
   *  launching. So system memory and device memory for output tensors are
   *  allocated for the maximum shapes.
   */
  auto graph_names = get_graph_names();
  for (auto graph_name : graph_names) {
    int ret = graphs_[graph_name]->alloc_tensors(io_mode_,
                                                 &input_shapes_[graph_name],
                                                 &output_shapes_[graph_name]);
    if (ret) {
      spdlog::error("Allocate tensors failed.");
      exit(SAIL_ERR_ENGINE_INNER);
    }
  }
  return 0;
}

int Engine::update_status(std::vector<std::string> graph_names) {
  for (auto graph_name : graph_names) {
    const bm_net_info_t* info = bmrt_get_network_info(p_bmrt_,
                                                      graph_name.c_str());
    std::shared_ptr<Graph> graph(new Graph(handle_.data(), p_bmrt_, graph_name));
    graphs_[graph_name] = graph;
    auto max_shape = get_max_input_shapes(graph_name);
    max_input_shapes_[graph_name] = max_shape;
    max_shape = get_max_output_shapes(graph_name);
    max_output_shapes_[graph_name] = max_shape;
    input_shapes_[graph_name] = max_input_shapes_[graph_name];
    output_shapes_[graph_name] = max_output_shapes_[graph_name];
    for (int i = 0; i < info->input_num; ++i) {
      input_scales_[graph_name][info->input_names[i]] = info->input_scales[i];
      input_dtypes_[graph_name][info->input_names[i]] = info->input_dtypes[i];
    }
    for (int i = 0; i < info->output_num; ++i) {
      output_scales_[graph_name][info->output_names[i]] = info->output_scales[i];
      output_dtypes_[graph_name][info->output_names[i]] = info->output_dtypes[i];
    }
    graphs_[graph_name]->init_dtypes(&input_dtypes_[graph_name],
                                     &output_dtypes_[graph_name]);
  }
  return 0;
}

void Engine::free() {
  // free graphs, and device memories will be freed by deconstructor of Graph
  for (auto& g : graphs_) {
    g.second.reset();
  }
  // Destroy bmruntime
  bmrt_destroy(p_bmrt_);
}

#ifdef PYTHON
Engine::Engine(
    pybind11::bytes& bmodel,
    int              bmodel_size,
    int              tpu_id,
    IOMode           mode)
    : io_mode_(mode), handle_(tpu_id), p_bmrt_(nullptr) {
  char* bmodel_ptr = nullptr;
  ssize_t size;
  if (PYBIND11_BYTES_AS_STRING_AND_SIZE(bmodel.ptr(), &bmodel_ptr, &size)) {
    spdlog::error("Unable to extract bytes contents!");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  if (bmodel_size != static_cast<int>(size)) {
    spdlog::error("Wrong bmodel_size.");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  if (init_bmrt()) {
    return;
  }
  if (load(bmodel_ptr, bmodel_size)) {
    return;
  }
  if (alloc_tensors()) {
    return;
  }
}

Engine::Engine(
    pybind11::bytes& bmodel,
    int              bmodel_size,
    const Handle&    handle,
    IOMode           mode)
    : io_mode_(mode), handle_(handle), p_bmrt_(nullptr) {
  char* bmodel_ptr = nullptr;
  ssize_t size;
  if (PYBIND11_BYTES_AS_STRING_AND_SIZE(bmodel.ptr(), &bmodel_ptr, &size)) {
    spdlog::error("Unable to extract bytes contents!");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  if (bmodel_size != static_cast<int>(size)) {
    spdlog::error("Wrong bmodel_size.");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  if (init_bmrt()) {
    return;
  }
  if (load(bmodel_ptr, bmodel_size)) {
    return;
  }
  if (alloc_tensors()) {
    return;
  }
}

bool Engine::load(
    pybind11::bytes& bmodel,
    int              bmodel_size) {
  char* bmodel_ptr = nullptr;
  ssize_t size;
  if (PYBIND11_BYTES_AS_STRING_AND_SIZE(bmodel.ptr(), &bmodel_ptr, &size)) {
    spdlog::error("Unable to extract bytes contents!");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  if (bmodel_size != static_cast<int>(size)) {
    spdlog::error("Wrong bmodel_size.");
    exit(SAIL_ERR_ENGINE_INIT);
  }
  if (!load(bmodel_ptr, bmodel_size)) {
    return false;
  }
  return true;
}

std::map<std::string, pybind11::array_t<float>> Engine::process(
    const std::string&                               graph_name,
    std::map<std::string, pybind11::array_t<float>>& input_tensors) {
  std::map<std::string, std::vector<int>> shapes;
  std::map<std::string, float*> tensors;
  for (auto& item : input_tensors) {
    std::string name = std::string(pybind11::str(item.first));
    auto buf = item.second.request();
    std::vector<int> shape;
    for (auto it : buf.shape) {
      shape.push_back(static_cast<int>(it));
    }
    shapes[name] = shape;
    tensors[name] = reinterpret_cast<float*>(buf.ptr);
  }
  reshape(graph_name, shapes);
  for (auto& item : tensors) {
    if (input_dtypes_[graph_name][item.first] == BM_FLOAT32) {
      graphs_[graph_name]->reset_input_tensor(item.first, item.second);
    } else {
      scale_input_tensor(graph_name, item.first, item.second);
    }
  }
  graphs_[graph_name]->inference();
  std::map<std::string, pybind11::array_t<float>> output;
  shapes = output_shapes_[graph_name];
  for (auto& s : shapes) {
    if (output_dtypes_[graph_name][s.first] == BM_INT8
        || output_dtypes_[graph_name][s.first] == BM_UINT8) {
      std::vector<ssize_t> shape;
      for (auto v : s.second) {
        shape.push_back(static_cast<ssize_t>(v));
      }
      auto ndarray = pybind11::array_t<float>(shape);
      float* data = ndarray.mutable_data();
      scale_output_tensor(graph_name, s.first, data);
      output[pybind11::str(s.first.c_str())] = ndarray;
    } else {
      pybind11::ssize_t item_size = sizeof(float);
      std::string format = pybind11::format_descriptor<float>::format();
      pybind11::ssize_t ndim = s.second.size();
      std::vector<pybind11::ssize_t> shape;
      for (auto it : s.second) {
          shape.push_back(it);
      }
      std::vector<pybind11::ssize_t> stride;
      for (size_t i = 1; i < s.second.size(); ++i) {
        pybind11::ssize_t inner_stride = std::accumulate(shape.begin() + i,
            shape.end(), sizeof(float), std::multiplies<pybind11::ssize_t>());
        stride.push_back(inner_stride);
      }
      stride.push_back(sizeof(float));
      void* ptr = get_output_tensor(graph_name, s.first)->sys_data();
      pybind11::buffer_info buf(ptr, item_size, format, ndim, shape, stride);
      output[pybind11::str(s.first.c_str())] = pybind11::array_t<float>(buf);
    }
  }
  return std::move(output);
}

void Engine::process(
    const std::string&              graph_name,
    std::map<std::string, Tensor&>& input_tensors,
    std::map<std::string, Tensor&>& output_tensors) {
  std::map<std::string, Tensor*> inputs;
  std::map<std::string, Tensor*> outputs;
  for (auto& item : input_tensors) {
    inputs[item.first] = &item.second;
  }
  for (auto& item : output_tensors) {
    outputs[item.first] = &item.second;
  }
  process(graph_name, inputs, outputs);
}

void Engine::process(
    const std::string&                       graph_name,
    std::map<std::string, Tensor&>&          input_tensors,
    std::map<std::string, std::vector<int>>& input_shapes,
    std::map<std::string, Tensor&>&          output_tensors) {
  std::map<std::string, Tensor*> inputs;
  std::map<std::string, Tensor*> outputs;
  for (auto& item : input_tensors) {
    inputs[item.first] = &item.second;
  }
  for (auto& item : output_tensors) {
    outputs[item.first] = &item.second;
  }
  process(graph_name, inputs, input_shapes, outputs);
}
#endif

}  // namespace sail
