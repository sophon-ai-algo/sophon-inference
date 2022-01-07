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

/** @file     graph.hpp
 *  @brief    Header file of graph
 *  @author   bitmain
 *  @version  2.0.3
 *  @date     2019-12-27
 */

#pragma once
#include <map>
#include <set>
#include <vector>
#include <string>
#include <memory>

#include "tensor.h"

/// Namespace containing all symbols from the sail library.
namespace sail {

/**
 * @brief A class handles one model on one TPU.
 *
 * It's the real worker to do inference of a model on one TPU. And it copys
 * input and output tensors between system memory and device memory.
 */
class Graph {
 public:
  /**
   * @brief Constructor.
   *
   * @param handle bm_handle.
   * @param p_bmrt Pointer of bmruntime
   * @param name   Name of the graph
   */
  explicit Graph(
      bm_handle_t        handle,
      void*              p_bmrt,
      const std::string& name);

  virtual ~Graph();

  /**
   * @brief Initialize data types for all tensors.
   *
   * @param input_dtypes  Data types of input tensors
   * @param output_dtypes Data types of output tensors
   */
  void init_dtypes(
      std::map<std::string, bm_data_type_t>* input_dtypes,
      std::map<std::string, bm_data_type_t>* output_dtypes);

  /**
   * @brief Allocate system and device memory for all tensors.
   *
   * @param input_shapes  Shapes of input tensors
   * @param output_shapes Shapes of output tensors
   * @return Program state
   *     @retval O      Success
   *     @retval others Failure
   */
  int alloc_tensors(
      IOMode                                   io_mode,
      std::map<std::string, std::vector<int>>* input_shapes,
      std::map<std::string, std::vector<int>>* output_shapes);

  /**
   * @brief Set IOMode.
   *
   * @param mode The specified IOMode
   */
  void set_io_mode(IOMode io_mode);

  /**
   * @brief Reshape input tensors.
   */
  void reshape();

  /**
   * @brief Reset system data pointer of input tensor.
   *
   * @param tensor_name Tensor name
   * @param data        Pointer to data of tensor
   */
  void reset_input_tensor(const std::string& tensor_name, void* data);

  /**
   * @brief Inference with builtin tensors.
   */
  void inference();

  /**
   * @brief Inference with provided input/output tensors.
   *
   * @param input_tensors  Input tensors
   * @param input_shapes   Real shapes of input tensors
   * @param output_tensors Output tensors
   * @return Real shapes of output tensors
   */
  std::map<std::string, std::vector<int>> inference(
      std::map<std::string, Tensor*>&          input,
      std::map<std::string, std::vector<int>>& input_shapes,
      std::map<std::string, Tensor*>&          output);

  /**
   * @brief Inference with provided input/output tensors for static models.
   *
   * @param input_tensors  Input tensors
   * @param output_tensors Output tensors
   * @return Real shapes of output tensors
   */
  std::map<std::string, std::vector<int>> inference(
      std::map<std::string, Tensor*>& input,
      std::map<std::string, Tensor*>& output);

  /**
   * @brief Get pointer to a input tensor of specified name.
   *
   * @param tensor_name Specified tensor name
   * @return Pointer to a Tensor instance
   */
  Tensor* get_input_tensor(const std::string& name);

  /**
   * @brief Get pointer to a output tensor of specified name.
   *
   * @param tensor_name Specified tensor name
   * @return Pointer to a Tensor instance
   */
  Tensor* get_output_tensor(const std::string& name);

  /**
   * @brief Clear information of tensors and free device memory structs.
   */
  void free();

 private:
  /**
   * @brief Forbidden copy constructor.
   */
  Graph(const Graph &other);

  /**
   * @brief Forbidden assignment function.
   */
  Graph& operator=(const Graph &other);

  /**
   * @brief Initialize graph information, such as input/output name and number
   */
  void init_graph_info();

  /**
   * @brief Copy all input tensors from system memory to device memory.
   *
   * @param input_tensors Input tensors
   * @param bm_inputs     A pointer to bm_tensor_t array for input tensors
   * @param input_shapes  Real shapes of input tensors
   */
  void input_s2d(
    std::map<std::string, Tensor*>&          input_tensors,
    bm_tensor_t*                             bm_inputs,
    std::map<std::string, std::vector<int>>& input_shapes);

  /**
   * @brief Copy all output tensors from device memory to system memory.
   *
   * @param output_tensors output tensors
   * @param bm_outputs     A pointer to bm_tensor_t array for output tensors
   * @param output_shapes  Real shapes of output tensors
   */
  void output_d2s(
    std::map<std::string, Tensor*>&          output_tensors,
    bm_tensor_t*                             bm_outputs,
    std::map<std::string, std::vector<int>>& output_shapes);

  /**
   * @brief Map tensors to bm_tensors.
   *
   * @param input        A map contains Tensor instances of input
   * @param output       A map contains Tensor instances of output
   * @param bm_in        A pointer to bm_tensor_t array for input tensors
   * @param bm_out       A pointer to bm_tensor_t array for output tensors
   */
  void map_tensors(
      std::map<std::string, Tensor*>&          input,
      std::map<std::string, Tensor*>&          output,
      bm_tensor_t*                             bm_in,
      bm_tensor_t*                             bm_out);

  void map_input_tensors(
            std::map<std::string, Tensor*>&          input,
            bm_tensor_t*                             bm_in);

  void map_output_tensors(
            std::map<std::string, Tensor*>&          output,
            bm_tensor_t*                             bm_out);

  void dump_io_tensors(const std::string& name, bm_tensor_t *ins, int in_num, bm_tensor_t *outs, int out_num);
  /**
   * @brief Map real input shapes to bm_tensors.
   *
   * @param input_shapes Real shapes of input tensors
   * @param bm_in        A pointer to bm_tensor_t array for input tensors
   */
  void map_input_shapes(
      std::map<std::string, std::vector<int>>& input_shapes,
      bm_tensor_t*                             bm_inputs);

  /// Graph name
  std::string name_;

  /// Indicator of where to store input and output tensors.
  /// SYSI: Input tensors are in system memory while output tensors are
  ///       in device memory.
  /// SYSO: Input tensors are in device memory while output tensors are
  ///       in system memory.
  /// SYSIO: Both input and output tensors are in system memory.
  /// DEVIO: Both input and output tensors are in device memory.
  IOMode io_mode_;

  /// bm_handle.
  bm_handle_t handle_;

  /// Pointer to bmruntime.
  void* p_bmrt_;

  /// Data type of input tensors
  std::map<std::string, bm_data_type_t>* input_dtypes_;

  /// Data type of output tensors
  std::map<std::string, bm_data_type_t>* output_dtypes_;

  /// Map of input tensor names and indexes
  std::map<std::string, size_t> input_map_;

  /// Map of output tensor names and indexes
  std::map<std::string, size_t> output_map_;

  /// Pointers to all input tensors needed by bmrt_launch
  bm_tensor_t* inputs_;

  /// Pointers to all output tensors needed by bmrt_launch
  bm_tensor_t* outputs_;

  /// Pointers to all input tensors
  std::map<std::string, Tensor*> input_tensors_;

  /// Pointers to all output tensors
  std::map<std::string, Tensor*> output_tensors_;

  /// Shapes of all input tensors
  std::map<std::string, std::vector<int>>* input_shapes_;

  /// Shapes of all output tensors
  std::map<std::string, std::vector<int>>* output_shapes_;
};

}  // namespace sail
