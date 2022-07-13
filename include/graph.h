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
  Graph(const Graph &other) = delete;

  /**
   * @brief Forbidden assignment function.
   */
  Graph& operator=(const Graph &other) = delete;

  class Graph_CC;
  class Graph_CC* const _impl;
};

}  // namespace sail
