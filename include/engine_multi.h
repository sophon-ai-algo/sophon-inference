/* Copyright 2016-2022 by Sophgo Technologies Inc. All rights reserved.

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

/** @file     engine_multi.h
 *  @brief    Header file of MultiEngine
 *  @author   sophgo
 *  @version  3.0.0
 *  @date     2022-05-11
 */

#pragma once
#include "engine.h"
#include <vector>
#include <map>
#include <iostream>

#ifdef PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eval.h>

namespace py = pybind11;
#endif

namespace sail {
#ifdef PYTHON
  std::vector<py::array_t<float>> split_np_array(py::array_t<float> ost_array,int split_count);
#endif
  class DECL_EXPORT MultiEngine {
  public:
    /**
     * @brief Constructor does not load bmodel.
     *
     * @param bmodel_path Path to bmodel
     * @param tpu_ids     TPU IDS. You can use bm-smi to see available IDs.
     * @param sys_out     Specify the output tensors are in system memory.
     * @param graph_idx   Specify graph idx to use.
     */
        
    MultiEngine(const std::string& bmodel_path,
                std::vector<int>   tpu_ids,
                bool               sys_out=true,
                int                graph_idx=0);
    ~MultiEngine();

    std::vector<std::map<std::string, Tensor*>> process(std::vector<std::map<std::string, Tensor*>>& input_tensors);

#ifdef PYTHON
    std::map<std::string, py::array_t<float>> process(std::map<std::string, py::array_t<float>>& input_tensors);
#endif

    void set_print_flag(bool print_flag); //设置是否打印标志位

    void set_print_time(bool print_flag); //设置打印时间的标志位

    /**
     * @brief Get device id of this engine..
     *
     * @return Device id.
     */
    std::vector<int> get_device_ids();

    /**
     * @brief Get all graph names in the loaded bomodels.
     *
     * @return All graph names
     */
    std::vector<std::string> get_graph_names();

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

  private:

    class MultiEngine_CC;
    class MultiEngine_CC* const _impl;

    /**
     * @brief Forbidden copy constructor.
     * @brief Copy constructor.
     *
     * @param other An other Engine instance.
     */
    MultiEngine(const MultiEngine& other) = delete;

    /**
     * @brief Forbidden assignment function.
     * @brief Assignment function.
     *
     * @param other An other Engine instance.
     * @return Reference of a Engine instance.
     */
    MultiEngine& operator=(const MultiEngine& other) = delete;
  };
}