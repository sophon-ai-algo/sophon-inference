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

#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
#include "engine_multi.h"
#include <fstream>
#include <chrono>
#include <algorithm>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>

#ifdef PYTHON
using namespace pybind11::literals;
#endif

using namespace std;

#define SAVE_DEBUG_DATA         0
#define MAX_BUFFER_EACH_ENGINE  8       //每个Engine最多输入的数量,超过此值时送数据将会失败

namespace sail {

typedef std::map<std::string, Tensor*> MAP_STR_TENSOR;

inline int shape_size_me(const std::vector<int> shape) {
    int resu = 1;
    for(int i =0;i<shape.size();++i){
        resu *= shape[i];
    }
    return resu;
}

#ifdef PYTHON
typedef std::map<std::string, py::array_t<float>> MAP_STR_NUMPY;

std::vector<py::array_t<float>> split_np_array(py::array_t<float> ost_array,int split_count)
{
    py::buffer_info buf = ost_array.request();
    std::vector<py::array_t<float>> output;

    float* ptr_start = static_cast<float*>(buf.ptr);
    std::vector<ssize_t> buf_shape = buf.shape;
    std::vector<ssize_t> buf_strides = buf.strides;

    if (buf_shape[0]%split_count != 0){
        SPDLOG_ERROR("Batch Size not support!");
        exit(SAIL_ERR_ENGINE_INNER);
    }
    int batch_size = buf_shape[0]/split_count;
    int split_data_len = batch_size;
    for (int i = 1; i < buf_shape.size(); i++){
        split_data_len*=buf_shape[i];
    }
    
    for(int i=0;i<split_count;++i){
        std::vector<ssize_t> buf_shape_temp = buf_shape;
        buf_shape_temp[0] = batch_size;
        std::vector<ssize_t> buf_strides_temp = buf_strides;

        py::buffer_info buf_temp(
            ptr_start+split_data_len*i,
            buf.itemsize,
            buf.format, 
            buf.ndim,
            buf_shape_temp,
            buf_strides_temp);
        py::array_t<float> np_array_temp(buf_temp);
        output.push_back(std::move(np_array_temp));
    }
    return output;
}

py::array_t<float> merage_np_array(std::vector<py::array_t<float>> ost_array_list)
{
    py::module np = py::module::import("numpy");  // like 'import numpy as np'
    py::list shape_temp;
    if(ost_array_list.size() <= 0){
        SPDLOG_ERROR("Input np.array list must not empty!");
        exit(SAIL_ERR_ENGINE_INNER);
    }
    py::buffer_info buf_0 = ost_array_list[0].request();
    std::vector<ssize_t> buf_0_shape = buf_0.shape;
    if(buf_0_shape.size() <= 0){
        SPDLOG_ERROR("Error shape!");
        exit(SAIL_ERR_ENGINE_INNER);
    }
    int shape_0 = buf_0_shape[0];
    int array_buffer_len = shape_0;
    for (size_t i = 1; i < ost_array_list.size(); ++i)   {
        py::buffer_info buf = ost_array_list[i].request();
        std::vector<ssize_t> buf_shape = buf.shape;
        if (buf_shape.size() != buf_0_shape.size()){
            SPDLOG_ERROR("Input np.array list must has same dims!");
            exit(SAIL_ERR_ENGINE_INNER);
        }
        for (size_t j = 0; j < buf_shape.size(); ++j) {
            if (buf_shape[j] != buf_0_shape[j]){
                SPDLOG_ERROR("Input np.array list must has same shape!");
                exit(SAIL_ERR_ENGINE_INNER);
            }
        }
        shape_0 += buf_shape[0];
    }
    shape_temp.append(shape_0);
    for (size_t j = 1; j < buf_0_shape.size(); ++j) {
        array_buffer_len *= buf_0_shape[j];
        shape_temp.append(buf_0_shape[j]);
    }
    py::array_t<float> arr = np.attr("zeros")(shape_temp, "dtype"_a="float32");
    py::buffer_info arr_buf = arr.request();
    float *arr_ptr = static_cast<float *>(arr_buf.ptr);
    for (size_t i = 0; i < ost_array_list.size(); ++i)   {
        py::buffer_info buf = ost_array_list[i].request();
        memcpy(arr_ptr+i*array_buffer_len,buf.ptr,sizeof(float)*array_buffer_len);
    }
    return arr;
}

py::list convert_intv_to_pylist(std::vector<int> shape)
{
    py::list result_shape;
    for (int i=0;i<shape.size();++i){
        result_shape.append(shape[i]);
    }
    return std::move(result_shape);
}

#endif

class MultiEngine::MultiEngine_CC{
public:
    MultiEngine_CC(const std::string& bmodel_path,
            std::vector<int>   tpu_ids,
            bool               sys_out,
            int                graph_idx=0);

    ~MultiEngine_CC();

    void set_print_flag(bool print_flag){
        print_flag_ = print_flag;
    }

    void set_print_time(bool print_flag){
        print_time_flag_ = print_flag;
    }
    /**
    * @brief Push Data to Inference buffer, if failed return false
    * @param input      Input tensors, Tensor必须外部缓存,程序内使用浅拷贝,只保存了数据的地址 
    */
    bool PushData(std::map<std::string, Tensor*>& input, bool delete_ost_data=false);

    /**
    * @brief Get Inference result
    * 
    * @return std::vector<std::map<std::string, Tensor*>> 
    */
    std::vector<std::map<std::string, Tensor*>> GetResult();

    /**
     * @brief MultiEngine process
     * 
     * @param input      Input tensors
     * @return std::vector<std::map<std::string, Tensor*>> 
     */
    std::vector<std::map<std::string, Tensor*>> process(
        std::vector<std::map<std::string, Tensor*>>& input_tensors);

#ifdef PYTHON
    std::map<std::string, py::array_t<float>> create_output_np_map(int split_num);

    std::map<std::string, Tensor*> create_tensors_map(std::map<std::string, py::array_t<float>>& input_tensors, int split_idx);

    int get_nparray_length(py::array_t<float> nparray);

    float* get_data_start_ptr(std::map<std::string, py::array_t<float>> np_map, std::string tensor_name, int idx, bool is_out=true);

    std::map<std::string, py::array_t<float>> GetResultNpyFast();

    std::map<std::string, py::array_t<float>> GetResultNpy();

    int get_input_split_count(py::array_t<float> np_array){
        int input_batch_size = np_array.request().shape[0];
        int flag_tmp = input_batch_size%batch_size_;
        if (flag_tmp != 0){
            return -1;
        }else{
            return input_batch_size/batch_size_;
        }
    }
    /**
     * @brief MultiEngine process
     * 
     * @param input      Input tensors
     * @return std::map<std::string, py::array_t<float>> 
     */
    std::map<std::string, py::array_t<float>> process(
        std::map<std::string, py::array_t<float>>& input_tensors);
    std::map<std::string, py::array_t<float>> process_2(
        std::map<std::string, py::array_t<float>>& input_tensors);
#endif

private:
    friend class MultiEngine;

    std::vector<Engine*> engine_list_;
    std::vector<std::thread> thread_list_;

    std::vector<int> tpu_ids_;

    std::deque<MAP_STR_TENSOR> input_map_;       //输入Tensor缓存
    std::map<int, MAP_STR_TENSOR> output_map_;   //输出Tensor缓存

    std::mutex input_data_mutex_;       //输入数据互斥
    std::mutex output_data_mutex_;      //输出数据互斥
    std::mutex stop_mutex_;             
    std::mutex process_mutex_;
    std::mutex result_mutex_;
    std::condition_variable process_cond;
    std::condition_variable result_cond;

    std::mutex print_mutex_;        //控制打印

    int had_stopped_count_;

    bool stop_thread_flag;


    int curr_data_idx_;      //当前获取数据在原始数据中的位置索引
    int max_data_idx_;       //最多输入图片的数量
    int process_times_;      //需要执行推理的次数
    int get_result_count_;   //已经推理完成的次数, 与process_times相等时表示全部推理完毕

    bool input_tensor_del_flag_;    //原始tensor是否需要释放的标志位,如果使用Tensor输入,则不需要释放,numpy输入需要释放。
    bool sys_output_flag_; 

    int batch_size_;
    std::string graph_name_;
    std::map<std::string, float> input_scale_map_;
    std::map<std::string, float> output_scale_map_;
    std::map<std::string, std::vector<int>> input_shape_map_;
    std::map<std::string, std::vector<int>> output_shape_map_;
    std::map<std::string, bm_data_type_t> input_dtype_map_;
    std::map<std::string, int> output_tensor_size_map_;
    std::map<std::string, int> input_tensor_size_map_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    bool print_flag_;
    bool print_time_flag_;

private:
    std::thread start_thread(Engine* engine_process);

    void run_inference(Engine* engine_process);

    void init_infer_threads();

    void set_stop_flag(bool value);

    bool get_stop_flag();

    int get_stop_count();

    int get_input_data(int& index, MAP_STR_TENSOR& value, bool& del_flag);

    void release();

    void dump_float32_data(const char* name, float *tensor_sys_mem, int data_len, int data_w, int data_h);

    float get_input_scale(std::string tensor_name)    {
        return input_scale_map_.find(tensor_name)->second;
    }

    float get_output_scale(std::string tensor_name) {
        return output_scale_map_.find(tensor_name)->second;
    }

    std::vector<int> get_input_shape(std::string tensor_name){
        return input_shape_map_.find(tensor_name)->second;
    }

    std::vector<int> get_output_shape(std::string tensor_name){
        return output_shape_map_.find(tensor_name)->second;
    }

    int get_output_length(std::string tensor_name){
        return output_tensor_size_map_.find(tensor_name)->second;
    }

    int get_input_length(std::string tensor_name){
        return input_tensor_size_map_.find(tensor_name)->second;
    }

    bm_data_type_t get_input_dtype(std::string tensor_name){
        return input_dtype_map_.find(tensor_name)->second;
    }

    /**
     * @brief Create input tensors(only system memory) map, according to and bmodel.
     */
    std::map<std::string, Tensor*> create_input_tensors_map_sys()
    {
        std::map<std::string, Tensor*> result;
        for(int i=0;i<input_names_.size();++i){
            std::vector<int> input_shape = get_input_shape(input_names_[i]);
            bm_data_type_t input_dtype = get_input_dtype(input_names_[i]);
    
            Tensor* temp_tensor = new Tensor(input_shape,input_dtype);
            result.insert(std::pair<std::string,Tensor*>(input_names_[i],temp_tensor));
        }
        return std::move(result);
    }

    void PrintThreadLog(std::string file_name, int line, std::string message)
    {
        if(print_flag_){
            std::lock_guard<std::mutex> lock_print(print_mutex_);
            std::cout << "# File[" << file_name << ":" << line << "], ";
            std::cout << "Thread[" << std::this_thread::get_id()<<"], "<< message << std::endl;
        }
    }
};

    MultiEngine::MultiEngine_CC::MultiEngine_CC(
        const std::string& bmodel_path,
            std::vector<int>   tpu_ids,
            bool               sys_out,
            int                graph_idx)
        :sys_output_flag_(sys_out),print_flag_(false),print_time_flag_(false){
        IOMode mode = IOMode::DEVIO; 
        if(sys_out){
            mode = IOMode::SYSO;
        }       
        tpu_ids_.clear();
        if(tpu_ids.size() <= 0){
            SPDLOG_ERROR("Input TPU List is empty!");
            exit(SAIL_ERR_ENGINE_INNER);
        }
        int tpu_count = get_available_tpu_num();
        for(int i=0;i<tpu_ids.size(); ++i){
            if(tpu_ids[i] >= tpu_count || tpu_ids[i] < 0){
                SPDLOG_ERROR("Can not find tpu '{}'.", tpu_ids[i]);
                exit(SAIL_ERR_ENGINE_INNER);
            }
            engine_list_.push_back(new Engine(bmodel_path, tpu_ids[i], mode));
            tpu_ids_.push_back(tpu_ids[i]);
        }
        if(graph_idx >= engine_list_[0]->get_graph_names().size()){
            SPDLOG_ERROR("graph_idx error: idx[{}] vs. size[{}]", graph_idx, engine_list_[0]->get_graph_names().size());
            exit(SAIL_ERR_ENGINE_INNER);
        }
        graph_name_ = engine_list_[0]->get_graph_names()[graph_idx];
        input_names_ = engine_list_[0]->get_input_names(graph_name_);
        batch_size_ = engine_list_[0]->get_input_shape(graph_name_,input_names_[0])[0];
        output_names_ = engine_list_[0]->get_output_names(graph_name_);

        for (int i=0;i<input_names_.size();++i){
            input_scale_map_.insert(std::pair<std::string, float>(input_names_[i],engine_list_[0]->get_input_scale(graph_name_, input_names_[i])));
            std::vector<int> input_shape_temp = engine_list_[0]->get_input_shape(graph_name_,input_names_[i]);
            int len_temp = 1;
            for (int j=0;j<input_shape_temp.size(); ++j){
                len_temp *= input_shape_temp[j];
            }
            input_tensor_size_map_.insert(std::pair<std::string,int>(input_names_[i],len_temp));
            input_shape_map_.insert(std::pair<std::string, std::vector<int>>(input_names_[i],input_shape_temp));
            input_dtype_map_.insert(std::pair<std::string, bm_data_type_t>(input_names_[i],engine_list_[0]->get_input_dtype(graph_name_, input_names_[i])));
        }

        for (int i=0;i<output_names_.size();++i){
            output_scale_map_.insert(std::pair<std::string, float>(output_names_[i],engine_list_[0]->get_output_scale(graph_name_, output_names_[i])));
            std::vector<int> output_shape_temp = engine_list_[0]->get_output_shape(graph_name_, output_names_[i]);
            int len_temp = 1;
            for(int j=0;j<output_shape_temp.size();++j){
                len_temp*= output_shape_temp[j];
            }
            output_tensor_size_map_.insert(std::pair<std::string, int>(output_names_[i],len_temp));
            output_shape_map_.insert(std::pair<std::string, std::vector<int>>(output_names_[i],output_shape_temp));
        }

        stop_thread_flag = false;
        had_stopped_count_ = 0;
        max_data_idx_ = MAX_BUFFER_EACH_ENGINE * tpu_count;
        curr_data_idx_ = 0;
        process_times_ = 0;
        get_result_count_ = 0;
        input_tensor_del_flag_ = false;
        init_infer_threads();
    }

    void MultiEngine::MultiEngine_CC::release()
    {
        set_stop_flag(true);
        while(true) {
            if (get_stop_count() >= tpu_ids_.size()){
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        tpu_ids_.clear();
        for(int i=0;i<engine_list_.size();++i){
            delete engine_list_[i];
            engine_list_[i] = NULL;
        }
        engine_list_.clear();
        input_map_.clear();
        output_map_.clear();
        input_scale_map_.clear();
        input_shape_map_.clear();
        input_dtype_map_.clear();
        output_scale_map_.clear();
        output_shape_map_.clear();
    }

    void MultiEngine::MultiEngine_CC::set_stop_flag(bool value)
    {
        {
            std::lock_guard<std::mutex> lock(stop_mutex_);
            stop_thread_flag = value;
        }
        {
            std::unique_lock<std::mutex> lck(process_mutex_);
	        process_cond.notify_all(); 
        }
    }

    bool MultiEngine::MultiEngine_CC::get_stop_flag()
    {
        std::lock_guard<std::mutex> lock(stop_mutex_);
        return stop_thread_flag;
    }

    int MultiEngine::MultiEngine_CC::get_stop_count()
    {
        std::lock_guard<std::mutex> lock(stop_mutex_);
        return had_stopped_count_;
    }

    int MultiEngine::MultiEngine_CC::get_input_data(int& index, MAP_STR_TENSOR& value, bool& del_flag)
    {
        PrintThreadLog(__FILE__,__LINE__,"get_input_data try lock!");
        double m_start_time = get_current_time_us();
        std::lock_guard<std::mutex> lock(input_data_mutex_);
        PrintThreadLog(__FILE__,__LINE__,"Start get_input_data!");
        if(input_map_.size() <= 0){
            PrintThreadLog(__FILE__,__LINE__,"Data buffer is empty!");
            return 1;
        }
        index = curr_data_idx_;
        value = std::move(input_map_.front());
        input_map_.pop_front();
        curr_data_idx_++;
        double m_end_time = get_current_time_us();
        char txt_temp[256] = {0};
        sprintf(txt_temp,"End get_input_data, time use: %.2f ms!",(m_end_time-m_start_time)/1000);
        PrintThreadLog(__FILE__,__LINE__,txt_temp);
        del_flag=input_tensor_del_flag_;
        return 0;
    }

    std::thread MultiEngine::MultiEngine_CC::start_thread(Engine* engine_process)
    {
        return std::thread(&MultiEngine::MultiEngine_CC::run_inference,this,engine_process);
    }

    void MultiEngine::MultiEngine_CC::init_infer_threads()
    {
        for (int i = 0; i < engine_list_.size();++i) {
            thread_list_.push_back(std::move(start_thread(engine_list_[i])));
            thread_list_[i].detach();
        }
    }

    void MultiEngine::MultiEngine_CC::run_inference(Engine* engine_process)
    {
        if (engine_process == NULL){
            SPDLOG_ERROR("Input Engine NULL.");
            exit(SAIL_ERR_ENGINE_INNER);
        }
        std::string graph_name(graph_name_);
        std::map<std::string, Tensor*> input_tensor_map = engine_process->create_input_tensors_map(graph_name);
        while(true){
            if (get_stop_flag()){
                break;
            }
            int index = -1;
            bool del_flag = false;
            MAP_STR_TENSOR in_data;
            PrintThreadLog(__FILE__,__LINE__,"Try Get Data!");
            if (get_input_data(index,in_data,del_flag) == 0){
                auto iter_temp = input_tensor_map.begin();
                while(iter_temp != input_tensor_map.end()){
                    auto iter_temp_a = in_data.find(iter_temp->first);
                    if(iter_temp_a == in_data.end()){
                        SPDLOG_ERROR("Input Tensor map error!");
                        exit(SAIL_ERR_ENGINE_INNER);
                    }
                    iter_temp_a->second->sync_to(iter_temp->second);
                    if(del_flag){
                        delete iter_temp_a->second;
                    }
                    iter_temp++;
                }
                PrintThreadLog(__FILE__,__LINE__,"Start Inference!");
                MAP_STR_TENSOR out_data = engine_process->create_output_tensors_map(graph_name);
                engine_process->process(graph_name, input_tensor_map, out_data);
                PrintThreadLog(__FILE__,__LINE__,"End Inference!");
                {
                    std::lock_guard<std::mutex> lock(output_data_mutex_);
                    // 将结果合并到map中
                    output_map_.insert(std::pair<int, MAP_STR_TENSOR>(index,std::move(out_data)));
                    //计数
                    get_result_count_++;
                }
                PrintThreadLog(__FILE__,__LINE__,"End Merge Result!");
                //发送信号
                {
                    std::unique_lock<std::mutex> lck(result_mutex_);
                    PrintThreadLog(__FILE__,__LINE__,"Result notify...");
                    result_cond.notify_all(); 
                }
            }else{
                PrintThreadLog(__FILE__,__LINE__,"Buffer Empty, Start waiting...");
                std::unique_lock<std::mutex> lck(process_mutex_);
	            process_cond.wait(lck);
            }
        }
        auto iter_temp = input_tensor_map.begin();
        while(iter_temp != input_tensor_map.end()){
            delete iter_temp->second;
            iter_temp++;
        }
        input_tensor_map.clear();
        //发送信号,防止取结果函数卡死
        {
            std::unique_lock<std::mutex> lck(result_mutex_);
            PrintThreadLog(__FILE__,__LINE__,"Result notify...");
            result_cond.notify_all(); 
        }
        {
            std::lock_guard<std::mutex> lock(stop_mutex_);
            had_stopped_count_++;
        }
        PrintThreadLog(__FILE__,__LINE__,"ThreadRelease!");
    }

    void MultiEngine::MultiEngine_CC::dump_float32_data(const char* name, float *tensor_sys_mem, int data_len, int data_w, int data_h)
    {
        FILE *fp_temp = fopen(name, "w+");
        int line_num = 0;
        for(int mm=0;mm<data_len;++mm){
            char txt_temp[32]={0};
            if(data_w >= 1){
            if (mm%data_w == data_w-1){
                sprintf(txt_temp,"%.3f\n",tensor_sys_mem[mm]);
                line_num++;
                if(line_num%data_h == 0){
                sprintf(txt_temp,"%.3f\n\n",tensor_sys_mem[mm]);
                }
            }else{
                sprintf(txt_temp,"%.3f ",tensor_sys_mem[mm]);
            }
            }else{
                sprintf(txt_temp,"%.3f ",tensor_sys_mem[mm]);
            }
            fputs(txt_temp,fp_temp);
        }
        fclose(fp_temp);
    }

    bool MultiEngine::MultiEngine_CC::PushData(std::map<std::string, Tensor*>& input, bool delete_ost_data)
    {
        {        
            std::lock_guard<std::mutex> lock(input_data_mutex_);

#if SAVE_DEBUG_DATA
            Tensor* temp_tensor = input.begin()->second;
            std::vector<int> shape = temp_tensor->shape();
            int data_len = shape_size_me(shape);

            int data_h = temp_tensor->shape()[2];
            int data_w = temp_tensor->shape()[3];
            char save_name_temp[256]={0};
            sprintf(save_name_temp,"%.0f_push.dat",get_current_time_us());
            dump_float32_data(save_name_temp, (float *)temp_tensor->sys_data(), data_len, data_w, data_h);
#endif
            input_map_.push_back(input);
            input_tensor_del_flag_ = delete_ost_data;
        }
        {
            if (print_flag_){
                SPDLOG_INFO("process_cond Notify...");
            }
            std::unique_lock<std::mutex> lck(process_mutex_);
            process_cond.notify_all(); 
        }
    }

    std::vector<std::map<std::string, Tensor*>> MultiEngine::MultiEngine_CC::GetResult()
    {
        if(process_times_ == 0){
            SPDLOG_ERROR("Input Empty!");
            exit(SAIL_ERR_ENGINE_INNER);
        }
        std::vector<std::map<std::string, Tensor*>> output(process_times_);
        while(true){
            if (get_stop_flag()){
                break;
            }
            {
                if (print_flag_){
                    SPDLOG_INFO("Start Waiting Result Notify...");
                }
                std::unique_lock<std::mutex> lck(result_mutex_);
                result_cond.wait(lck);
                if (print_flag_){
                    SPDLOG_INFO("Get Result Notify!");
                }
            }
            {
                std::lock_guard<std::mutex> lock(output_data_mutex_);
                //结果合并输出
                if (print_flag_){
                    SPDLOG_INFO("Start Merage Output...");
                }
                auto iter_temp = output_map_.begin();
                while(iter_temp != output_map_.end()){
                    output[iter_temp->first] = std::move(iter_temp->second);
                    iter_temp++;
                }
                output_map_.clear();
                
                if(get_result_count_ == process_times_){
                    process_times_ = 0;
                    get_result_count_ = 0;
                    curr_data_idx_ = 0;
                    break;
                }
            }
        }
        if (print_flag_){
            SPDLOG_INFO("Had Get All Result!");
        }
        return std::move(output);
    }

    std::vector<std::map<std::string, Tensor*>> MultiEngine::MultiEngine_CC::process(std::vector<std::map<std::string, Tensor*>>& input_tensors)
    {
        {        
            std::lock_guard<std::mutex> lock(input_data_mutex_);
            process_times_ = input_tensors.size();
        }
        for (int i = 0; i <input_tensors.size();++i){
            if (print_flag_){
                SPDLOG_INFO("PushData [{}]/[{}]",i+1,input_tensors.size());
            }
            PushData(input_tensors[i],false);
        }
        return GetResult();
    }

#ifdef PYTHON
    std::map<std::string, py::array_t<float>> MultiEngine::MultiEngine_CC::create_output_np_map(int split_num)
    {
        py::module np = py::module::import("numpy");  // like 'import numpy as np'
        std::map<std::string, py::array_t<float>> result;
        for (int i = 0; i < output_names_.size();++i){
            std::vector<int> shape = get_output_shape(output_names_[i]);
            shape[0] = shape[0]*split_num;
            py::list shape_temp = convert_intv_to_pylist(shape);
            py::array_t<float> arr = np.attr("zeros")(shape_temp, "dtype"_a="float32");
            result.insert(std::pair<std::string, py::array_t<float>>(output_names_[i],std::move(arr)));
        }
        return std::move(result);
    }

    std::map<std::string, Tensor*> MultiEngine::MultiEngine_CC::create_tensors_map(std::map<std::string, py::array_t<float>>& input_tensors, int split_idx)
    {
        std::map<std::string, Tensor*> result;
        for (int i=0;i<input_names_.size();i++){
            std::string tensor_name(input_names_[i]);

            float* np_ptr = get_data_start_ptr(input_tensors, tensor_name, split_idx, false);

            std::vector<int> input_shape = get_input_shape(tensor_name);
            bm_data_type_t input_dtype = get_input_dtype(tensor_name);
            Tensor* temp_tensor = new Tensor(input_shape,input_dtype);

            if (input_dtype == BM_FLOAT32){
                int data_len = shape_size_me(input_shape);
                memcpy(temp_tensor->sys_data(),np_ptr,data_len* sizeof(float));
            }else{
                float scale_temp = get_input_scale(tensor_name);
                temp_tensor->scale_from(np_ptr, scale_temp);
            }
            // temp_tensor->scale_from(np_ptr, scale_temp);
            result.insert(std::pair<std::string,Tensor*>(input_names_[i],temp_tensor));
        }
        return std::move(result);
    }


    int MultiEngine::MultiEngine_CC::get_nparray_length(py::array_t<float> nparray)
    {
        py::buffer_info buffer = nparray.request();
        int len_result = 1;
        for (size_t i = 0; i < buffer.shape.size(); ++i)        {
            len_result*=buffer.shape[i];
        }
        return len_result;
    }

    float* MultiEngine::MultiEngine_CC::get_data_start_ptr(std::map<std::string, py::array_t<float>> np_map, std::string tensor_name, int idx, bool is_out)
    {
        auto iter = np_map.find(tensor_name);
        if(iter == np_map.end()){
            return NULL;
        }
        float* result_ptr = static_cast<float*>(iter->second.request().ptr);
        int array_len = 0;
        if(is_out){
            array_len = get_output_length(tensor_name);
        }else{
            array_len = get_input_length(tensor_name);
        }
        int offset = array_len*idx;

        return result_ptr+offset;
    }

    std::map<std::string, py::array_t<float>> MultiEngine::MultiEngine_CC::GetResultNpyFast()
    {
        double start_create_time = get_current_time_us();
        std::map<std::string, py::array_t<float>> output_xxx = create_output_np_map(process_times_);
        double end_create_time = get_current_time_us();
        if(print_flag_){
            SPDLOG_INFO("Create Output Numpy Array Time Use {} ms\n",abs(end_create_time-start_create_time)/1000.0); 
        }
        while(true){
            if (get_stop_flag()){
                break;
            }
            {
                if (print_flag_){
                    SPDLOG_INFO("Start Waiting Result Notify...");
                }
                std::unique_lock<std::mutex> lck(result_mutex_);
                result_cond.wait(lck);
                if (print_flag_){
                    SPDLOG_INFO("Get Result Notify!");
                }
            }
            {
                std::lock_guard<std::mutex> lock(output_data_mutex_);
                //结果合并输出
                if (print_flag_){
                    SPDLOG_INFO("Start Merage Output...");
                }
                double start_merge_time = get_current_time_us();
                auto iter_temp = output_map_.begin();
                while(iter_temp != output_map_.end()){
                    int idx_temp = iter_temp->first;
                    MAP_STR_TENSOR out_data = iter_temp->second;
                    auto iter_out = out_data.begin();
                    while(iter_out != out_data.end()){
                        std::string tensor_name(iter_out->first);
                        Tensor* temp_tensor = iter_out->second;
                        if(temp_tensor->own_sys_data()){
                            float* ptr_temp = get_data_start_ptr(output_xxx, tensor_name, idx_temp);
                            float scale_temp = get_output_scale(tensor_name);
                            temp_tensor->scale_to(ptr_temp, scale_temp);
                        }
                        delete temp_tensor;
                        iter_out++;
                    }
                    iter_temp++;
                }
                double end_merge_time = get_current_time_us();
                output_map_.clear();
                if (print_flag_){
                    SPDLOG_INFO("Get Result {}/{}, time use: {}ms\n",get_result_count_,process_times_,abs(end_merge_time-start_merge_time)/1000.0);
                }
                if(get_result_count_ == process_times_){
                    process_times_ = 0;
                    get_result_count_ = 0;
                    curr_data_idx_ = 0;
                    break;
                }
            }
        }
        if (print_flag_){
            SPDLOG_INFO("Had Get All Result!");
        }
        if (print_time_flag_){
            PRINT_function_Time_ms("MultiEngine  Get Result", start_create_time,get_current_time_us())
        }
        return std::move(output_xxx);
    }

    std::map<std::string, py::array_t<float>> MultiEngine::MultiEngine_CC::GetResultNpy()
    {
        if(process_times_ == 0){
            SPDLOG_ERROR("Input Empty!");
            exit(SAIL_ERR_ENGINE_INNER);
        }
        std::vector<std::map<std::string, py::array_t<float>>> output(process_times_);
        while(true){
            if (get_stop_flag()){
                break;
            }
            {
                if (print_flag_){
                    SPDLOG_INFO("Start Waiting Result Notify...");
                }
                std::unique_lock<std::mutex> lck(result_mutex_);
                result_cond.wait(lck);
                if (print_flag_){
                    SPDLOG_INFO("Get Result Notify!");
                }
            }
            {
                std::lock_guard<std::mutex> lock(output_data_mutex_);
                //结果合并输出
                if (print_flag_){
                    SPDLOG_INFO("Start Merage Output...");
                }
                auto iter_temp = output_map_.begin();
                while(iter_temp != output_map_.end()){
                    int idx_temp = iter_temp->first;
                    MAP_STR_TENSOR out_data = iter_temp->second;
                    auto iter_out = out_data.begin();
                    while(iter_out != out_data.end()){
                        std::string tensor_name(iter_out->first);
                        Tensor* temp_tensor = iter_out->second;
                        if(temp_tensor->own_sys_data()){
                            float scale_temp = get_output_scale(tensor_name);
                            py::array_t<float> np_out = temp_tensor->scale_to(scale_temp);

                            auto item_temp = output[idx_temp].find(tensor_name);
                            if(item_temp == output[idx_temp].end()){
                                output[idx_temp].insert(std::pair<std::string, py::array_t<float>>(tensor_name,std::move(np_out)));
                            }
                        }
                        delete temp_tensor;
                        iter_out++;
                    }
                    iter_temp++;
                }
                output_map_.clear();
                if (print_flag_){
                    SPDLOG_INFO("Get Result {}/{}",get_result_count_,process_times_);
                }
                if(get_result_count_ == process_times_){
                    process_times_ = 0;
                    get_result_count_ = 0;
                    curr_data_idx_ = 0;
                    break;
                }
            }
        }
        std::map<std::string, py::array_t<float>> output_result_map;
        if(sys_output_flag_){
            double start_merged_time = get_current_time_us();
            //结果合并，name相同的合并,再组个
            for(int i=0; i<output_names_.size(); i++){
                std::string tensor_name(output_names_[i]);
                std::vector<py::array_t<float>> sig_array_list;
                for(int j=0;j<output.size();++j){
                    auto item = output[j].find(tensor_name);
                    if(item == output[j].end()){
                        SPDLOG_ERROR("Result error, not find tensor {}!",tensor_name);
                        exit(SAIL_ERR_ENGINE_INNER);
                    }
                    sig_array_list.push_back(std::move(item->second));
                }

                py::array_t<float> result_spli = merage_np_array(sig_array_list);
                output_result_map.insert(std::pair<std::string,py::array_t<float>>(tensor_name,std::move(result_spli)));
            }
            double end_merged_time = get_current_time_us();
            if(print_flag_){
                SPDLOG_INFO("Convert Result Time Use {} ms\n",abs(end_merged_time-start_merged_time)/1000.0); 
            }

        }
        if (print_flag_){
            SPDLOG_INFO("Had Get All Result!");
        }
        return std::move(output_result_map);
    }

    std::map<std::string, py::array_t<float>> MultiEngine::MultiEngine_CC::process(std::map<std::string, py::array_t<float>>& input_tensors){
        
        double process_start_time_push = get_current_time_us();
        int split_num = get_input_split_count(input_tensors.begin()->second);
        if (split_num < 0){
            SPDLOG_ERROR("Batch Size not support!");
            exit(SAIL_ERR_ENGINE_INNER);
        }
        if (split_num > max_data_idx_){
            SPDLOG_ERROR("Input batch size is too Large, max is {}, current is {}",max_data_idx_*batch_size_,split_num*batch_size_);
            exit(SAIL_ERR_ENGINE_INNER);
        }
        {
            std::lock_guard<std::mutex> lock(input_data_mutex_);
            process_times_ = split_num;
        }
        for (size_t i = 0; i < split_num; ++i)
        {
            std::map<std::string, Tensor*> input_tensor_map = create_tensors_map(input_tensors,i);
            {
                if (print_flag_){
                    SPDLOG_INFO("PushData [{}]/[{}] ",i,split_num);
                }
                PushData(input_tensor_map,true);
            }
        }
        
        if (print_flag_){
            SPDLOG_INFO("End PushData, start writing result...");
        }
        if (print_time_flag_){
            PRINT_function_Time_ms("MultiEngine Push data", process_start_time_push,get_current_time_us())
        }
        return GetResultNpyFast();

    }

    std::map<std::string, py::array_t<float>> MultiEngine::MultiEngine_CC::process_2(std::map<std::string, py::array_t<float>>& input_tensors)
    {
        std::map<int, MAP_STR_NUMPY> input_map;
        py::module np = py::module::import("numpy");  // like 'import numpy as np'
        auto item = input_tensors.begin();
        int split_num = 1;
        while (item != input_tensors.end()){
            std::string name = item->first;

            py::array_t<float> arr_c = np.attr("ascontiguousarray")(item->second, "dtype"_a="float32");
            py::buffer_info buf = item->second.request();

            int input_shape_temp = buf.shape[0];
            if (input_shape_temp % batch_size_ != 0) {
                SPDLOG_ERROR("Batch Size not support!");
                exit(SAIL_ERR_ENGINE_INNER);
            }
            split_num = input_shape_temp/batch_size_;
            // py::list split_array = split_np_array_list(item.second,split_num);
            double start_split_time = get_current_time_us();
            std::vector<py::array_t<float>> split_array = split_np_array(item->second,split_num);
            double end_split_time = get_current_time_us();

            if (print_flag_){
                SPDLOG_INFO("split data time use: {} ms!",(end_split_time - start_split_time)/1000);
            }
            

            for (int i=0;i<split_array.size();i++){
                auto temp_iter = input_map.find(i);
                if (temp_iter ==input_map.end()){

#if SAVE_DEBUG_DATA
                    int data_h = split_array[i].request().shape[2];
                    int data_w = split_array[i].request().shape[3];
                    int size = split_array[i].request().shape[0] * split_array[i].request().shape[1] * data_h *data_w;
                    char save_name_temp[256]={0};
                    sprintf(save_name_temp,"%.0f_split_%d.dat",get_current_time_us(),i);
                    dump_float32_data(save_name_temp, (float*)split_array[i].mutable_data(), size, data_w, data_h);
#endif
                    MAP_STR_NUMPY data_slice_temp;
                    data_slice_temp.insert(std::pair<std::string, py::array_t<float>>(name,std::move(split_array[i])));

#if SAVE_DEBUG_DATA
                    sprintf(save_name_temp,"%.0f_split_0_%d.dat",get_current_time_us(),i);
                    dump_float32_data(save_name_temp, (float*) data_slice_temp.begin()->second.mutable_data(), size, data_w, data_h);
#endif

                    input_map.insert(std::pair<int,MAP_STR_NUMPY>(i, std::move(data_slice_temp)));
                }else{
                    temp_iter->second.insert(std::pair<std::string,py::array_t<float>>(name,std::move(split_array[i])));
                }
            }
            item++;
        }
        {
            std::lock_guard<std::mutex> lock(input_data_mutex_);
            process_times_ = split_num;
        }
        auto iter_np = input_map.begin();
        int push_count = 0;
        while (iter_np != input_map.end()){
            MAP_STR_NUMPY data_slice_temp = iter_np->second;
            std::map<std::string, Tensor*> input_tensor_map = create_input_tensors_map_sys();
            auto item = input_tensor_map.begin();
            while (item != input_tensor_map.end()){
                std::string tensor_name = item->first;
                auto iter_slice = data_slice_temp.find(tensor_name);
                if (iter_slice == data_slice_temp.end()){
                    SPDLOG_ERROR("Error Tensor map!");
                    exit(SAIL_ERR_ENGINE_INNER);
                }
                Tensor* temp_tensor = item->second;
                py::buffer_info buf = iter_slice->second.request();
                float* ptr_start = static_cast<float*>(buf.ptr);
                if (temp_tensor->dtype() == BM_FLOAT32){
                    //memcpy
                    std::vector<int> shape = temp_tensor->shape();
                    int size = shape_size_me(shape);
                    memcpy(temp_tensor->sys_data(), ptr_start, size * sizeof(float));


#if SAVE_DEBUG_DATA
                    int data_h = shape[2];
                    int data_w = shape[3];
                    char save_name_temp[256]={0};
                    sprintf(save_name_temp,"%.0f_npy.dat",get_current_time_us());
                    dump_float32_data(save_name_temp, ptr_start, size, data_w, data_h);
#endif
                        
                }else{
                    //scale
                    float scale_temp = get_input_scale(tensor_name);
                    temp_tensor->scale_from(ptr_start,scale_temp);
                }
                item++;
            }
            //互斥 and pushdata
            {
                if (print_flag_){
                    SPDLOG_INFO("PushData [{}]/[{}}] ",push_count++,split_num);
                }
                PushData(input_tensor_map,true);
            }
            iter_np++;
        }
        
        if (print_flag_){
            SPDLOG_INFO("End PushData, start writing result...");
        }
        // return GetResultNpy();
        return GetResultNpyFast();
    }
#endif

     MultiEngine::MultiEngine_CC::~MultiEngine_CC()
     {
        release();
     }

     MultiEngine::MultiEngine(const std::string& bmodel_path,
                std::vector<int>   tpu_ids,
                bool               sys_out,
                int                graph_idx)
                :_impl(new MultiEngine_CC(bmodel_path, tpu_ids, sys_out, graph_idx))
    { }

    MultiEngine::~MultiEngine()
    { 
        delete _impl;
    }

    std::vector<int> MultiEngine::get_device_ids()
    {
        return _impl->tpu_ids_;
    }

    std::vector<std::string> MultiEngine::get_graph_names()
    {
        return _impl->engine_list_[0]->get_graph_names();
    }

    std::vector<std::string> MultiEngine::get_input_names(const std::string& graph_name)
    {
        return _impl->engine_list_[0]->get_input_names(graph_name);
    }

    std::vector<std::string> MultiEngine::get_output_names(const std::string& graph_name)
    {
        return _impl->engine_list_[0]->get_output_names(graph_name);
    }

    std::vector<int> MultiEngine::get_input_shape(const std::string& graph_name,const std::string& tensor_name)
    {
        return _impl->engine_list_[0]->get_input_shape(graph_name,tensor_name);
    }

    std::vector<int> MultiEngine::get_output_shape(const std::string& graph_name,const std::string& tensor_name)
    {
        return _impl->engine_list_[0]->get_output_shape(graph_name,tensor_name);
    }

    float MultiEngine::get_input_scale(const std::string& graph_name,const std::string& tensor_name)
    {
        return _impl->engine_list_[0]->get_input_scale(graph_name,tensor_name);
    }

    bm_data_type_t MultiEngine::get_input_dtype(const std::string& graph_name,const std::string& tensor_name)
    {
        return _impl->engine_list_[0]->get_input_dtype(graph_name,tensor_name);
    }

    std::vector<std::map<std::string, Tensor*>> MultiEngine::process(std::vector<std::map<std::string, Tensor*>>& input_tensors)
    {
        return _impl->process(input_tensors);;
    }

    void MultiEngine::set_print_flag(bool show_time)
    {
        return _impl->set_print_flag(show_time);
    }

    void MultiEngine::set_print_time(bool show_time)
    {
        return _impl->set_print_time(show_time);
    }
#ifdef PYTHON
    std::map<std::string, py::array_t<float>> MultiEngine::process(std::map<std::string, py::array_t<float>>& input_tensors)
    {
        std::map<std::string, py::array_t<float>> result = _impl->process(input_tensors);
        return std::move(result);
    }
#endif
}