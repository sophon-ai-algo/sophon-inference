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
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif  // _WIND32


namespace sail {

    int get_available_tpu_num() {
        int count = 0;
        bm_dev_getcount(&count);
        return count;
    }

#ifdef _WIN32
    int setenv(const char* name, const char* value, int overwrite)
    {
        int errcode = 0;
        if (!overwrite) {
            size_t envsize = 0;
            errcode = getenv_s(&envsize, NULL, 0, name);
            if (errcode || envsize) return errcode;
        }
        return _putenv_s(name, value);
    }
#endif

    int set_print_flag(bool print_flag)  {
        if(print_flag)
        return setenv("SAIL_PRINT_VIP_TIMES", "1", 1);
        else
        return setenv("SAIL_PRINT_VIP_TIMES", "0", 1);
    }

    int set_dump_io_flag(bool dump_io_flag){
        if(dump_io_flag)
        return setenv("SAIL_SAVE_IO_TENSORS", "1", 1);
        else
        return setenv("SAIL_SAVE_IO_TENSORS", "0", 1);
    }

    bool get_print_flag()  {
        const char *print_flag = getenv("SAIL_PRINT_VIP_TIMES");
        if(print_flag != nullptr && 0 == strcmp(print_flag,"1"))
            return true;
        return false;
    }

    // 获取系统的当前时间，单位微秒(us)
    double get_current_time_us()
    {
    #ifdef _WIN32
    // 从1601年1月1日0:0:0:000到1970年1月1日0:0:0:000的时间(单位100ns)
    #define EPOCHFILETIME   (116444736000000000UL)
        FILETIME ft;
        LARGE_INTEGER li;
        double tt = 0;
        GetSystemTimeAsFileTime(&ft);
        li.LowPart = ft.dwLowDateTime;
        li.HighPart = ft.dwHighDateTime;
        // 从1970年1月1日0:0:0:000到现在的微秒数(UTC时间)
        tt = (li.QuadPart - EPOCHFILETIME) /10;
        return tt;
    #else
        timeval tv;
        gettimeofday(&tv, 0);
        return (double)tv.tv_sec * 1000000 + (double)tv.tv_usec;
    #endif // _WIN32
        return 0;
    }

    void delete_shaptr_bm_handle_t(bm_handle_t* handle_ptr){
        SPDLOG_INFO("Start delete_shaptr_bm_handle_t!");
        delete handle_ptr;
        SPDLOG_INFO("End delete_shaptr_bm_handle_t!");
    }

    void delete_shaptr_bm_handle_t_allocated(bm_handle_t* handle_ptr){
        SPDLOG_INFO("Start delete_shaptr_bm_handle_t_allocated!");
        bm_dev_free(handle_ptr[0]);
        delete handle_ptr;
        SPDLOG_INFO("End delete_shaptr_bm_handle_t_allocated!");
    }

    std::shared_ptr<bm_handle_t> make_shaptr_bm_handle_t(int dev_id){
        std::shared_ptr<bm_handle_t> ptr_temp = std::shared_ptr<bm_handle_t>(new bm_handle_t[1],delete_shaptr_bm_handle_t_allocated);
        if (bm_dev_query(dev_id)) {
            printf("Error: Invalid tpu id: %d!\n", dev_id);
            exit(SAIL_ERR_DEVICE_INIT);
        }
        bm_dev_request(&ptr_temp.get()[0], dev_id);
        return std::move(ptr_temp);
    }

    std::shared_ptr<bm_handle_t> make_shaptr_bm_handle_t(bm_handle_t handle){
        std::shared_ptr<bm_handle_t> ptr_temp = std::shared_ptr<bm_handle_t>(new bm_handle_t[1],delete_shaptr_bm_handle_t);
        ptr_temp.get()[0] = handle;
        return std::move(ptr_temp);
    }

    void get_sail_version(char* sail_version){
        char s_month[5];
        int month, day, year;
        int hour, minute, second;
        static const char month_names[] = "JanFebMarAprMayJunJulAugSepOctNovDec";
        sscanf(__DATE__, "%s %d %d", s_month, &day, &year);
        month = (strstr(month_names, s_month)-month_names)/3;
        sscanf(__TIME__, "%d:%d:%d", &hour, &minute, &second);
        sprintf(sail_version,"master(%d%02d%02d_%02d%02d%02d)",year,month+1,day,hour, minute, second);
    }

    class Handle::Handle_CC{
    public:
        explicit Handle_CC()
            : dev_id_(-1) {};

        explicit Handle_CC(bm_handle_t handle)  
            : dev_id_(-2) {
                handle_ = make_shaptr_bm_handle_t(handle);
            };

        explicit Handle_CC(int dev_id) 
            : dev_id_(-1) {
            handle_ = make_shaptr_bm_handle_t(dev_id);
            dev_id_ = dev_id;
        };

        ~Handle_CC(){
            free();
        };

        /**
         * @brief Free inner bm_handle_t.
         */
        void free();

        std::shared_ptr<bm_handle_t> handle_;
        int dev_id_;
    };

    void Handle::Handle_CC::free(){
        dev_id_ = -1;
    }

    Handle::Handle() : _impl(new Handle_CC()) {}

    Handle::Handle(bm_handle_t handle) : _impl(new Handle_CC(handle)) {}

    Handle::Handle(int dev_id) : _impl(new Handle_CC(dev_id)) {}

    Handle::Handle(const Handle &other): _impl(new Handle_CC()) {
        _impl->handle_ = other._impl->handle_;
        _impl->dev_id_ = other._impl->dev_id_;
    }

    Handle &Handle::operator=(const Handle &other) {
        if (this != &other) {
            _impl->free();
            _impl->handle_ = other._impl->handle_;
            _impl->dev_id_ = other._impl->dev_id_;
        }
        return *this;
    }

    Handle::~Handle() {
        delete _impl;
    }

    bm_handle_t Handle::data() {
        // SPDLOG_INFO("num_list.use_count: {}",_impl->handle_.use_count());
        return _impl->handle_.get()[0];
    }

    int Handle::get_device_id() {
        return _impl->dev_id_;
    }

    std::string Handle::get_sn() {
        char sn_num[18]={0};
        bm_status_t r_value = bm_get_sn(this->data(),sn_num);
        if(r_value == BM_SUCCESS){
            return std::string(sn_num);
        }
        return "Error";
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

    class Tensor::Tensor_CC{
    public:
        Tensor_CC();

        ~Tensor_CC(){};

        explicit Tensor_CC(
            const Handle&           handle,
            const std::vector<int>& shape,
            bm_data_type_t          dtype,
            bool                    own_sys_data,
            bool                    own_dev_data);

        explicit Tensor_CC(
            const std::vector<int>& shape,
            bm_data_type_t          dtype);

        void free();
        
        void reset(const std::vector<int>& shape, bm_data_type_t dtype);

        void reset_sys_data(void *data, std::vector<int> &shape);

        void reset_dev_data(bm_device_mem_t data);

        void sync_s2d();

        void sync_s2d(int size);

        void sync_d2s();

        void sync_d2s(int size);

        void sync_from(Tensor_CC* src);

        void sync_to(Tensor_CC* dst);

#ifdef PYTHON

        explicit Tensor_CC(Handle handle, 
                        bm_data_type_t dtype,
                        const pybind11::buffer_info& buf, 
                        bool own_sys_data);

        void update_data(const pybind11::buffer_info& buf, int type_size);
        
#endif
    private:
        friend class Tensor;
        /// Handle instance.
        Handle handle_;

        /// Data type
        bm_data_type_t dtype_;

        /// Shape of the tensor.
        std::vector<int> shape_;

        /// Indicator of whether own the data pointer in system memory.
        bool own_sys_data_{false};
        bool own_sys_data_is_mmap_{false};

        /// Indicator of whether own the device memory struct.
        bool own_dev_data_{false};

        /// Data pointer in system memory of the tensor.
        void* sys_data_{nullptr};

        /// Instance of device memory structure.
        bm_device_mem_t dev_data_ {};
        /// data size
        uint32_t data_size_ {0};

    private:
        /**
         * @brief Judge if a tensor shape is valid.
         *
         * @param shape Shape of a tensor
         * @return True for valid and flase for invalid..
         */
        bool shape_is_valid(const std::vector<int>& shape);
    };

    bool Tensor::Tensor_CC::shape_is_valid(const std::vector<int>& shape){
        if (shape.empty()) {
            return false;
        }
        if (std::any_of(shape.begin(), shape.end(), [](int i) { return i <= 0; })) {
            return false;
        }
        return true;
    }

    Tensor::Tensor_CC::Tensor_CC():own_sys_data_(false),own_sys_data_is_mmap_(false),
        own_dev_data_(false),sys_data_(nullptr), dev_data_({}), data_size_(0){}

    Tensor::Tensor_CC::Tensor_CC(
        const Handle& handle,
        const std::vector<int>& shape,
        bm_data_type_t dtype,
        bool own_sys_data,
        bool own_dev_data)
        :handle_(handle), shape_(shape), dtype_(dtype),own_sys_data_(own_sys_data), 
        own_dev_data_(own_dev_data), sys_data_(nullptr), dev_data_({}), data_size_(0){
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

                int c = 0;
                void* value = (void*)&c;
                ret = bm_memset_device_ext(handle_.data(), value, 1, dev_data_);
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("bm_memset_device failed, return={}", ret);
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
                memset(sys_data_,0,data_size_);
#endif
            }
            //} else {
            //  spdlog::error("tensor shape is not valid!");
            //  exit(SAIL_ERR_TENSOR_INIT);
        }        
    }

    Tensor::Tensor_CC::Tensor_CC(const std::vector<int>& shape,bm_data_type_t dtype)
    : shape_(shape), dtype_(dtype), own_sys_data_(true), own_dev_data_(false), 
    sys_data_(nullptr), dev_data_({}), data_size_(0), own_sys_data_is_mmap_(false){
        int type_size = get_type_size(dtype);
        data_size_ = std::accumulate(shape_.begin(), shape_.end(),
                                     type_size, std::multiplies<int>());
        if (data_size_ > 0) {
            sys_data_ = malloc(data_size_);
            memset(sys_data_,0,data_size_);
        }
    }

    void Tensor::Tensor_CC::free() {
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

    void Tensor::Tensor_CC::reset(const std::vector<int> &shape, bm_data_type_t dtype) {
        if (!shape_is_valid(shape)) {
            spdlog::error("Invalid tensor shape!");
            exit(SAIL_ERR_TENSOR_INNER);
        }

        int ret = 0;
        int size_shape = std::accumulate(shape.begin(), shape.end(),
                                         1, std::multiplies<int>());

        int data_size = size_shape * get_type_size(dtype);

        if (data_size_ != data_size) {
            if (own_dev_data_) {
                bm_free_device(handle_.data(), dev_data_);
                ret = bm_malloc_device_byte_heap_mask(handle_.data(), &dev_data_, 7, data_size);
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("bm_malloc_device_byte err={}, size={}", ret, data_size);
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

    void Tensor::Tensor_CC::reset_sys_data(void *data, std::vector<int> &shape)
    {
        reset(shape, dtype_);
        if (own_dev_data_) {
            double process_start_time_befor = get_current_time_us();
            if (sys_data_ != nullptr) {
                memcpy(sys_data_, data, data_size_);
            } else {
                spdlog::error("Cannot reset_sys_data when own_dev_data is true.");
                exit(SAIL_ERR_TENSOR_INNER);
            }
            PRINT_TIME_MS("memcpy_cpu_to_cpu_0", process_start_time_befor)
        } else if (own_sys_data_) {
            double process_start_time_befor = get_current_time_us();
            if (sys_data_ != nullptr) {
                memcpy(sys_data_, data, data_size_);
            }
            PRINT_TIME_MS("memcpy_cpu_to_cpu_1", process_start_time_befor)
        } else {
            sys_data_ = data;
        }
    }

    void Tensor::Tensor_CC::reset_dev_data(bm_device_mem_t data)  {
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

    void Tensor::Tensor_CC::sync_d2s(int size) {
        if (own_dev_data_) {
            if (sys_data_) {
                if (!own_sys_data_is_mmap_) {
                    double process_start_time_d2s = get_current_time_us();
                    if (bm_memcpy_d2s_partial(handle_.data(), sys_data_, dev_data_, size) != 0) {
                        SPDLOG_ERROR("bm_memcpy_d2s_partial() err");
                    }
                    PRINT_TIME_MS("bm_memcpy_d2s_partial", process_start_time_d2s)
                } else {
                    bm_mem_invalidate_device_mem(handle_.data(), &dev_data_);
                }
            }
        } else {
            if (sys_data_ != nullptr) {
                if (!own_sys_data_is_mmap_) {
                    double process_start_time_d2s = get_current_time_us();
                    if (BM_SUCCESS != bm_memcpy_d2s_partial(handle_.data(), sys_data_, dev_data_, size)) {
                        SPDLOG_ERROR("bm_memcpy_d2s_partial() err");
                    }
                    PRINT_TIME_MS("bm_memcpy_d2s_partial", process_start_time_d2s)
                } else {
                    bm_mem_invalidate_device_mem(handle_.data(), &dev_data_);
                }
            }
        }
    }

    void Tensor::Tensor_CC::sync_s2d(int size) {
        if (sys_data_) {
            if (own_sys_data_is_mmap_) {
                bm_mem_flush_partial_device_mem(handle_.data(), &dev_data_, 0, size);
            } else {
                double process_start_time = get_current_time_us();
                int ret = bm_memcpy_s2d_partial(handle_.data(), dev_data_, sys_data_, size);
                PRINT_TIME_MS("bm_memcpy_s2d_partial",process_start_time);

                if (ret != BM_SUCCESS) {
                    spdlog::error("bm_memcpy_s2d_partial(sys_data=%p, dev_data.addr=%p, size=%d)\n",
                                  sys_data_, (void *) dev_data_.u.device.device_addr, size);
                }
            }
        }
    }

    void Tensor::Tensor_CC::sync_d2s() {
        sync_d2s(data_size_);
    }

    void Tensor::Tensor_CC::sync_s2d() {
        if (sys_data_) {
            if (own_sys_data_is_mmap_) {
                bm_mem_flush_device_mem(handle_.data(), &dev_data_);
            } else {
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(handle_.data(), dev_data_, sys_data_);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
            }
        }
    }

    void Tensor::Tensor_CC::sync_from(Tensor_CC* src)   {
        if (dtype_ != src->dtype_) {
            spdlog::error("sync_from: data type not match!");
            exit(SAIL_ERR_TENSOR_INNER);
        }
        int size = std::accumulate(shape_.begin(), shape_.end(),
                                   1, std::multiplies<int>());
        auto src_shape = src->shape_;
        int src_size = std::accumulate(src_shape.begin(), src_shape.end(),
                                       1, std::multiplies<int>());
        if (size != src_size) {
            spdlog::error("sync_from: tensor size not match!");
            exit(SAIL_ERR_TENSOR_INNER);
        }
        auto src_handle = src->handle_.data();
        auto dtype_size = get_type_size(dtype_);
        void *src_sys_data = src->sys_data_;
        bool src_own_dev_data = src->own_dev_data_;
        bm_device_mem_t src_dev_data = src->dev_data_;
        if (sys_data_) {
            if (src_own_dev_data) {
                double process_start_time_d2s = get_current_time_us();
                bm_memcpy_d2s(src_handle, sys_data_, src_dev_data);
                PRINT_TIME_MS("bm_memcpy_d2s", process_start_time_d2s)
            } else if (src_sys_data) {
                memcpy(sys_data_, src_sys_data, size * dtype_size);
            }
            if (own_dev_data_) {
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(handle_.data(), dev_data_, sys_data_);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
            }
        } else if (own_dev_data_) {
            if (src_sys_data) {
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(handle_.data(), dev_data_, src_sys_data);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
            } else if (src_own_dev_data) {
                void *tmp = malloc(size * dtype_size);
                double process_start_time_d2s = get_current_time_us();
                bm_memcpy_d2s(handle_.data(), tmp, src_dev_data);
                PRINT_TIME_MS("bm_memcpy_d2s", process_start_time_d2s)
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(handle_.data(), dev_data_, tmp);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
                std::free(tmp);
            }
        }
    }

    void Tensor::Tensor_CC::sync_to(Tensor_CC* dst){
        if (dtype_ != dst->dtype_) {
            spdlog::error("dst_from: data type not match!");
            exit(SAIL_ERR_TENSOR_INNER);
        }
        int size = std::accumulate(shape_.begin(), shape_.end(),
                                   1, std::multiplies<int>());
        auto dst_shape = dst->shape_;
        int dst_size = std::accumulate(dst_shape.begin(), dst_shape.end(),
                                       1, std::multiplies<int>());
        if (size != dst_size) {
            spdlog::error("dst_from: tensor size not match!");
            exit(SAIL_ERR_TENSOR_INNER);
        }
        auto dst_handle = dst->handle_.data();
        auto dtype_size = get_type_size(dtype_);
        void *dst_sys_data = dst->sys_data_;
        bool dst_own_dev_data = dst->own_dev_data_;
        bm_device_mem_t dst_dev_data = dst->dev_data_;
        if (dst_sys_data) {
            if (own_dev_data_) {
                double process_start_time_d2s = get_current_time_us();
                bm_memcpy_d2s(handle_.data(), dst_sys_data, dev_data_);
                PRINT_TIME_MS("bm_memcpy_d2s", process_start_time_d2s)
            } else if (sys_data_) {
                memcpy(dst_sys_data, sys_data_, size * dtype_size);
            }
            if (dst_own_dev_data) {
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(dst_handle, dst_dev_data, sys_data_);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
            }
        } else if (dst_own_dev_data) {
            if (sys_data_) {
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(dst_handle, dst_dev_data, sys_data_);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
            } else if (own_dev_data_) {
                void *tmp = malloc(size * dtype_size);
                double process_start_time_d2s = get_current_time_us();
                bm_memcpy_d2s(handle_.data(), tmp, dev_data_);
                PRINT_TIME_MS("bm_memcpy_d2s", process_start_time_d2s)
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(handle_.data(), dst_dev_data, tmp);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
                std::free(tmp);
            }
        }
    }


#ifdef PYTHON

    Tensor::Tensor_CC::Tensor_CC(Handle handle, 
                    bm_data_type_t dtype,
                    const pybind11::buffer_info& buf, 
                    bool own_sys_data)
        :handle_(handle),dtype_(dtype),own_sys_data_(own_sys_data),
        own_dev_data_(true),sys_data_(nullptr),dev_data_({}){
        if (buf.ndim < 1) {
            spdlog::error("Invalid tensor shape!");
            exit(SAIL_ERR_TENSOR_INIT);
        }
        shape_.clear();
        for (auto it : buf.shape) {
            shape_.push_back(static_cast<int>(it));
        }

        void* numpy_ptr = buf.ptr;

        pybind11::array_t<float> arr_float;
        pybind11::array_t<int8_t> arr_int8_t;
        pybind11::array_t<uint8_t> arr_uint8_t;
        pybind11::array_t<int32_t> arr_int32_t;

        pybind11::module np = pybind11::module::import("numpy");  // like 'import numpy as np'
        if(BM_FLOAT32 == dtype){
            pybind11::array_t<float> buf_temp(buf);
            arr_float = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_float.request().ptr;
        }else if(BM_INT8 == dtype){
            pybind11::array_t<int8_t> buf_temp(buf);
            arr_int8_t = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_int8_t.request().ptr;
        }else if(BM_UINT8 == dtype){
            pybind11::array_t<uint8_t> buf_temp(buf);
            arr_uint8_t = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_uint8_t.request().ptr;
        }else if(BM_INT32 == dtype){
            pybind11::array_t<int32_t> buf_temp(buf);
            arr_int32_t = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_int32_t.request().ptr;
        }else{
            SPDLOG_ERROR("Input Data Type not supported: {}",dtype);
            exit(SAIL_ERR_TENSOR_INIT);
        }

        // alloc dev_mem
        int data_size = std::accumulate(shape_.begin(), shape_.end(),
                        get_type_size(dtype), std::multiplies<int>());
        data_size_ = data_size;
        if (own_dev_data_) {
            int ret = bm_malloc_device_byte_heap_mask(handle_.data(), &dev_data_, 7, data_size);
            if (BM_SUCCESS != ret) {
                SPDLOG_ERROR("bm_malloc_device_byte_heap_mask() err={}", ret);
                exit(SAIL_ERR_BMCV_INIT);
            }
        }
        if (own_sys_data_) {
#ifndef IS_SOC_MODE
            sys_data_ = new uint8_t[data_size];
#else
            bm_mem_mmap_device_mem(handle_.data(), &dev_data_, (unsigned long long*)&sys_data_);
            own_sys_data_is_mmap_ = true;
#endif
            memcpy(sys_data_, numpy_ptr, data_size); 
        }else{
            double process_start_time = get_current_time_us();
            bm_memcpy_s2d(handle_.data(), dev_data_, numpy_ptr);
            int ret = bm_memcpy_s2d_partial(handle_.data(), dev_data_, numpy_ptr, data_size);
            
            if (ret != BM_SUCCESS) {
                spdlog::error("bm_memcpy_s2d_partial(sys_data=%p, dev_data.addr=%p, size=%d)\n",
                    numpy_ptr, (void *) dev_data_.u.device.device_addr, data_size);
            }

            PRINT_TIME_MS("bm_memcpy_s2d_partial",process_start_time);
        }
    }

    void Tensor::Tensor_CC::update_data(const pybind11::buffer_info& buf, int type_size)
    {
        if (buf.ndim != shape_.size()) {
            SPDLOG_ERROR("Invalid tensor shape dims {} vs. {}!",shape_.size(),buf.ndim);
            exit(SAIL_ERR_TENSOR_INNER);
        }
        std::vector<int> shape;
        for (auto it : buf.shape) {
            shape.push_back(static_cast<int>(it));
        }

        for (int i=0;i<shape.size();++i){ 
            if(shape[i] != shape_[i]){
                char str_shape_old[256]={};
                char str_shape_new[256]={};
                sprintf(str_shape_old,"[");
                sprintf(str_shape_new,"[");
                for (int j=0;j<shape.size();++j){
                    sprintf(&str_shape_new[strlen(str_shape_new)],"%d,",shape[j]);
                    sprintf(&str_shape_old[strlen(str_shape_old)],"%d,",shape_[j]);
                }
                str_shape_new[strlen(str_shape_new)-1] = ']';
                str_shape_old[strlen(str_shape_old)-1] = ']';
                SPDLOG_ERROR("Invalid tensor shape {} vs. {}!",str_shape_old,str_shape_new);
                exit(SAIL_ERR_TENSOR_INNER);
            }
        }
        size_t type_size_tmep = 1;
        if (dtype_ == BM_FLOAT32) {
            type_size_tmep = sizeof(float);
        } else if (dtype_ == BM_INT8) {
            type_size_tmep = sizeof(int8_t);
        } else if (dtype_ == BM_UINT8) {
            type_size_tmep = sizeof(uint8_t);
        } else if (dtype_ == BM_INT32) {
            type_size_tmep = sizeof(int32_t);
        }

        int old_size = std::accumulate(shape_.begin(), shape_.end(),
            type_size_tmep, std::multiplies<int>());
        int new_size = std::accumulate(shape.begin(), shape.end(),
            type_size, std::multiplies<int>());
        if (new_size > old_size) {
            spdlog::error("Data size exceeds tensor size!");
            exit(SAIL_ERR_TENSOR_INNER);
        }

        void* numpy_ptr = buf.ptr;

        pybind11::array_t<float> arr_float;
        pybind11::array_t<int8_t> arr_int8_t;
        pybind11::array_t<uint8_t> arr_uint8_t;
        pybind11::array_t<int32_t> arr_int32_t;
        pybind11::module np = pybind11::module::import("numpy");  // like 'import numpy as np'
        if(BM_FLOAT32 == dtype_){
            pybind11::array_t<float> buf_temp(buf);
            arr_float = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_float.request().ptr;
        }else if(BM_INT8 == dtype_){
            pybind11::array_t<int8_t> buf_temp(buf);
            arr_int8_t = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_int8_t.request().ptr;
        }else if(BM_UINT8 == dtype_){
            pybind11::array_t<uint8_t> buf_temp(buf);
            arr_uint8_t = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_uint8_t.request().ptr;
        }else if(BM_INT32 == dtype_){
            pybind11::array_t<int32_t> buf_temp(buf);
            arr_int32_t = np.attr("ascontiguousarray")(buf_temp);
            numpy_ptr = arr_int32_t.request().ptr;
        }

        if (own_sys_data_){

#ifndef IS_SOC_MODE
    //    if (own_sys_data_) {
    //      std::free(sys_data_);
    //      own_sys_data_ = false;
    //    }
    //    sys_data_ = buf.ptr;
            memcpy(sys_data_, numpy_ptr, new_size);

#else
            memcpy(sys_data_, numpy_ptr, new_size);
#endif
        } else if(own_dev_data_){
            double process_start_time = get_current_time_us();
            // bm_memcpy_s2d(handle_.data(), dev_data_, buf.ptr);
            int ret = bm_memcpy_s2d_partial(handle_.data(), dev_data_, numpy_ptr, new_size);
            
            if (ret != BM_SUCCESS) {
                spdlog::error("bm_memcpy_s2d_partial(sys_data=%p, dev_data.addr=%p, size=%d)\n",
                    numpy_ptr, (void *) dev_data_.u.device.device_addr, new_size);
            }
            PRINT_TIME_MS("bm_memcpy_s2d_partial",process_start_time);
        }else{
            spdlog::error("Can not found device memory or host memory!");
            exit(SAIL_ERR_TENSOR_INNER);
        }
    }

#endif

    Tensor::Tensor(
            const Handle &handle,
            const std::vector<int> &shape,
            bm_data_type_t dtype,
            bool own_sys_data,
            bool own_dev_data)
            : _impl(new Tensor_CC(handle,shape,dtype,own_sys_data,own_dev_data)){}

    Tensor::Tensor(
            const std::vector<int> &shape,
            bm_data_type_t dtype)
            : _impl(new Tensor_CC(shape,dtype)){}

    Tensor::Tensor(const Tensor &other):_impl(new Tensor_CC()) {
        _impl->handle_ = other._impl->handle_;
        _impl->dtype_ = other._impl->dtype_;
        _impl->shape_ = other._impl->shape_;
        _impl->own_sys_data_ = other._impl->own_sys_data_;
        _impl->own_dev_data_ = other._impl->own_dev_data_;
        _impl->own_sys_data_is_mmap_ = other._impl->own_sys_data_is_mmap_;
        int type_size = get_type_size(_impl->dtype_);
        _impl->data_size_ = std::accumulate(_impl->shape_.begin(), _impl->shape_.end(),
                                     type_size, std::multiplies<int>());
        if (_impl->own_dev_data_) {
            int ret = bm_malloc_device_byte_heap_mask(_impl->handle_.data(), &_impl->dev_data_, 7, _impl->data_size_);
            if (BM_SUCCESS != ret) {
                SPDLOG_ERROR("bm_malloc_device_byte_heap_mask() err={}", ret);
                exit(SAIL_ERR_BMCV_INIT);
            }
        }

#ifndef IS_SOC_MODE
        if (_impl->own_sys_data_) {
            _impl->sys_data_ = malloc(_impl->data_size_);
            memcpy(_impl->sys_data_, other._impl->sys_data_,_impl->data_size_);
            double process_start_time = get_current_time_us();
            bm_memcpy_s2d(_impl->handle_.data(), _impl->dev_data_, _impl->sys_data_);
            PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
        } else {
            void *tmp = malloc(_impl->data_size_);
            double process_start_time_d2s = get_current_time_us();
            bm_memcpy_d2s(_impl->handle_.data(), tmp, other._impl->dev_data_);
            PRINT_TIME_MS("bm_memcpy_d2s", process_start_time_d2s)
            double process_start_time = get_current_time_us();
            bm_memcpy_s2d(_impl->handle_.data(), _impl->dev_data_, tmp);
            PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
            std::free(tmp);
        }
#else
        if (_impl->own_sys_data_) {
          if (_impl->own_dev_data_) {
            bm_mem_mmap_device_mem(_impl->handle_.data(), &_impl->dev_data_,
                                   (unsigned long long*)&_impl->sys_data_);
            _impl->own_sys_data_is_mmap_ = true;
          } else {
            _impl->sys_data_ = malloc(_impl->data_size_);
          }
          memcpy(_impl->sys_data_, other._impl->sys_data_, _impl->data_size_);
        } else {
            void* tmp = malloc(_impl->data_size_);
            double process_start_time_d2s = get_current_time_us();
            bm_memcpy_d2s(_impl->handle_.data(), tmp, other._impl->dev_data_);
            PRINT_TIME_MS("bm_memcpy_d2s", process_start_time_d2s)
            double process_start_time = get_current_time_us();
            bm_memcpy_s2d(_impl->handle_.data(), _impl->dev_data_, tmp);
            PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
            std::free(tmp);
        }
#endif
    }

    Tensor::Tensor(Tensor &&other):_impl(new Tensor_CC()) {
        *this = std::move(other);
    }

    Tensor &Tensor::operator=(const Tensor &other) {
        if (this != &other) {
            free();
            _impl->handle_ = other._impl->handle_;
            _impl->dtype_ = other._impl->dtype_;
            _impl->shape_ = other._impl->shape_;
            _impl->own_sys_data_ = other._impl->own_sys_data_;
            _impl->own_dev_data_ = other._impl->own_dev_data_;
            _impl->own_sys_data_is_mmap_ = other._impl->own_sys_data_is_mmap_;
            int type_size = get_type_size(_impl->dtype_);
            _impl->data_size_ = std::accumulate(_impl->shape_.begin(), _impl->shape_.end(),
                                         type_size, std::multiplies<int>());
            if (_impl->own_dev_data_) {
                int ret = bm_malloc_device_byte_heap_mask(_impl->handle_.data(), &_impl->dev_data_, 7, _impl->data_size_);
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("bm_malloc_device_byte_heap_mask() err={}", ret);
                    exit(SAIL_ERR_BMCV_INIT);
                }
            }
#ifndef IS_SOC_MODE
            if (_impl->own_sys_data_) {
                _impl->sys_data_ = malloc(_impl->data_size_);
                memcpy(_impl->sys_data_, other._impl->sys_data_, _impl->data_size_);
                if (_impl->own_dev_data_) {
                    double process_start_time = get_current_time_us();
                    bm_memcpy_s2d(_impl->handle_.data(), _impl->dev_data_, _impl->sys_data_);
                    PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
                }
            } else {
                void *tmp = malloc(_impl->data_size_);
                double process_start_time_d2s = get_current_time_us();
                bm_memcpy_d2s(_impl->handle_.data(), tmp, other._impl->dev_data_);
                PRINT_TIME_MS("bm_memcpy_d2s", process_start_time_d2s)
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(_impl->handle_.data(), _impl->dev_data_, tmp);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
                std::free(tmp);
            }
#else
            if (_impl->own_sys_data_) {
              if (_impl->own_dev_data_) {
                bm_mem_mmap_device_mem(_impl->handle_.data(), &_impl->dev_data_,
                                       (unsigned long long*)&_impl->sys_data_);
                _impl->own_sys_data_is_mmap_ = true;
              } else {
                _impl->sys_data_ = malloc(_impl->data_size_);
              }
              memcpy(_impl->sys_data_, other._impl->sys_data_, _impl->data_size_);
            } else {
                void* tmp = malloc(_impl->data_size_);
                double process_start_time_d2s = get_current_time_us();
                bm_memcpy_d2s(_impl->handle_.data(), tmp, other._impl->dev_data_);
                PRINT_TIME_MS("bm_memcpy_d2s", process_start_time_d2s)
                double process_start_time = get_current_time_us();
                bm_memcpy_s2d(_impl->handle_.data(), _impl->dev_data_, tmp);
                PRINT_TIME_MS("bm_memcpy_s2d",process_start_time);
                std::free(tmp);
            }
#endif
        }
        return *this;
    }

    Tensor &Tensor::operator=(Tensor &&other) {
        if (this != &other) {
            std::swap(_impl->handle_, other._impl->handle_);
            std::swap(_impl->dtype_, other._impl->dtype_);
            std::swap(_impl->shape_, other._impl->shape_);
            std::swap(_impl->own_sys_data_, other._impl->own_sys_data_);
            std::swap(_impl->own_dev_data_, other._impl->own_dev_data_);
            std::swap(_impl->sys_data_, other._impl->sys_data_);
            std::swap(_impl->dev_data_, other._impl->dev_data_);
            std::swap(_impl->data_size_, other._impl->data_size_);
            std::swap(_impl->own_sys_data_is_mmap_, other._impl->own_sys_data_is_mmap_);
        }
        return *this;
    }

    void Tensor::free() {
        if(_impl) _impl->free();
    }

    Tensor::~Tensor() {
        if(_impl){
            free();
            delete _impl;
        }
    }

    Handle &Tensor::get_handle() {
        return _impl->handle_;
    }

    const std::vector<int> &Tensor::shape() const {
        return _impl->shape_;
    }

    bm_data_type_t Tensor::dtype() const {
        return _impl->dtype_;
    }

    void Tensor::reset(const std::vector<int> &shape, bm_data_type_t dtype) {
        return _impl->reset(shape, dtype);
    }

    void Tensor::reshape(const std::vector<int> &shape) {
        return reset(shape, _impl->dtype_);
    }

    void Tensor::reset_sys_data(void *data, std::vector<int> &shape) {
        return _impl->reset_sys_data(data, shape);
    }

    void Tensor::reset_dev_data(bm_device_mem_t data) {
        return _impl->reset_dev_data(data);
    }

    bool &Tensor::own_sys_data() {
        return _impl->own_sys_data_;
    }

    bool &Tensor::own_dev_data() {
        return _impl->own_dev_data_;
    }

    bm_device_mem_t Tensor::dev_data() {
        return _impl->dev_data_;
    }

    void *Tensor::sys_data() {
        return _impl->sys_data_;
    }

    void Tensor::sync_s2d() {
        return _impl->sync_s2d();
    }

    void Tensor::sync_s2d(int size) {
        return _impl->sync_s2d(size);
    }

    void Tensor::sync_d2s() {
        return _impl->sync_d2s();
    }

    void Tensor::sync_d2s(int size) {
        return _impl->sync_d2s(size);
    }

    void Tensor::sync_from(Tensor *src) {
        return _impl->sync_from(src->_impl);
    }

    void Tensor::sync_to(Tensor *dst) {
        return _impl->sync_to(dst->_impl);
    }

    void Tensor::scale_from(float *src, float scale) {
        int size = std::accumulate(_impl->shape_.begin(), _impl->shape_.end(),
                                   1, std::multiplies<int>());
        scale_from(src, scale, size);
    }

#if USE_ASM_SSE

    void Tensor::scale_from(float *src, float scale, int size) {
        if (nullptr == _impl->sys_data_) {
            spdlog::error("When call scale_from, own_sys_data must be true");
            exit(EXIT_FAILURE);
        }
        double process_start_time_scale = get_current_time_us();
        AnyScale_SSE(src, BM_FLOAT32, _impl->sys_data_, _impl->dtype_, scale, size);
        PRINT_TIME_MS("AnyScale_SSE", process_start_time_scale)
    }

    void Tensor::scale_from_int32(int32_t *src, float scale, int size) {
        if (nullptr == _impl->sys_data_) {
            spdlog::error("When call scale_from_int32, own_sys_data must be true");
            exit(EXIT_FAILURE);
        }
        double process_start_time_scale = get_current_time_us();
        AnyScale_SSE(src, BM_INT32, _impl->sys_data_, _impl->dtype_, scale, size);
        PRINT_TIME_MS("AnyScale_SSE", process_start_time_scale)
    }

    void Tensor::scale_to(float *dst, float scale, int size) {

        if (nullptr == _impl->sys_data_) {
            SPDLOG_ERROR("When call scale_to, own_sys_data must be true");
            exit(EXIT_FAILURE);
        }

        double process_start_time_scale = get_current_time_us();
        AnyScale_SSE(_impl->sys_data_, _impl->dtype_, dst, BM_FLOAT32, scale, size);
        PRINT_TIME_MS("AnyScale_SSE", process_start_time_scale)
    }

#else // don't use asm language.

    void Tensor::scale_from(float *src, float scale, int size) {
        double process_start_time_scale = get_current_time_us();
        AnyScale(src, BM_FLOAT32, _impl->sys_data_, _impl->dtype_, scale, size);
        PRINT_TIME_MS("AnyScale", process_start_time_scale)
    }

    void Tensor::scale_to(float *dst, float scale, int size) {
        double process_start_time_scale = get_current_time_us();
        AnyScale(_impl->sys_data_, _impl->dtype_, dst, BM_FLOAT32, scale, size);
        PRINT_TIME_MS("AnyScale", process_start_time_scale)
    }


    void Tensor::scale_from_int32(int32_t* src, float scale, int size) {
      if (nullptr == _impl->sys_data_) {
        spdlog::error("When call scale_from_int32, own_sys_data must be true");
        exit(EXIT_FAILURE);
      }
        double process_start_time_scale = get_current_time_us();
        AnyScale(src, BM_INT32, _impl->sys_data_, _impl->dtype_, scale, size);
        PRINT_TIME_MS("AnyScale", process_start_time_scale)
    }
#endif

    void Tensor::scale_to(float *dst, float scale) {
        int size = std::accumulate(_impl->shape_.begin(), _impl->shape_.end(),
                                   1, std::multiplies<int>());
        scale_to(dst, scale, size);
    }

    void Tensor::memory_set(int c)
    {
        if(_impl->sys_data_){
            memset(_impl->sys_data_,c,_impl->data_size_);
        }
        if(_impl->dev_data_.u.device.device_addr != 0){
            void* value = (void*)&c;
            bm_memset_device_ext(_impl->handle_.data(), value, 1, _impl->dev_data_);
        }
    }

#ifdef PYTHON
    Tensor::Tensor(Handle handle, pybind11::array_t<float>&   data)
        :_impl (new Tensor_CC(handle, BM_FLOAT32, data.request(), 1)){
    }

    Tensor::Tensor(Handle handle, pybind11::array_t<int8_t>&  data)
        :_impl (new Tensor_CC(handle, BM_INT8, data.request(), 1)){
    }

    Tensor::Tensor(Handle handle, pybind11::array_t<uint8_t>& data)
        :_impl (new Tensor_CC(handle, BM_UINT8, data.request(), 1)){
    }
    
    Tensor::Tensor(Handle handle, pybind11::array_t<int32_t>& data)
        :_impl (new Tensor_CC(handle, BM_INT32, data.request(), 1)){
    }

    Tensor::Tensor(Handle handle, pybind11::array_t<float>&   data, bool own_sys_data)
        :_impl (new Tensor_CC(handle, BM_FLOAT32, data.request(), own_sys_data)){
    }

    Tensor::Tensor(Handle handle, pybind11::array_t<int8_t>&  data, bool own_sys_data)
        :_impl (new Tensor_CC(handle, BM_INT8, data.request(), own_sys_data)){
    }

    Tensor::Tensor(Handle handle, pybind11::array_t<uint8_t>& data, bool own_sys_data)
        :_impl (new Tensor_CC(handle, BM_UINT8, data.request(), own_sys_data)){
    }
    
    Tensor::Tensor(Handle handle, pybind11::array_t<int32_t>& data, bool own_sys_data)
        :_impl (new Tensor_CC(handle, BM_INT32, data.request(), own_sys_data)){
    }

    void Tensor::scale_from(pybind11::array_t<float> &data, float scale) {
        auto buf = data.request();
        int size = 1;
        for (auto it : buf.shape) {
            size *= static_cast<int>(it);
        }
        int tensor_size = std::accumulate(_impl->shape_.begin(), _impl->shape_.end(),
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
        int tensor_size = std::accumulate(_impl->shape_.begin(), _impl->shape_.end(),
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
        for (auto v : _impl->shape_) {
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
        int tensor_size = std::accumulate(_impl->shape_.begin(), _impl->shape_.end(),
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
        void *data = _impl->sys_data_;
        if (_impl->sys_data_ == nullptr) {
            if (_impl->dev_data_.u.device.device_addr == 0) {
                SPDLOG_ERROR("asnumpy: sys_data=null and dev_data is null!");
                exit(SAIL_ERR_TENSOR_INNER);
            }
            ptr.reset(new uint8_t[_impl->data_size_]);
            data = ptr.get();
            double process_start_time_d2s = get_current_time_us();
            if (BM_SUCCESS != bm_memcpy_d2s_partial(_impl->handle_.data(), data, _impl->dev_data_, _impl->data_size_)) {
                SPDLOG_ERROR("bm_memcpy_d2s_partial() err");
            }
            PRINT_TIME_MS("bm_memcpy_d2s_partial", process_start_time_d2s)
        }

        // fill numpy array
        pybind11::ssize_t item_size = 1;
        std::string format;
        if (_impl->dtype_ == BM_FLOAT32) {
            item_size = sizeof(float);
            format = pybind11::format_descriptor<float>::format();
        } else if (_impl->dtype_ == BM_INT8) {
            item_size = sizeof(int8_t);
            format = pybind11::format_descriptor<int8_t>::format();
        } else if (_impl->dtype_ == BM_UINT8) {
            item_size = sizeof(uint8_t);
            format = pybind11::format_descriptor<uint8_t>::format();
        } else if (_impl->dtype_ == BM_INT32) {
            item_size = sizeof(int32_t);
            format = pybind11::format_descriptor<int32_t>::format();
        }

        pybind11::ssize_t ndim = _impl->shape_.size();
        std::vector<pybind11::ssize_t> shape;
        for (auto it : _impl->shape_) {
            shape.push_back(it);
        }
        std::vector<pybind11::ssize_t> stride;
        for (size_t i = 1; i < _impl->shape_.size(); i++) {
            pybind11::ssize_t inner_stride = std::accumulate(shape.begin() + i,
                                                             shape.end(), item_size,
                                                             std::multiplies<pybind11::ssize_t>());
            stride.push_back(inner_stride);
        }
        stride.push_back(item_size);

        pybind11::buffer_info output_buf(data, item_size, format,
                                         ndim, shape, stride);
        if (_impl->dtype_ == BM_FLOAT32) {
            return std::move(pybind11::array_t<float>(output_buf));
        } else if (_impl->dtype_ == BM_INT8) {
            return std::move(pybind11::array_t<int8_t>(output_buf));
        } else if (_impl->dtype_ == BM_UINT8) {
            return std::move(pybind11::array_t<uint8_t>(output_buf));
        } else if (_impl->dtype_ == BM_INT32) {
            return std::move(pybind11::array_t<int32_t>(output_buf));
        } else {
            return pybind11::cast<pybind11::none>(Py_None);;
        }
    }

    pybind11::object Tensor::asnumpy(const std::vector<int> &shape) {
        std::unique_ptr<uint8_t[]> ptr;
        void *data = _impl->sys_data_;
        if (_impl->sys_data_ == nullptr) {
            if (_impl->dev_data_.u.device.device_addr == 0) {
                SPDLOG_ERROR("asnumpy: sys_data=null and dev_data is null!");
                exit(SAIL_ERR_TENSOR_INNER);
            }
            ptr.reset(new uint8_t[_impl->data_size_]);
            data = ptr.get();
            double process_start_time_d2s = get_current_time_us();
            if (BM_SUCCESS != bm_memcpy_d2s_partial(_impl->handle_.data(), data, _impl->dev_data_, _impl->data_size_)) {
                SPDLOG_ERROR("bm_memcpy_d2s_partial() err");
            }
            PRINT_TIME_MS("bm_memcpy_d2s_partial", process_start_time_d2s)
        }
        // fill numpy array
        pybind11::ssize_t item_size = 1;
        std::string format;
        if (_impl->dtype_ == BM_FLOAT32) {
            item_size = sizeof(float);
            format = pybind11::format_descriptor<float>::format();
        } else if (_impl->dtype_ == BM_INT8) {
            item_size = sizeof(int8_t);
            format = pybind11::format_descriptor<int8_t>::format();
        } else if (_impl->dtype_ == BM_UINT8) {
            item_size = sizeof(uint8_t);
            format = pybind11::format_descriptor<uint8_t>::format();
        } else if (_impl->dtype_ == BM_INT32) {
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
        if (_impl->dtype_ == BM_FLOAT32) {
            return std::move(pybind11::array_t<float>(output_buf));
        } else if (_impl->dtype_ == BM_INT8) {
            return std::move(pybind11::array_t<int8_t>(output_buf));
        } else if (_impl->dtype_ == BM_UINT8) {
            return std::move(pybind11::array_t<uint8_t>(output_buf));
        } else if (_impl->dtype_ == BM_INT32) {
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
        dst[0] = (long) _impl->sys_data_;
        //memcpy(dst, &ldata, 1 * sizeof(int));
        //printf("pysys_data sysd=0x%x\n",dst[0]);
        return std::move(ndarray);
    }

    void Tensor::update_data(const pybind11::buffer_info& buf, int type_size)
    {
        return _impl->update_data(buf,type_size);
    }

#endif

}  // namespace sail
