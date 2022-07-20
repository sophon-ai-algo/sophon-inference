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

#include "cvwrapper.h"
#include <spdlog/spdlog.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <fstream>
#include "internal.h"
#include "tensor.h"

#ifdef USE_OPENCV

#include "opencv2/opencv.hpp"

#endif

#ifdef PYTHON
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eval.h>

using namespace pybind11::literals;
#endif

typedef unsigned char u8;

static int IMAGE_H = 1024;
static int IMAGE_W = 1920;

#define AUTO_PTR(name, type, size) std::unique_ptr<type[]> up##name(new type[size]);\
                                   type *name=up##name.get();
namespace sail {
    inline bool string_start_with(const std::string &s, const std::string &head) {
        return s.compare(0, head.size(), head) == 0;
    }
    /**
     * @brief Get the decoder env int object
     * 
     * @param env_name 
     * @param value 
     * @return bool, if get value return true,else return false.
     */
    bool get_decoder_env_int(std::string env_name, int& value);

    /**
     * @brief Get the decoder env string object
     * 
     * @param env_name 
     * @param env_value return value
     * @return bool, if get value return true,else return false.
     */
    bool get_decoder_env_string(std::string env_name, std::string &env_value);

    bm_data_type_t get_bm_data_type_sail(bm_image_data_format_ext fmt) {
        std::string sfmt;
        switch (fmt) {
            case DATA_TYPE_EXT_FLOAT32:
                return BM_FLOAT32;
            case DATA_TYPE_EXT_1N_BYTE_SIGNED:
                return BM_INT8;
            case DATA_TYPE_EXT_1N_BYTE:
                return BM_UINT8;
            case DATA_TYPE_EXT_4N_BYTE_SIGNED:
                sfmt = "DATA_TYPE_EXT_4N_BYTE_SIGNED";
                break;
            case DATA_TYPE_EXT_4N_BYTE:
                sfmt = "DATA_TYPE_EXT_4N_BYTE";
                break;
            default:
                assert(0);
        }
        spdlog::error("No matching bm_data_type_t from bm_image_data_format_ext ({}).", sfmt);
        exit(SAIL_ERR_BMCV_TRANS);
        // return BM_FLOAT32;
    }

    bm_image_data_format_ext get_bm_image_data_format_sail(bm_data_type_t dtype) {
        std::string sfmt;
        switch (dtype) {
            case BM_FLOAT32:
                return DATA_TYPE_EXT_FLOAT32;
            case BM_INT8:
                return DATA_TYPE_EXT_1N_BYTE_SIGNED;
            case BM_UINT8:
                return DATA_TYPE_EXT_1N_BYTE;
            case BM_FLOAT16:
                sfmt = "BM_FLOAT16";
                break;
            case BM_INT16:
                sfmt = "BM_INT16";
                break;
            case BM_UINT16:
                sfmt = "BM_UINT16";
                break;
            case BM_INT32:
                sfmt = "BM_INT32";
                break;
            case BM_UINT32:
                sfmt = "BM_UINT32";
                break;
            default:
                assert(0);
        }
        spdlog::error("No matching bm_image_data_format_ext from bm_data_type_t ({}).", sfmt);
        exit(SAIL_ERR_BMCV_TRANS);
        // return DATA_TYPE_EXT_FLOAT32;
    }


#ifdef USE_OPENCV

    inline bool check_shape(const std::vector<cv::Mat> &mats) {
        if (mats.empty()) return false;
        int h = mats[0].rows;
        int w = mats[0].cols;
        for (size_t i = 1; i < mats.size(); i++) {
            if (mats[i].rows != h || mats[i].cols != w) return false;
        }
        return true;
    }

    int get_cv_depth(bm_data_type_t dtype) {
        std::string sfmt;
        switch (dtype) {
            case BM_FLOAT32:
                return CV_32F;
            case BM_INT8:
                return CV_8S;
            case BM_UINT8:
                return CV_8U;
            case BM_FLOAT16:
                sfmt = "BM_FLOAT16";
                break;
            case BM_INT16:
                return CV_16S;
            case BM_UINT16:
                return CV_16U;
            case BM_INT32:
                return CV_32S;
            case BM_UINT32:
                sfmt = "BM_UINT32";
                break;
            default:
                assert(0);
        }
        spdlog::error("No matching cv depth from bm_data_type_t ({}).", sfmt);
        exit(SAIL_ERR_BMCV_TRANS);
        // return CV_32F;
    }

    template<typename T>
    void mat_to_tensor_(std::vector<cv::Mat> &mats, Tensor &tensor) {
        if (!check_shape(mats)) {
            spdlog::error("mat_to_tensor(): check mat shape failed.");
            exit(SAIL_ERR_BMCV_TRANS);
        }
        if (!tensor.own_sys_data() || !tensor.own_dev_data()) {
            spdlog::error("mat_to_tensor(): tensor should own sys & dev memory.");
            exit(SAIL_ERR_BMCV_TRANS);
        }

        int n = static_cast<int>(mats.size());
        int c = 3;
        int h = mats[0].rows;
        int w = mats[0].cols;
        int depth = get_cv_depth(tensor.dtype());

        tensor.reshape({n, c, h, w});

        T *addr = reinterpret_cast<T *>(tensor.sys_data());
        for (auto mat : mats) {
            if (mat.depth() != depth) {
                mat.convertTo(mat, CV_MAKETYPE(depth, 3));
            }
            std::vector<cv::Mat> channels;
            for (int i = 0; i < mat.channels(); i++) {
                channels.emplace_back(h, w, CV_MAKETYPE(mat.depth(), 1), addr);
                addr += h * w;
            }
            cv::split(mat, channels);
        }
    }

    void mat_to_tensor(std::vector<cv::Mat> &mats, Tensor &tensor) {
        if (mats.empty()) return;
        if (mats[0].depth() == CV_32F) {
            mat_to_tensor_<float>(mats, tensor);
        } else {
            mat_to_tensor_<int8_t>(mats, tensor);
        }
    }

    void mat_to_tensor(cv::Mat &mat, sail::Tensor &tensor) {
        std::vector<cv::Mat> mats{mat};
        mat_to_tensor(mats, tensor);
    }

#endif

    inline bool is_jpg_file(const std::string &filename) {
        size_t len = filename.size();
        return (filename.compare(len - 3, 3, "jpg") == 0 ||
                filename.compare(len - 3, 3, "JPG") == 0 ||
                filename.compare(len - 4, 4, "jpeg") == 0 ||
                filename.compare(len - 4, 4, "JPEG") == 0);
    }

    PaddingAtrr::PaddingAtrr(unsigned int crop_start_x,
                            unsigned int crop_start_y,
                            unsigned int crop_width,
                            unsigned int crop_height,
                            unsigned char padding_value_r,
                            unsigned char padding_value_g,
                            unsigned char padding_value_b):
      dst_crop_stx(crop_start_x),dst_crop_sty(crop_start_y),dst_crop_w(crop_width),dst_crop_h(crop_height),
      padding_r(padding_value_r),padding_g(padding_value_g),padding_b(padding_value_b){};

    PaddingAtrr::PaddingAtrr(const PaddingAtrr& other)
    {
        dst_crop_stx = other.dst_crop_stx;
        dst_crop_sty = other.dst_crop_sty;
        dst_crop_w = other.dst_crop_w;
        dst_crop_h = other.dst_crop_h;
        padding_r = other.padding_r;
        padding_g = other.padding_g;
        padding_b = other.padding_b;
    }

    void PaddingAtrr::set_stx(unsigned int stx) {
        dst_crop_stx = stx;
    }

    void PaddingAtrr::set_sty(unsigned int sty) {
        dst_crop_sty = sty;
    }

    void PaddingAtrr::set_w(unsigned int w) {
        dst_crop_w = w;
    }

    void PaddingAtrr::set_h(unsigned int h) {
        dst_crop_h = h;
    }

    void PaddingAtrr::set_r(unsigned int r) {
        padding_r = r;
    }

    void PaddingAtrr::set_g(unsigned int g) {
        padding_g = g;
    }

    void PaddingAtrr::set_b(unsigned int b) {
        padding_b = b;
    }

    int set_decoder_env(std::string env_name, std::string env_value)
    {
        std::string env_name_temp = std::string("SAIL_DECODER_")+env_name;
        return setenv(env_name_temp.c_str(), env_value.c_str(), 1);
    }

    bool get_decoder_env_int(std::string env_name, int& value)
    {
        std::string env_name_temp = std::string("SAIL_DECODER_")+env_name;
        const char *e_value = getenv(env_name_temp.c_str());
        if(e_value != nullptr){
            value = atoi(e_value);
            return true;
        }
        return false;
    }

    bool get_decoder_env_string(std::string env_name, std::string &env_value)
    {
        std::string env_name_temp = std::string("SAIL_DECODER_")+env_name;
        const char *e_value = getenv(env_name_temp.c_str());
        if(e_value != nullptr){
            env_value = std::string(e_value);
            return true;
        }
        return false;
    }

#ifdef USE_FFMPEG

    class Decoder::Decoder_CC{
    public:
        explicit Decoder_CC(
            const std::string& file_path,
            bool               compressed,
            int                tpu_id);

        ~Decoder_CC();

        int decode_jpeg(Handle& handle, bm_image& image);
        
        int read_(Handle& handle, bm_image& image);

        float get_fps() const;

        void release();

    private:
        friend class Decoder;

        /**
         * @brief Grabs the next frame.
         *
         * @param frame Reference of frame to be read to
         * @return True for success and false for failure
         */
        bool grab(Frame& frame);
        /**
         * @brief Convert frame with format of AV_PIX_FMT_NV12 to bm_image.
         *
         * @param image Reference of BMImage to convert to
         */
        void nv12_frame_to_image(Handle& handle, bm_image& image);
        void convert_to_yuv420p();
        void reset_decode(const std::string& file_path,
                        bool   compressed = true,
                        int    tpu_id = 0);
        void reconnect();

        /// Path to the Decoder file.
        std::string file_path_;
        /// TPU ID
        int tpu_id_;
        /// Pointer to an AVFormatContext instance.
        AVFormatContext* fmt_ctx_;
        /// Pointer to an AVCodecContext instance.
        AVCodecContext* video_dec_ctx_;
        /// Pointer to an AVStream instance.
        AVStream* video_stream_;
        /// Index of stream.
        int video_stream_idx_;
        /// An AVPacket instance.
        AVPacket pkt_;
        /// Count of Decoder frames.
        int video_frame_count_;
        /// Number of decoded frame of a packet.
        int got_frame_;
        /// Hight of frame.
        int height_;
        /// Width of frame.
        int width_;
        /// bm_handle
        bm_handle_t handle_;
        /// Decoded frame
        Frame frame_;
        /// Indicator of whether the frame is compressed.
        bool compressed_;
        /// Status of opening the Decoder file.
        bool opened_;
        /// Indicator of whether the input source is rtsp stream.
        bool is_rtsp_;
        /// Indicator of whether the input source is jpg image file.
        bool is_jpeg_file_;
        /// Flag of whether to read to end of the video.
        bool end_of_file_;
        int errcnt_;

        std::string refcounted_frames_value;
        std::string extra_frame_buffer_num_value;
        std::string rtsp_transport_value;
        std::string stimeout_value;
        std::string rtsp_flags_value;
        int buffer_size_value;
        int max_delay_value;
        std::string probesize;
        std::string analyzeduration;
        int get_ffmpeg_valuse()
            {
                refcounted_frames_value = "1";
                extra_frame_buffer_num_value = "2";
                rtsp_transport_value = "tcp";
                stimeout_value = "20000000";
                rtsp_flags_value = "prefer_tcp";
                buffer_size_value = 1024000;
                max_delay_value = 500000;
                probesize = "0";
                analyzeduration = "0";
                get_decoder_env_string("refcounted_frames", refcounted_frames_value);
                get_decoder_env_string("extra_frame_buffer_num", extra_frame_buffer_num_value);
                get_decoder_env_string("rtsp_transport", rtsp_transport_value);
                get_decoder_env_string("stimeout", stimeout_value);
                get_decoder_env_string("rtsp_flags", rtsp_flags_value);
                get_decoder_env_int("buffer_size",buffer_size_value);
                get_decoder_env_int("max_delay",max_delay_value);
                get_decoder_env_string("probesize", probesize);         //400
                get_decoder_env_string("analyzeduration", analyzeduration); //100
                return 0;
            }
        
#ifdef USE_OPENCV
          cv::Mat m1;
#endif
    };

    Decoder::Decoder_CC::Decoder_CC(
            const std::string &file_path,
            bool compressed,
            int tpu_id)
            : handle_(nullptr), file_path_(file_path), tpu_id_(tpu_id), fmt_ctx_(nullptr),
              video_dec_ctx_(nullptr), video_stream_(nullptr), video_stream_idx_(-1),
              video_frame_count_(0), got_frame_(0), height_(0), width_(0),
              compressed_(compressed), is_rtsp_(false), is_jpeg_file_(false), errcnt_(0),
              opened_(false), end_of_file_(false) {
        if (is_jpg_file(file_path_)) {
            is_jpeg_file_ = true;
            opened_ = true;
            return;
        }

        std::cout << "decoder ctor: filepath=" << file_path << std::endl;
        // register all formats and codecs
        av_register_all();
        avdevice_register_all();
        AVDictionary *opts = NULL;
#ifndef IS_SOC_MODE
        av_dict_set_int(&opts, "pcie_board_id", tpu_id_, 0);
#endif
        if (file_path_.compare(0, 5, "rtsp:") == 0) {
            is_rtsp_ = true;
            get_ffmpeg_valuse();

            SPDLOG_INFO("refcounted_frames: {}",refcounted_frames_value);
            SPDLOG_INFO("extra_frame_buffer_num:{}",extra_frame_buffer_num_value);
            SPDLOG_INFO("rtsp_transport: {}",rtsp_transport_value);
            SPDLOG_INFO("stimeout: {}",stimeout_value);
            SPDLOG_INFO("rtsp_flags: {}",rtsp_flags_value);
            SPDLOG_INFO("buffer_size: {}",buffer_size_value);
            SPDLOG_INFO("max_delay: {}",max_delay_value);
            SPDLOG_INFO("probesize: {}",probesize);
            SPDLOG_INFO("analyzeduration: {}",analyzeduration);

            avformat_network_init();
            // Init the decoders, with reference counting
            av_dict_set(&opts, "refcounted_frames", refcounted_frames_value.c_str(), 0);
            // frame buffer set,same as opencv, ost is 20
            av_dict_set(&opts, "extra_frame_buffer_num", extra_frame_buffer_num_value.c_str(), 0);
            // set tcp
            av_dict_set(&opts, "rtsp_transport", rtsp_transport_value.c_str(), 0);
            // set timeout (same as opencv),ost is 10000000
            av_dict_set(&opts, "stimeout", stimeout_value.c_str(), 0);

            // add same as opencv
            av_dict_set(&opts, "rtsp_flags", rtsp_flags_value.c_str(), 0);
            av_dict_set_int(&opts, "buffer_size", buffer_size_value, 0);
            av_dict_set_int(&opts, "max_delay", max_delay_value, 0);

            if (probesize != "0" && analyzeduration != "0"){
                av_dict_set(&opts, "probesize", probesize.c_str(), 0);
                av_dict_set(&opts, "analyzeduration", analyzeduration.c_str(), 0);
            }
        }
        if (!is_jpeg_file_ && compressed_) {
            // set compressed output
            av_dict_set(&opts, "output_format", "101", 0);
        }
        //av_log_set_level(AV_LOG_TRACE);
        std::cout << "open filepath=" << file_path << std::endl;
        AVInputFormat *input_fmt = nullptr;
        if (string_start_with(file_path, "/dev/video")) {
            input_fmt = av_find_input_format("video4linux2");
            if (input_fmt == nullptr) {
                printf("ERROR:can't find format: video4linux2\n");
            } else {
                printf("find video4linux2 success!\n");
                const char *pixfmt = getenv("SAIL_CAP_PIXFMT");
                if (pixfmt == nullptr) pixfmt = "mjpeg";
                av_dict_set(&opts, "pixel_format", pixfmt, 0);
            }
        }

        // open input file, and allocate format context
        int ret = avformat_open_input(&fmt_ctx_, file_path_.c_str(),
                                      input_fmt, &opts);
        if (ret < 0) {
            spdlog::error("Failed to open input file: {}", file_path_);
            //exit(SAIL_ERR_DECODER_INIT);
            throw std::runtime_error("Failed to open input file");
        }
        // retrieve stream information
        ret = avformat_find_stream_info(fmt_ctx_, nullptr);
        if (ret < 0) {
            spdlog::error("Failed to find stream information.");
            //exit(SAIL_ERR_DECODER_INIT);
            throw std::runtime_error("Failed to find stream information");
        }

        ret = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (ret < 0) {
            spdlog::error("Failed to find a video stream.");
            //exit(SAIL_ERR_DECODER_INIT);
            throw std::runtime_error("Failed to find a video stream");
        }
        video_stream_idx_ = ret;
        video_stream_ = fmt_ctx_->streams[video_stream_idx_];
        video_dec_ctx_ = video_stream_->codec;
        AVCodec *dec = avcodec_find_decoder(video_dec_ctx_->codec_id);
        if (!dec) {
            spdlog::error("Failed to find codec.");
            //exit(SAIL_ERR_DECODER_INIT);
            throw std::runtime_error("Failed to find codec");
        }
        ret = avcodec_open2(video_dec_ctx_, dec, &opts);
        if (ret < 0) {
            spdlog::error("Failed to open codec");
            //exit(SAIL_ERR_DECODER_INIT);
            throw std::runtime_error("Failed to open codec");

        }
        height_ = video_dec_ctx_->height;
        width_ = video_dec_ctx_->width;

        // initialize packet, set data to NULL, let the demuxer fill it
        av_init_packet(&pkt_);
        pkt_.data = nullptr;
        pkt_.size = 0;
        opened_ = true;
        // destroy avdict
        av_dict_free(&opts);
    }

    Decoder::Decoder_CC::~Decoder_CC() {
        release();
    }

    bool Decoder::Decoder_CC::grab(Frame &frame) {
        if (end_of_file_) {
            return false;
        }
        AVFrame *p_frame = frame.get();
        bool valid = false;
        while (!valid) {
            av_packet_unref(&pkt_);
            int ret = av_read_frame(fmt_ctx_, &pkt_);
            if (ret == AVERROR(EAGAIN)) {
                continue;
            }
            if (ret < 0) {
                if (ret == static_cast<int>(AVERROR_EOF)) {
                    end_of_file_ = true;
                    return false;
                }
                break;
            }
            if (pkt_.stream_index != video_stream_idx_) {
                av_packet_unref(&pkt_);
                continue;
            }
            // decode video frame
            ret = avcodec_decode_video2(video_dec_ctx_, p_frame, &got_frame_, &pkt_);
            if (got_frame_) {
                valid = true;
            }
        }
        if (valid) {
            ++video_frame_count_;
        }
        return valid;
    }

    int Decoder::Decoder_CC::decode_jpeg(Handle &handle, bm_image &image) {
        std::ifstream filestr;
        filestr.open(file_path_, std::ios::binary);
        std::filebuf *pbuf = filestr.rdbuf();
        size_t size = pbuf->pubseekoff(0, std::ios::end, std::ios::in);
        char *buffer = new char[size];
        pbuf->pubseekpos(0, std::ios::in);
        pbuf->sgetn((char *) buffer, size);
        bm_handle_t bm_handle = handle.data();
        bm_image image1;
        memset(&image1, 0, sizeof(image1));
        int ret = bmcv_image_jpeg_dec(bm_handle, (void **) &buffer, &size, 1, &image1);
        if (BM_SUCCESS != ret) {
#ifdef USE_OPENCV
          spdlog::info(
              "bmcv_image_jpeg_dec err={}, fallback to software decode ...\n",
              ret);
          std::vector<char> pic(buffer, buffer+size);
        //   cv::Mat m1;
        //   m1.allocator = cv::hal::getAllocator();
          cv::imdecode(pic, cv::IMREAD_COLOR, &m1, handle.get_device_id());
          memset(&image, 0, sizeof(image));
          ret = cv::bmcv::toBMI(m1, &image);
          if (ret != BM_SUCCESS) {
            spdlog::error("cv::bmcv::toBMI() err {},{}", __FILE__, __LINE__);
            ret = BM_NOT_SUPPORTED;
          }else {
            ret = BM_SUCCESS;
          }
#else
          ret = BM_NOT_SUPPORTED;
#endif
        } else {
            if (image1.width % 2 != 0 || image1.height % 2 != 0) {
                // width and height align 2
                int new_w = FFALIGN(image1.width, 2);
                int new_h = FFALIGN(image1.height, 2);
                bm_image image2;
                int stride = FFALIGN(new_w, SAIL_ALIGN);
                ret = bm_image_create(bm_handle, new_h, new_w, FORMAT_BGR_PLANAR,
                                DATA_TYPE_EXT_1N_BYTE, &image2, &stride);
                assert(ret == 0);
                ret = bm_image_alloc_dev_mem_heap_mask(image2, 6);
                if (ret != 0) {
                    SPDLOG_ERROR("bm_image_alloc_dev_mem_heap_mask err={}", ret);
                    bm_image_destroy(image2);
                    bm_image_destroy(image1);
                    return ret;
                }
                ret = bmcv_image_vpp_csc_matrix_convert(bm_handle, 1, image1, &image2, CSC_YPbPr2RGB_BT601);
                if (0 != ret) {
                    SPDLOG_ERROR("bmcv_image_vpp_convert error={}", ret);
                    print_image(image1, "src:");
                    print_image(image2, "dst:");
                }
                image = image2;
                bm_image_destroy(image1);
            } else {
                image = image1;
            }
        }

        filestr.close();
        delete[] buffer;
        return ret;
    }

    void Decoder::Decoder_CC::nv12_frame_to_image(Handle &handle, bm_image &image) {
        AVFrame *p_frame = frame_.get();
        bm_image temp_img;
        if (compressed_) {
            bm_image_create(handle.data(),
                            frame_.get_height(),
                            frame_.get_width(),
                            FORMAT_COMPRESSED,
                            DATA_TYPE_EXT_1N_BYTE,
                            &temp_img);

            // calculate physical address of frame
            bm_device_mem_t input_addr[4];
            int size = frame_.get_height() * p_frame->linesize[4];
            input_addr[0] = bm_mem_from_device((unsigned long long) p_frame->data[6], size);
            size = (frame_.get_height() / 2) * p_frame->linesize[5];
            input_addr[1] = bm_mem_from_device((unsigned long long) p_frame->data[4], size);
            size = p_frame->linesize[6];
            input_addr[2] = bm_mem_from_device((unsigned long long) p_frame->data[7], size);
            size = p_frame->linesize[7];
            input_addr[3] = bm_mem_from_device((unsigned long long) p_frame->data[5], size);
            bm_image_attach(temp_img, input_addr);

        } else {
            int stride[2];
            stride[0] = p_frame->linesize[4];
            stride[1] = p_frame->linesize[5];
            bm_image_create(handle.data(),
                            frame_.get_height(),
                            frame_.get_width(),
                            FORMAT_NV12,
                            DATA_TYPE_EXT_1N_BYTE,
                            &temp_img,
                            stride);
            // calculate physical address of yuv data
            bm_device_mem_t input_addr[2];
            int size = p_frame->height * stride[0];
            input_addr[0] = bm_mem_from_device((unsigned long long) p_frame->data[4], size);
            size = p_frame->height / 2 * stride[1];
            input_addr[1] = bm_mem_from_device((unsigned long long) p_frame->data[5], size);
            // attach memory to bm_image
            bm_image_attach(temp_img, input_addr);
        }

        if (image.image_private == nullptr || image.width == 0 || image.height == 0) {
            bm_image_create(handle.data(),
                            frame_.get_height(),
                            frame_.get_width(),
                            FORMAT_BGR_PLANAR,
                            DATA_TYPE_EXT_1N_BYTE,
                            &image);
          bm_image_alloc_dev_mem(image, BMCV_HEAP1_ID);
        }

        int ret = 0;
        const char *env_csc_YPbPr2RGB = getenv("SAIL_USE_CSC_YPbPr2RGB");
        bool use_csc_YPbPr2RGB = env_csc_YPbPr2RGB != nullptr ? 0==strcmp(env_csc_YPbPr2RGB, "1"): false;
        if (use_csc_YPbPr2RGB) {
            ret = bmcv_image_vpp_csc_matrix_convert(handle.data(), 1, temp_img, &image, CSC_YPbPr2RGB_BT601);
        }
        ret = bmcv_image_vpp_convert(handle.data(), 1, temp_img, &image);
        if (ret != 0) {
            SPDLOG_ERROR("bm_image_vpp_convert err={}", ret);
            print_image(temp_img, " src:");
            print_image(image, " dst:");
        }
        bm_image_destroy(temp_img);
    }

    int Decoder::Decoder_CC::read_(Handle &handle, bm_image &image) {
        handle_ = handle.data();
        
        int curr_id = bm_get_devid(handle_);
        if (curr_id != tpu_id_){
            SPDLOG_ERROR("Input Handle error, Decoder TPU:{} vs. Handle TPU:{}",tpu_id_,curr_id);
            exit(SAIL_ERR_DECODER_READ);
        }
        if (is_jpeg_file_) {
            return decode_jpeg(handle, image);
        }
        if (!opened_){
            return BM_ERR_PARAM;
        }
        //reconnect
        if (errcnt_ >= 20) {
            reset_decode(file_path_, compressed_, tpu_id_);
        }

        int ret = grab(frame_);
        if (!ret) {
            errcnt_++;
            return 1;
        }
        AVFrame *p_frame = frame_.get();

        if (p_frame->height <= 0 || p_frame->width <= 0) {
            spdlog::error("fatal error: {} {}", p_frame->width, p_frame->height);
            errcnt_++;
            return 1;
        }

        if (p_frame->format != AV_PIX_FMT_NV12 &&
            p_frame->format != AV_PIX_FMT_YUV420P &&
            p_frame->format != AV_PIX_FMT_YUVJ420P) {
            //Convert to YUV420P
            convert_to_yuv420p();
            p_frame = frame_.get();
        }

        // create bm_image with YUV-nv12 format
        if (p_frame->format == AV_PIX_FMT_NV12) {
            nv12_frame_to_image(handle, image);
            // SPDLOG_INFO("decode with nv12");
        } else {
            // SPDLOG_INFO("decode with other");
            bm_image_create(handle_, p_frame->height, p_frame->width,
                FORMAT_YUV420P,
                DATA_TYPE_EXT_1N_BYTE,
                &image);

            if (p_frame->data[4] != nullptr) {
                bm_mem_desc_t src_plane[4];
                src_plane[0] = bm_mem_from_device((uint64_t) p_frame->data[6], p_frame->linesize[6]);
                src_plane[1] = bm_mem_from_device((uint64_t) p_frame->data[4], p_frame->linesize[4] * p_frame->height);
                src_plane[2] = bm_mem_from_device((uint64_t) p_frame->data[7], p_frame->linesize[7]);
                src_plane[3] = bm_mem_from_device((uint64_t) p_frame->data[5],
                                                  p_frame->linesize[4] * p_frame->height / 2);
                bm_image_attach(image, src_plane);
            } else {
                void *src_plane[4];
                ret = bm_image_alloc_dev_mem_heap_mask(image, 6);
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("bm_image_alloc_dev_mem_heap_mask err={}", ret);
                    exit(EXIT_FAILURE);
                }
                // need copy to device memory
                src_plane[0] = p_frame->data[0];
                src_plane[1] = p_frame->data[1];
                src_plane[2] = p_frame->data[2];
                src_plane[3] = p_frame->data[3];

                ret = bm_image_copy_host_to_device(image, src_plane);
                assert(ret == 0);
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("bm_image_copy_host_to_device err={}", ret);
                    exit(EXIT_FAILURE);
                }
            }
        }
        errcnt_ = 0;
        return 0;
    }
    void Decoder::Decoder_CC::reconnect() {
        reset_decode(file_path_, compressed_, tpu_id_);
    }

    void Decoder::Decoder_CC::release() {
        if (!opened_) return;
        if (!is_jpeg_file_) {
            av_free_packet(&pkt_);
            if (video_dec_ctx_) {
                avcodec_close(video_dec_ctx_);
            }
            avformat_close_input(&fmt_ctx_);
        }

        handle_ = nullptr;
        fmt_ctx_ = nullptr;
        video_dec_ctx_ = nullptr;
        video_stream_ = nullptr;
        video_stream_idx_ = -1;
        video_frame_count_ = 0;
        got_frame_ = 0;
        height_ = width_ = 0;
        is_rtsp_ = false;
        is_jpeg_file_ = false;
        opened_ = false;
        end_of_file_ = false;
    }

    void Decoder::Decoder_CC::reset_decode(const std::string &file_path,
                               bool compressed,
                               int tpu_id) {
        release();
        if (is_jpg_file(file_path_)) {
            is_jpeg_file_ = true;
            return;
        }

        std::cout << "decoder ctor: filepath=" << file_path << std::endl;
        // register all formats and codecs
        av_register_all();
        avdevice_register_all();
        AVDictionary *opts = NULL;
#ifndef IS_SOC_MODE
        av_dict_set_int(&opts, "pcie_board_id", tpu_id_, 0);
#endif
        if (file_path_.compare(0, 5, "rtsp:") == 0) {
            is_rtsp_ = true;
            avformat_network_init();
            // Init the decoders, with reference counting
            av_dict_set(&opts, "refcounted_frames", "1", 0);
            // frame buffer set,same as opencv, ost is 20
            av_dict_set(&opts, "extra_frame_buffer_num", "5", 0);
            // set tcp
            av_dict_set(&opts, "rtsp_transport", "tcp", 0);
            // set timeout (same as opencv),ost is 10000000
            av_dict_set(&opts, "stimeout", "20000000", 0);

            // add same as opencv
            av_dict_set(&opts, "rtsp_flags", "prefer_tcp", 0);
            av_dict_set_int(&opts, "buffer_size", 1024000, 0);
            av_dict_set_int(&opts, "max_delay", 500000, 0);
        }
        if (!is_jpeg_file_ && compressed_) {
            // set compressed output
            av_dict_set(&opts, "output_format", "101", 0);
        }
        //av_log_set_level(AV_LOG_TRACE);
        std::cout << "open filepath=" << file_path << std::endl;
        AVInputFormat *input_fmt = nullptr;
        if (string_start_with(file_path, "/dev/video")) {
            input_fmt = av_find_input_format("video4linux2");
            if (input_fmt == nullptr) {
                printf("ERROR:can't find format: video4linux2\n");
            } else {
                printf("find video4linux2 success!\n");
                const char *pixfmt = getenv("SAIL_CAP_PIXFMT");
                if (pixfmt == nullptr) pixfmt = "mjpeg";
                av_dict_set(&opts, "pixel_format", pixfmt, 0);
            }
        }

        // open input file, and allocate format context
        int ret = avformat_open_input(&fmt_ctx_, file_path_.c_str(),
                                      input_fmt, &opts);
        if (ret < 0) {
            spdlog::error("Failed to open input file: {}", file_path_);
            //exit(SAIL_ERR_DECODER_INIT);
            throw std::runtime_error("Failed to open input file");
        }
        // retrieve stream information
        ret = avformat_find_stream_info(fmt_ctx_, nullptr);
        if (ret < 0) {
            spdlog::error("Failed to find stream information.");
            //exit(SAIL_ERR_DECODER_INIT);
            throw std::runtime_error("Failed to find stream information");
        }

        ret = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (ret < 0) {
            spdlog::error("Failed to find a video stream.");
            //exit(SAIL_ERR_DECODER_INIT);
            throw std::runtime_error("Failed to find a video stream");
        }
        video_stream_idx_ = ret;
        video_stream_ = fmt_ctx_->streams[video_stream_idx_];
        video_dec_ctx_ = video_stream_->codec;
        AVCodec *dec = avcodec_find_decoder(video_dec_ctx_->codec_id);
        if (!dec) {
            spdlog::error("Failed to find codec.");
            //exit(SAIL_ERR_DECODER_INIT);
            throw std::runtime_error("Failed to find codec");
        }
        ret = avcodec_open2(video_dec_ctx_, dec, &opts);
        if (ret < 0) {
            spdlog::error("Failed to open codec");
            //exit(SAIL_ERR_DECODER_INIT);
            throw std::runtime_error("Failed to open codec");
        }
        height_ = video_dec_ctx_->height;
        width_ = video_dec_ctx_->width;

        // initialize packet, set data to NULL, let the demuxer fill it
        av_init_packet(&pkt_);
        pkt_.data = nullptr;
        pkt_.size = 0;
        opened_ = true;
        // destroy avdict
        av_dict_free(&opts);

    }

    void Decoder::Decoder_CC::convert_to_yuv420p() {
        AVFrame *src = frame_.get();

        struct SwsContext *convert_ctx = NULL;
        enum AVPixelFormat src_pix_fmt = (enum AVPixelFormat) src->format;
        if (src_pix_fmt == AV_PIX_FMT_YUVJ420P) {
            src_pix_fmt = AV_PIX_FMT_YUV420P;
        }

        AVFrame *dst = av_frame_alloc();
        enum AVPixelFormat dst_pix_fmt = AV_PIX_FMT_YUV420P;

        dst->width = src->width;
        dst->height = src->height;
        dst->format = dst_pix_fmt;

        int ret = av_frame_get_buffer(dst, 16);
        assert(ret == 0);

        convert_ctx = sws_getContext(src->width, src->height, src_pix_fmt, dst->width, dst->height, dst_pix_fmt,
                                     SWS_FAST_BILINEAR, NULL, NULL, NULL);
        assert(convert_ctx != nullptr);

        ret = sws_scale(convert_ctx, src->data, src->linesize, 0, src->height, dst->data, dst->linesize);
        assert(ret >= 0);

        sws_freeContext(convert_ctx);

        frame_.set_frame(dst);
    }

    float Decoder::Decoder_CC::get_fps() const {
        if (video_stream_) {
            return video_stream_->avg_frame_rate.num / (float) video_stream_->avg_frame_rate.den;
        } else
            return -1;
    }

// ref: https://ffmpeg.org/doxygen/trunk/demuxing_8c-source.html
    Decoder::Decoder(
            const std::string &file_path,
            bool compressed,
            int tpu_id)
            : _impl(new Decoder_CC(file_path,compressed,tpu_id)){
    }

    bool Decoder::is_opened() {
        return _impl->opened_;
    }

    std::vector<int> Decoder::get_frame_shape() {
        std::vector<int> shape(4);
        shape[0] = 1;
        shape[1] = 3;
        shape[2] = _impl->height_;
        shape[3] = _impl->width_;
        return std::move(shape);
    }

    Decoder::~Decoder() {
        // SPDLOG_INFO("Start ~Decoder()!");
        delete _impl;
        // SPDLOG_INFO("Start ~Decoder()!");
    }

    int Decoder::decode_jpeg(Handle &handle, BMImage &image) {
        return _impl->decode_jpeg(handle, image.data());
    }

    int Decoder::decode_jpeg(Handle &handle, bm_image &image) {
        return _impl->decode_jpeg(handle, image);
    }

    void Decoder::release()
    {
        return _impl->release();
    }

    int Decoder::reconnect()
    {
        if(is_opened()){
            _impl->reconnect();
            return 0;
        }else{
            SPDLOG_INFO("Decoder not opened!");
            return 1;
        }
    }

    int Decoder::read(Handle &handle, BMImage &image) {
        bm_image img;
        int ret = read_(handle, img);
        if( ret != 0 ) {
            SPDLOG_INFO("Decoder read end or err={}", ret);
            return ret;
        }
        BMImage temp_img;
        temp_img = std::move(img);
        if(!image.empty_check()){
            image = std::move(temp_img);
        }else if(image.width() == temp_img.width() && image.height() == temp_img.height() && image.format() == temp_img.format()){
            image = std::move(temp_img);
        }else{
            Bmcv bmcv_temp(handle);
            ret = bmcv_temp.vpp_resize(temp_img, image, image.width(), image.height());
        }
        return ret;
    }

    BMImage Decoder::read(Handle &handle) {
        BMImage image;
        read(handle, image);
        return std::move(image);
    }

    int Decoder::read_(Handle &handle, bm_image &image) {
        return _impl->read_(handle, image);
    }

    bm_image Decoder::read_(Handle &handle) {
        bm_image image;
        read_(handle, image);
        return image;
    }

    float Decoder::get_fps() const {
        return _impl->get_fps();
    }

#endif //USE_FFMPEG

#ifdef USE_BMCV

    class BMImage::BMImage_CC{
    public:
        BMImage_CC();
        BMImage_CC(bm_image &img);
          
        BMImage_CC(
            Handle& handle,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext dtype);

        BMImage_CC(
            Handle& handle,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext dtype,
            int *stride);

        ~BMImage_CC(){};

    protected:
    /// inner bm_image
        void reset(int w, int h);
        bm_image img_;

    private:
        void create(
            Handle&                  handle,
            int                      h,
            int                      w,
            bm_image_format_ext      format,
            bm_image_data_format_ext dtype,
            int                      *stride=NULL);
        void destroy();
        void allocate();
        bool is_created() const;

        bool need_to_free_;
        friend class BMImage;
    };

    BMImage::BMImage_CC::BMImage_CC():img_({}), need_to_free_(false) {
        img_.image_format = FORMAT_BGR_PLANAR;
        img_.data_type = DATA_TYPE_EXT_1N_BYTE;
        img_.width = 0;
        img_.height = 0;
    }

    BMImage::BMImage_CC::BMImage_CC(bm_image &img) : img_(img), need_to_free_(false) {}

    BMImage::BMImage_CC::BMImage_CC(
            Handle &handle,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext dtype
    ) : img_({}), need_to_free_(false) {
        create(handle, h, w, format, dtype);
        allocate();
    }

    BMImage::BMImage_CC::BMImage_CC(
            Handle &handle,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext dtype,
            int *stride
    ) : img_({}), need_to_free_(false) {
        create(handle, h, w, format, dtype, stride);
        allocate();
    }

    void BMImage::BMImage_CC::create(
            Handle &handle,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext data_type,
            int *stride
    ) {
        destroy();
        bm_image_create(handle.data(), h, w, format, data_type, &img_, stride);
        need_to_free_ = true;
    }

    void BMImage::BMImage_CC::destroy() {
        if (need_to_free_) {
            bm_image_destroy(img_);
            img_.image_private = nullptr;
            need_to_free_ = false;
        }
    }

    void BMImage::BMImage_CC::allocate() {
        bm_image_alloc_dev_mem_heap_mask(img_, 6);
        need_to_free_ = true;
    }

    bool BMImage::BMImage_CC::is_created() const {
        return img_.image_private != nullptr;
    }

    void BMImage::BMImage_CC::reset(int w, int h)
    {
        if (img_.width != w || img_.height != h)
        {
            bm_handle_t bmHandle=nullptr;
            if (need_to_free_) {
                bmHandle = bm_image_get_handle(&img_);
                bm_image_destroy(img_);
                img_.image_private = nullptr;
                need_to_free_ = false;
            }

            if (bmHandle != nullptr) {
                bm_image_create(bmHandle, h, w, img_.image_format, img_.data_type, &img_);
            }
        }

    }


    BMImage::BMImage() : _impl(new BMImage_CC()){}

    BMImage::BMImage(bm_image &img) : _impl(new BMImage_CC(img)){}

    BMImage::BMImage(
            Handle &handle,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext dtype
    ) : _impl(new BMImage_CC(handle, h, w, format, dtype)) {}

    BMImage::BMImage(
            Handle &handle,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext dtype,
            int *stride
    ) : _impl(new BMImage_CC(handle, h, w, format, dtype, stride)) {}

    BMImage::BMImage(BMImage &&other) : _impl(new BMImage_CC()){
        *this = std::move(other);
    }

    BMImage &BMImage::operator=(BMImage &&other) {
        if (this != &other) {
            destroy();
            _impl->img_.width = other._impl->img_.width;
            _impl->img_.height = other._impl->img_.height;
            _impl->img_.image_format = other._impl->img_.image_format;
            _impl->img_.data_type = other._impl->img_.data_type;
            _impl->img_.image_private = other._impl->img_.image_private;
            _impl->need_to_free_ = other._impl->need_to_free_;
            other._impl->img_.image_private = nullptr;
            other._impl->need_to_free_ = false;
        }
        return *this;
    }

    BMImage &BMImage::operator=(bm_image &&other) {
        destroy();
        _impl->img_.width = other.width;
        _impl->img_.height = other.height;
        _impl->img_.image_format = other.image_format;
        _impl->img_.data_type = other.data_type;
        _impl->img_.image_private = other.image_private;
        _impl->need_to_free_ = true;
        other = {};
        return *this;
    }

    BMImage::~BMImage() {
        destroy();
        delete _impl;
    }

    bm_image &BMImage::data() {
        return _impl->img_;
    }

    bm_image BMImage::data() const {
        return _impl->img_;
    }

    int BMImage::width() const { return _impl->img_.width; }

    int BMImage::height() const { return _impl->img_.height; }

    bm_image_format_ext BMImage::format() const { return _impl->img_.image_format; }

    bm_image_data_format_ext BMImage::dtype() const { return _impl->img_.data_type; }

    bool BMImage::need_to_free() const {
        return _impl->need_to_free_; 
    }

    int BMImage::empty_check() const {
        if (!_impl->img_.image_private)
            return 0;
        return 1;
    }

    int BMImage::get_plane_num() const {
        return bm_image_get_plane_num(_impl->img_);
    }

    void BMImage::create(
            Handle &handle,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext data_type,
            int *stride
    ) {
        return _impl->create(handle, h, w, format, data_type, stride);
    }

    void BMImage::destroy() {
        return _impl->destroy();
    }

    void BMImage::allocate() {
        return _impl->allocate();
    }

    bool BMImage::is_created() const {
        return _impl->is_created();
    }

    void BMImage::reset(int w, int h)
    {
        return _impl->reset(w, h);
    }

    template<std::size_t N>
    BMImageArray<N>::BMImageArray() : need_to_free_(false) {
        for(int i= 0;i < N; ++i) {
            this->at(i).image_format = FORMAT_BGR_PLANAR;
            this->at(i).data_type = DATA_TYPE_EXT_1N_BYTE;
            this->at(i).width = this->at(i).height = 0;
            this->at(i).image_private = nullptr;
        }
    }

    template<std::size_t N>
    BMImageArray<N>::BMImageArray(
        Handle                   &handle,
        int                      h,
        int                      w,
        bm_image_format_ext      format,
        bm_image_data_format_ext dtype
    ) : need_to_free_(false) {
        create(handle, h, w, format, dtype);
    //allocate();
    }

    template<std::size_t N>
    BMImageArray<N>::BMImageArray(
        Handle                   &handle,
        int                      h,
        int                      w,
        bm_image_format_ext      format,
        bm_image_data_format_ext dtype,
        int                      *stride
    ) : need_to_free_(false) {
        create(handle, h, w, format, dtype, stride);
    //allocate();
    }

    // template<std::size_t N>
    // int BMImageArray<N>::attach_data(const std::vector<BMImage> &data){
    //     int ret = 0;
    //     return ret;
    // }

    // template<std::size_t N>
    // BMImageArray<N>::BMImageArray(const std::vector<BMImage> &data)
    // : need_to_free_(false) {
    //     if(data.size() != N){
    //         SPDLOG_ERROR("Error Input size {} vs. {}!", N, data.size());
    //         exit(SAIL_ERR_BMCV_INIT);
    //     }
    // }

    // template<std::size_t N>
    // BMImageArray<N>::BMImageArray(const BMImage &data){

    // }

    template<std::size_t N>
    int BMImageArray<N>::copy_from(int i, BMImage &data){
        int ret = 0;
        if(need_to_free_){
            if (this->at(0).width        != data.width()  || 
                this->at(0).height       != data.height() || 
                this->at(0).image_format != data.format() ||
                this->at(0).data_type    != data.dtype()) {
                SPDLOG_ERROR("requires src image's format is same as dst");
                print_image(this->at(0));
                print_image(data.data());
            }
        }
        bm_handle_t handle = bm_image_get_handle(&data.data());
        if(!need_to_free_){
            int stride[3]={0};
            bm_image_get_stride(data.data(), stride);
            for(int i=0; i<N; i++) {
                ret = bm_image_create(handle,
                    data.height(),
                    data.width(),
                    data.format(), 
                    data.dtype(),
                    &(*this)[i],stride);
                if (ret != BM_SUCCESS){
                    SPDLOG_ERROR("bm_image_create err={}", ret);
                    return ret;
                }
                ret = bm_image_alloc_dev_mem_heap_mask((*this)[i], 6);
                if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("bm_image_alloc_dev_mem_heap_mask err={}", ret);
                    return ret;
                }
            }
            need_to_free_ = true;
        }
        bmcv_copy_to_atrr_t attr;
        memset(&attr, 0, sizeof(attr));
        ret = bmcv_image_copy_to(handle, attr, data.data(), (*this)[i]);
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("bmcv_image_copy_to err={}", ret);
        }
        return ret;
    }

    template<std::size_t N>
    int BMImageArray<N>::attach_from(int i, BMImage &data){
        int ret = 0;
        if(need_to_free_){
            destroy();
        }
        if(is_created()){
            if (this->at(0).width        != data.width()  || 
                this->at(0).height       != data.height() || 
                this->at(0).image_format != data.format() ||
                this->at(0).data_type    != data.dtype()) {
                SPDLOG_ERROR("requires src image's format is same as dst");
                print_image(this->at(0));
                print_image(data.data());
            }
        }else{
            bm_handle_t handle = bm_image_get_handle(&data.data());
            int stride[3]={0};
            bm_image_get_stride(data.data(), stride);
            for(int i=0; i<N; i++) {
                ret = bm_image_create(handle,
                    data.height(),
                    data.width(),
                    data.format(), 
                    data.dtype(),
                    &(*this)[i],stride);
                if (ret != BM_SUCCESS){
                    SPDLOG_ERROR("bm_image_create err={}", ret);
                    return ret;
                }
            }
        }
        bm_device_mem_t dev_mem[3];
        ret = bm_image_get_device_mem(data.data(), dev_mem);
        if (ret != BM_SUCCESS){
            SPDLOG_ERROR("bm_image_get_device_mem err={}", ret);
            return ret;
        }
        ret = bm_image_attach((*this)[i], dev_mem);
        if (ret != BM_SUCCESS){
            SPDLOG_ERROR("bm_image_attach err={}", ret);
        }
        // SPDLOG_INFO("bm_image_attach idx {}", i);
        return ret;
    }

    template<std::size_t N>
    BMImageArray<N>::~BMImageArray() {
        destroy();
    }

    template<std::size_t N>
    BMImageArray<N>::BMImageArray(BMImageArray<N> &&other) : need_to_free_(false) {
        *this = std::move(other);
    }

    template<std::size_t N>
    BMImageArray<N>& BMImageArray<N>::operator=(BMImageArray<N> &&other)
    {
        if (this != &other) {
            destroy();
            for (size_t i = 0; i < N; i ++) {
            this->at(i).width         = other.at(i).width;
            this->at(i).height        = other.at(i).height;
            this->at(i).image_format  = other.at(i).image_format;
            this->at(i).data_type     = other.at(i).data_type;
            this->at(i).image_private = other.at(i).image_private;
            other.at(i).image_private = nullptr;
            }
            this->need_to_free_ = other.need_to_free_;
            other.need_to_free_ = false;
        }
        return *this;
    }

    template<std::size_t N>
    bool BMImageArray<N>::is_created() const {
        return !this->empty() && (this->at(0).image_private != nullptr);
    }

    template<std::size_t N>
    bm_image_format_ext BMImageArray<N>::format(int index) const {
        return this->at(index).image_format;
    }

    template<std::size_t N>
    void BMImageArray<N>::create(
    Handle                   &handle,
    int                      h,
    int                      w,
    bm_image_format_ext      format,
    bm_image_data_format_ext dtype,
    int                      *stride
    ) {
        // clear old before.
        destroy();
        // create new instance
        for (size_t i = 0; i < N; i++) {
            bm_image_create(handle.data(), h, w, format, dtype, &this->at(i), stride);
        }
        // int ret = bm_image_alloc_contiguous_mem_heap_mask(N, this->data(),6);
        int ret = bm_image_alloc_contiguous_mem(N, this->data());
        if(ret != BM_SUCCESS) {
            char error_info[512]={0};
            sprintf(error_info,"bm_image_alloc_contiguous_mem error:%d,N[%d],h[%d],w[%d],format[%d],dtype[%d]", 
                ret, N, h, w, format, dtype);
            SPDLOG_ERROR(error_info);
            exit(1);
        }
        need_to_free_ = true;
    }

    template<std::size_t N>
    void BMImageArray<N>::create(
            Handle                   &handle,
            int                      h,
            int                      w,
            bm_image_format_ext      format,
            bm_image_data_format_ext dtype
    ) {
        create(handle, h, w, format, dtype, nullptr);
    }

    template<std::size_t N>
    void BMImageArray<N>::reset(
            int                      h,
            int                      w) {

        if (this->at(0).width != w || this->at(0).height != h)
        {
            SPDLOG_INFO("reset image, src({},{}) dst({},{})",
                    this->at(0).width,
                    this->at(0).height,
                    w, h);

            bm_handle_t bmHandle=nullptr;
            if (need_to_free_) {
                bmHandle = bm_image_get_handle(&this->at(0));
                bm_image_free_contiguous_mem(N, this->data());
                need_to_free_ = false;
            }

            for (size_t i = 0; i < N; i ++) {
                bm_image_destroy(this->at(i));
                this->at(i).image_private=nullptr;
            }

            if (bmHandle != nullptr) {
                for (size_t i = 0; i < N; i ++) {
                    bm_image_create(bmHandle, h, w, this->at(0).image_format, this->at(0).data_type, &this->at(i));
                }
            }

            int ret = bm_image_alloc_contiguous_mem(N, this->data());
            // int ret = bm_image_alloc_contiguous_mem_heap_mask(N, this->data(),6);
            if(ret != BM_SUCCESS) {
                SPDLOG_ERROR("bm_image_alloc_contiguous_mem err={}",ret);
                exit(1);
            }
            need_to_free_ = true;
        }
    }

    template<std::size_t N>
    void BMImageArray<N>::destroy() {
        if (need_to_free_){
            bm_image_free_contiguous_mem(N, this->data());
            need_to_free_ = false;
        }

        for (size_t i = 0; i < N; i++) {
                bm_image_destroy(this->at(i));
                this->at(i).image_private = nullptr;
        }
    }

    template<std::size_t N>
    void BMImageArray<N>::to_tensor(Tensor &tensor){
        if(N <= 0){
            SPDLOG_ERROR("The size of the array must be greater than zero.");
            return;
        }
        if(this->at(0).image_format != FORMAT_RGB_PLANAR &&
            this->at(0).image_format != FORMAT_BGR_PLANAR) {
            SPDLOG_ERROR("Only support image format FORMAT_BGR_PLANAR or FORMAT_RGB_PLANAR {}. Please convert it first.",
                this->at(0).image_format) ;
            return;
        }
        int ret = 0;
        bm_data_type_t dtype = get_bm_data_type_sail(this->at(0).data_type);
        bm_device_mem_t addr;
        if (!need_to_free_) {
            SPDLOG_ERROR("input BMImage doesn't have continuous memory!");
            return;
        }
        ret = bm_image_get_contiguous_device_mem(N, this->data(), &addr);
        if (ret != BM_SUCCESS) {
            SPDLOG_ERROR("bm_image_to_tensor err={}", ret);
            exit(EXIT_FAILURE);
        }

        int h = this->at(0).height;
        int w = this->at(0).width;

        tensor.reset({N, 3, h, w}, dtype);
        tensor.reset_dev_data(addr);
    }


    Bmcv::Bmcv(Handle &handle) : handle_(handle) {}

    Bmcv::~Bmcv() {}

#if defined(USE_BMCV) && defined(USE_OPENCV)

    int Bmcv::mat_to_bm_image(cv::Mat &mat, BMImage &img) {
        if (mat.cols == 0 || mat.rows == 0) {
            SPDLOG_ERROR("mat_to_bm_image err = input mat must not empty!");
            exit(SAIL_ERR_BMCV_TRANS);
        }
        return cv::bmcv::toBMI(mat, &img.data());
    }

    BMImage Bmcv::mat_to_bm_image(cv::Mat &mat) {
        BMImage img;
        mat_to_bm_image(mat, img);
        return std::move(img);
    }

    int Bmcv::bm_image_to_mat(BMImage &img, cv::Mat &mat) {
        return cv::bmcv::toMAT(&img.data(), mat);
    }

    cv::Mat Bmcv::bm_image_to_mat(BMImage &img) {
        cv::Mat mat;
        bm_image_to_mat(img, mat);
        return std::move(mat);
    }

#endif // ! USE_BMCV

    void Bmcv::bm_image_to_tensor(BMImage &img, Tensor &tensor) {
        if (img.data().image_format != FORMAT_RGB_PLANAR && img.data().image_format != FORMAT_BGR_PLANAR) {
            spdlog::error(
                    "Only support image format FORMAT_BGR_PLANAR {} or FORMAT_RGB_PLANAR. Please convert it first.",
                    img.data().image_format);
            exit(SAIL_ERR_BMCV_TRANS);
        }

        bm_data_type_t dtype = get_bm_data_type(img.dtype());
        bm_device_mem_t addr;
        bm_image_get_device_mem(img.data(), &addr);

        int h = img.height();
        int w = img.width();

        tensor.reset({1, 3, h, w}, dtype);
        tensor.reset_dev_data(addr);
#if 0
#ifdef IS_SOC_MODE
        uint8_t * sys_data= new uint8_t[addr.size];
        double process_start_time_d2s = get_current_time_us();
        bm_memcpy_d2s_partial(get_handle().data(), sys_data,addr, addr.size);
        PRINT_TIME_MS("bm_memcpy_d2s_partial", process_start_time_d2s)
        delete [] sys_data;
#endif
#endif
    }

    Tensor Bmcv::bm_image_to_tensor(BMImage &img) {
        Tensor tensor(get_handle());
        bm_image_to_tensor(img, tensor);
        return std::move(tensor);
    }

    void Bmcv::tensor_to_bm_image(Tensor &tensor, BMImage &img, bool bgr2rgb/*false*/) {
        auto shape = tensor.shape();
        int h = shape[2];
        int w = shape[3];

        bm_image_data_format_ext dtype = get_bm_image_data_format(tensor.dtype());

        if (img.is_created()) {
            img.destroy();
        }

        if (!img.is_created()) {
            int dtype_size = bm_image_data_type_size(dtype);
            int stride = FFALIGN(w * dtype_size, SAIL_ALIGN); // ceiling to 64 * N
            img.create(
                    handle_,
                    h,
                    w,
                    bgr2rgb ? FORMAT_RGB_PLANAR : FORMAT_BGR_PLANAR,
                    dtype,
                    &stride);
        }

        bm_device_mem_t mem = tensor.dev_data();
        bm_image_attach(img.data(), &mem);
    }

    BMImage Bmcv::tensor_to_bm_image(Tensor &tensor, bool bgr2rgb/*false*/) {
        BMImage img;
        tensor_to_bm_image(tensor, img, bgr2rgb);
        return std::move(img);
    }

    int Bmcv::crop_and_resize(
            bm_image *input,
            bm_image *output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h,
            int input_num
    ) {
        if(input_num <= 0){
            SPDLOG_ERROR("crop_and_resize error, invalid input_num[{}]!",input_num);
            return BM_ERR_DATA;
        }
        bmcv_resize_t attr1;
        attr1.start_x = crop_x0;
        attr1.start_y = crop_y0;
        attr1.in_width = crop_w;
        attr1.in_height = crop_h;
        attr1.out_width = resize_w;
        attr1.out_height = resize_h;

        bmcv_resize_image attr0;
        attr0.resize_img_attr = &attr1;
        attr0.roi_num = 1;
        attr0.stretch_fit = 1;
        attr0.interpolation = BMCV_INTER_NEAREST;

        int ret = bmcv_image_resize(
                handle_.data(),
                input_num,
                &attr0,
                input,
                output);
        return ret;
    }

    int Bmcv::crop_and_resize(
            BMImage &input,
            BMImage &output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h
    ) {
        
        if (output.is_created()) {
            if(output.width() != resize_w || output.height() != resize_h){
                SPDLOG_INFO("output will be reset to {}x{}",resize_w, resize_h);
            }
            output.reset(resize_w, resize_h);
        }

        if (!output.is_created()) {
            /* vpp limitation: 64-aligned */
            int dtype_size = bm_image_data_type_size(input.dtype());
            
            bm_image_format_ext temp_format = input.format();
            if(temp_format != FORMAT_BGR_PLANAR && temp_format != FORMAT_RGB_PLANAR){
                temp_format = FORMAT_BGR_PLANAR;
            }

            int stride = FFALIGN(resize_w * dtype_size, SAIL_ALIGN); // ceiling to 64 * N
            output.create(
                    handle_,
                    resize_h,
                    resize_w,
                    temp_format, // same as input format or FORMAT_RGB_PLANAR
                    input.dtype(),
                    &stride);
            output.allocate();
        }

        return crop_and_resize(&input.data(), &output.data(),
            crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h);
    }

    BMImage Bmcv::crop_and_resize(
            BMImage &input,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h) {
        BMImage output;
        crop_and_resize(input, output, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h);
        return std::move(output);
    }

    int Bmcv::crop(
            BMImage &input,
            BMImage &output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h
    ) {
        return crop_and_resize(input, output, crop_x0, crop_y0, crop_w, crop_h, crop_w, crop_h);
    }

    BMImage Bmcv::crop(
            BMImage &input,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h) {
        BMImage output;
        crop(input, output, crop_x0, crop_y0, crop_w, crop_h);
        return std::move(output);
    }

    int Bmcv::resize(
            BMImage &input,
            BMImage &output,
            int resize_w,
            int resize_h
    ) {
        return crop_and_resize(input, output, 0, 0, input.data().width, input.data().height, resize_w, resize_h);
    }

    BMImage Bmcv::resize(
            BMImage &input,
            int resize_w,
            int resize_h) {
        BMImage output;
        resize(input, output, resize_w, resize_h);
        return std::move(output);
    }

    int Bmcv::vpp_crop_and_resize(
            bm_image *input,
            bm_image *output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h,
            int input_num
    ) {
        if(input_num <= 0){
            printf("vpp_crop_and_resize error, invalid input_num[%d]!\n",input_num);
            return BM_ERR_DATA;
        }

        bmcv_rect rect;
        rect.start_x = crop_x0;
        rect.start_y = crop_y0;
        rect.crop_w = crop_w;
        rect.crop_h = crop_h;
        
        int ret = BM_SUCCESS;
        for (size_t i = 0; i < input_num; i++)        {
            ret = bmcv_image_vpp_convert(
                handle_.data(),
                1,
                input[i],
                &output[i],
                &rect);
            if(ret == BM_NOT_SUPPORTED){
                break;
            }
        }
        if (ret == BM_NOT_SUPPORTED) {
            SPDLOG_INFO("vpp not support, try tpu resize");
            print_image(input[0], " src:");
            print_image(output[0], " dst:");
            // vpp not support, try tpu resize
            bmcv_resize_t roi_attr[1];
            bmcv_resize_image resize_attr[1];
            memset(resize_attr, 0, sizeof(resize_attr));
            resize_attr[0].roi_num = 1;
            resize_attr[0].stretch_fit =1 ;
            roi_attr[0].start_x = 0;
            roi_attr[0].start_y = 0;
            roi_attr[0].in_width = input[0].width;
            roi_attr[0].in_height = input[0].height;
            roi_attr[0].out_width = output[0].width;
            roi_attr[0].out_height = output[0].height;
            resize_attr[0].resize_img_attr = &roi_attr[0];
            ret = bmcv_image_resize(handle_.data(), input_num, resize_attr, input, output);
            if (BM_SUCCESS != ret) {
                SPDLOG_ERROR("bmcv_image_resize err={}", ret);
                print_image(input[0], " src:");
                print_image(output[0], " dst:");
            }
        }else {
            if (BM_SUCCESS != ret) {
                SPDLOG_ERROR("bmcv_image_vpp_convert err={}", ret);
                print_image(input[0], " src:");
                print_image(output[0], " dst:");
            }
        }


        return ret;
    }
    int Bmcv::vpp_crop_and_resize(
            BMImage &input,
            BMImage &output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h
    ) {
        if (output.is_created()) {
            if(output.width() != resize_w || output.height() != resize_h){
                SPDLOG_INFO("output will be reset to {}x{}",resize_w, resize_h);
            }
            output.reset(resize_w, resize_h);
        }

        if (!output.is_created()) {
            /* vpp limitation: 64-aligned */
            int dtype_size = bm_image_data_type_size(input.dtype());
            bm_image_format_ext temp_format = input.format();
            if(temp_format != FORMAT_BGR_PLANAR && temp_format != FORMAT_RGB_PLANAR){
                temp_format = FORMAT_BGR_PLANAR;
            }
            int stride = FFALIGN(resize_w * dtype_size, SAIL_ALIGN); // ceiling to 64 * N
            output.create(
                    handle_,
                    resize_h,
                    resize_w,
                    temp_format, // same as input format or FORMAT_RGB_PLANAR
                    input.dtype(),
                    &stride
            );
            output.allocate();
        }
        return vpp_crop_and_resize(&input.data(), &output.data(),
            crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h);
    }

    BMImage Bmcv::vpp_crop_and_resize(
            BMImage &input,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h) {
        BMImage output;
        vpp_crop_and_resize(input, output, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h);
        return std::move(output);
    }

    int Bmcv::vpp_crop_and_resize_padding(
      bm_image                     *input,
      bm_image                     *output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in,
      int                          input_num){
        if(input_num <= 0){
            printf("vpp_crop_and_resize_padding error, invalid input_num[%d]!\n",input_num);
            return BM_ERR_DATA;
        }

        bmcv_rect rect;
        rect.start_x = crop_x0;
        rect.start_y = crop_y0;
        rect.crop_w = crop_w;
        rect.crop_h = crop_h;

        int ret = 0;
        for (size_t i = 0; i < input_num; i ++) {

            // filling output
            bmcv_padding_atrr_t padding;
            padding.dst_crop_stx = padding_in.dst_crop_stx;
            padding.dst_crop_sty = padding_in.dst_crop_sty;
            padding.dst_crop_w   = padding_in.dst_crop_w;
            padding.dst_crop_h   = padding_in.dst_crop_h;
            padding.if_memset    = 0;

            int width  = output[i].width;
            int height = output[i].height;
            int stride = 0;
            bm_image_get_stride(output[i],&stride);

            if (output[i].image_format == FORMAT_BGR_PLANAR){
                char color[4] = {0};
                bm_device_mem_t mem[4];
                bm_image_get_device_mem(output[i], mem);
                mem[0].size = height*stride;

                color[0] = padding_in.padding_b;
                bm_memset_device_ext(handle_.data(),(void*)color,1,mem[0]);

                color[0] = padding_in.padding_g;
                mem[0].u.device.device_addr += height*stride;
                bm_memset_device_ext(handle_.data(),(void*)color,1,mem[0]);

                color[0] = padding_in.padding_r;
                mem[0].u.device.device_addr += height*stride;
                bm_memset_device_ext(handle_.data(),(void*)color,1,mem[0]);
            }else if (output[i].image_format == FORMAT_BGR_PACKED){
                char color[4] = {0};
                color[0] = padding_in.padding_b;
                color[1] = padding_in.padding_g;
                color[2] = padding_in.padding_r;
                bm_device_mem_t addr;
                bm_image_get_device_mem(output[i], &addr);
                bm_memset_device_ext(handle_.data(),(void*)color,3,addr);
            }else if (output[i].image_format == FORMAT_RGB_PLANAR){
                char color[4] = {0};
                bm_device_mem_t mem[4];
                bm_image_get_device_mem(output[i], mem);
                mem[0].size = height*stride;

                color[0] = padding_in.padding_r;
                bm_memset_device_ext(handle_.data(),(void*)color,1,mem[0]);

                color[0] = padding_in.padding_g;
                mem[0].u.device.device_addr += height*stride;
                bm_memset_device_ext(handle_.data(),(void*)color,1,mem[0]);

                color[0] = padding_in.padding_b;
                mem[0].u.device.device_addr += height*stride;
                bm_memset_device_ext(handle_.data(),(void*)color,1,mem[0]);
            }else if (output[i].image_format == FORMAT_RGB_PACKED){
                char color[4] = {0};
                color[0] = padding_in.padding_r;
                color[1] = padding_in.padding_g;
                color[2] = padding_in.padding_b;
                bm_device_mem_t addr;
                bm_image_get_device_mem(output[i], &addr);
                bm_memset_device_ext(handle_.data(),(void*)color,3,addr);
            }else{
                spdlog::error("Only support image format FORMAT_BGR_PLANAR or FORMAT_RGB_PLANAR or FORMAT_BGR_PACKED or FORMAT_RGB_PACKED. Please convert it first.");
                return -1;
            } 
            // if (input[i].width % 64 == 0){
            ret = bmcv_image_vpp_convert_padding(
                handle_.data(),
                1,
                input[i],
                &output[i],
                &padding,
                &rect);
            if (BM_SUCCESS != ret) {
                SPDLOG_ERROR("bmcv_image_vpp_convert_padding err={}", ret);
                print_image(input[i], " src:");
                print_image(output[i], " dst:");
                break;
            }
            // }
            // else{
            //     SPDLOG_ERROR("bmcv_image_vpp_convert_padding not support width: {}", input[i].width);
            //     print_image(input[i], " src:");
            //     print_image(output[i], " dst:");
            //     break;
            // }
        }
        return ret;
    }

    int Bmcv::vpp_crop_and_resize_padding(
            BMImage &input,
            BMImage &output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h,
            PaddingAtrr &padding_in) {

        if (output.is_created()) {
            if(output.width() != resize_w || output.height() != resize_h){
                SPDLOG_INFO("output will be reset to {}x{}",resize_w, resize_h);
            }
            output.reset(resize_w, resize_h);
        }

        if (!output.is_created()) {
            /* vpp limitation: 64-aligned */
            int dtype_size = bm_image_data_type_size(input.dtype());
            bm_image_format_ext temp_format = input.format();
            if(temp_format != FORMAT_BGR_PLANAR && temp_format != FORMAT_RGB_PLANAR){
                temp_format = FORMAT_BGR_PLANAR;
            }
            int stride = FFALIGN(resize_w * dtype_size, SAIL_ALIGN); // ceiling to 64 * N
            output.create(
                    handle_,
                    resize_h,
                    resize_w,
                    temp_format, // same as input format or FORMAT_RGB_PLANAR
                    input.dtype(),
                    &stride
            );
            output.allocate();
        }

        return vpp_crop_and_resize_padding(&input.data(), &output.data(), crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h,padding_in);
    }

    BMImage Bmcv::vpp_crop_and_resize_padding(
            BMImage &input,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h,
            PaddingAtrr &padding_in) {
        BMImage output;
        vpp_crop_and_resize_padding(input, output, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h, padding_in);
        return std::move(output);
    }

    int Bmcv::vpp_crop(
            BMImage &input,
            BMImage &output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h
    ) {
        return vpp_crop_and_resize(input, output, crop_x0, crop_y0, crop_w, crop_h, crop_w, crop_h);
    }

    BMImage Bmcv::vpp_crop(
            BMImage &input,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h) {
        BMImage output;
        vpp_crop(input, output, crop_x0, crop_y0, crop_w, crop_h);
        return std::move(output);
    }

    int Bmcv::vpp_crop_padding(
            BMImage &input,
            BMImage &output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            PaddingAtrr &padding_in
    ) {
        return vpp_crop_and_resize_padding(input, output, crop_x0, crop_y0, crop_w, crop_h, crop_w, crop_h, padding_in);
    }

    BMImage Bmcv::vpp_crop_padding(
            BMImage &input,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            PaddingAtrr &padding_in) {
        BMImage output;
        vpp_crop_padding(input, output, crop_x0, crop_y0, crop_w, crop_h, padding_in);
        return std::move(output);
    }

    int Bmcv::vpp_resize(
            BMImage &input,
            BMImage &output,
            int resize_w,
            int resize_h
    ) {
        return vpp_crop_and_resize(input, output, 0, 0, input.data().width, input.data().height, resize_w, resize_h);
    }

    BMImage Bmcv::vpp_resize(
            BMImage &input,
            int resize_w,
            int resize_h) {
        BMImage output;
        vpp_resize(input, output, resize_w, resize_h);

        /*
        if (bm_image_get_plane_num(output.data()) != 1 || bm_image_get_plane_num(input.data()) != 3)
            std::cout << "****** vpp_resize: " << bm_image_get_plane_num(input.data()) << " "
                      << bm_image_get_plane_num(output.data()) << endl;
         */
        return std::move(output);
    }

    int Bmcv::vpp_resize_padding(
            BMImage &input,
            BMImage &output,
            int resize_w,
            int resize_h,
            PaddingAtrr &padding_in
    ) {
        return vpp_crop_and_resize_padding(input, output, 0, 0, input.data().width, input.data().height, resize_w,
                                           resize_h, padding_in);
    }

    BMImage Bmcv::vpp_resize_padding(
            BMImage &input,
            int resize_w,
            int resize_h,
            PaddingAtrr &padding_in) {
        BMImage output;
        vpp_resize_padding(input, output, resize_w, resize_h, padding_in);
        return std::move(output);
    }

    int Bmcv::warp(
        bm_image *input,
        bm_image *output,
        const std::pair<
        std::tuple<float, float, float>,
        std::tuple<float, float, float>> *matrix,
        int input_num
    ){
        if(input_num <= 0){
            printf("warp error, invalid input_num[%d]!\n",input_num);
            return BM_ERR_DATA;
        }
        for (int i = 0; i < input_num; i ++) {
            if (input[i].image_format != FORMAT_RGB_PLANAR && input[i].image_format != FORMAT_BGR_PLANAR) {
                spdlog::error("Only support image format FORMAT_BGR_PLANAR or FORMAT_RGB_PLANAR. Please convert it first.");
                return -1;
            }
        }
        
        bmcv_warp_image_matrix *attr0 = new bmcv_warp_image_matrix[input_num];
        bmcv_warp_matrix *attr1 = new bmcv_warp_matrix[input_num];
        for (int i = 0; i < input_num; i ++) {
            
            attr1[i].m[0] = std::get<0>(matrix[i].first);
            attr1[i].m[1] = std::get<1>(matrix[i].first);
            attr1[i].m[2] = std::get<2>(matrix[i].first);
            attr1[i].m[3] = std::get<0>(matrix[i].second);
            attr1[i].m[4] = std::get<1>(matrix[i].second);
            attr1[i].m[5] = std::get<2>(matrix[i].second);

            attr0[i].matrix = &attr1[i];
            attr0[i].matrix_num = 1;
        }

        int return_value = bmcv_image_warp_affine(handle_.data(), input_num, attr0, input, output);
        delete []attr0;
        delete []attr1;
        return return_value;
    }

    int Bmcv::warp(
            BMImage &input,
            BMImage &output,
            const std::pair<
                    std::tuple<float, float, float>,
                    std::tuple<float, float, float>> &matrix
    ) {
        if (output.is_created()) {
            output.reset(input.width(), input.height());
        }

        if (!output.is_created()) {
            output.create(
                    handle_,
                    input.height(),
                    input.width(),
                    input.format(),
                    input.dtype()
            );
            output.allocate();
        }

        return warp(&input.data(), &output.data(), &matrix, 1);
    }

    BMImage Bmcv::warp(
            BMImage &input,
            const std::pair<
                    std::tuple<float, float, float>,
                    std::tuple<float, float, float>> &matrix) {
        BMImage output;
        warp(input, output, matrix);
        return std::move(output);
    }

    int Bmcv::convert_to(
      bm_image *input,
      bm_image *output,
      const std::tuple<
        std::pair<float, float>,
        std::pair<float, float>,
        std::pair<float, float>>   &alpha_beta,
        int input_num){
        if(input_num <= 0){
            printf("bmcv_image_convert_to error, invalid input_num[%d]!\n",input_num);
            return BM_ERR_DATA;
        }

        bmcv_convert_to_attr attr;
        attr.alpha_0 = std::get<0>(alpha_beta).first;
        attr.beta_0 = std::get<0>(alpha_beta).second;
        attr.alpha_1 = std::get<1>(alpha_beta).first;
        attr.beta_1 = std::get<1>(alpha_beta).second;
        attr.alpha_2 = std::get<2>(alpha_beta).first;
        attr.beta_2 = std::get<2>(alpha_beta).second;
        
        int ret = bmcv_image_convert_to(handle_.data(), input_num, attr, input, output);
        if (ret != BM_SUCCESS) {
            printf("bmcv_image_convert_to error, src.format=%d, dst.format=%d", input->image_format, output->image_format);
        }
        return BM_SUCCESS;
    }

    int Bmcv::convert_to(
            BMImage &input,
            BMImage &output,
            const std::tuple<
                    std::pair<float, float>,
                    std::pair<float, float>,
                    std::pair<float, float>> &alpha_beta
    ) {
        if (output.is_created()) {
            output.reset(input.width(), input.height());
        }

        if (!output.is_created()) {
            output.create(
                    handle_,
                    input.height(),
                    input.width(),
                    FORMAT_BGR_PLANAR, // force to this format
                    input.dtype()
            );
            output.allocate();
        }
        return convert_to(&input.data(), &output.data(),alpha_beta,1);
    }

    BMImage Bmcv::convert_to(
            BMImage &input,
            const std::tuple<
                    std::pair<float, float>,
                    std::pair<float, float>,
                    std::pair<float, float>> &alpha_beta
    ) {
        BMImage output;
        convert_to(input, output, alpha_beta);
        return std::move(output);
    }

    int Bmcv::yuv2bgr(
            BMImage &input,
            BMImage &output
    ) {

        if (output.is_created()) {
            output.reset(input.width(), input.height());
        }
        if (!output.is_created()) {
            int stride = FFALIGN(input.width(), SAIL_ALIGN); // ceiling to 64 * N
            output.create(
                    handle_,
                    input.height(),
                    input.width(),
                    FORMAT_BGR_PLANAR, // force to this format
                    input.dtype(),
                    &stride
            );
            output.allocate();
        }

        int ret = bmcv_image_yuv2bgr_ext(handle_.data(), 1, &input.data(), &output.data());

        return ret;
    }

    BMImage Bmcv::yuv2bgr(BMImage &input) {
        BMImage output;
        yuv2bgr(input, output);
        return std::move(output);
    }

    int Bmcv::vpp_convert_format(
            BMImage &input,
            BMImage &output
    ) {
        if (output.is_created()) {
            if(output.width() != input.width() || output.height() != input.height()) {
                SPDLOG_INFO("output will be reset to {}x{}",input.width(), input.height());
            }
            output.reset(input.width(), input.height());
        }

        if (!output.is_created()) {
            /* vpp limitation: 64-aligned */
            int dtype_size = bm_image_data_type_size(input.dtype());
            int stride = FFALIGN(input.width() * dtype_size, SAIL_ALIGN); // ceiling to 64 * N

            output.create(
                    handle_,
                    input.height(),
                    input.width(),
                    FORMAT_BGR_PLANAR, // force to this format
                    input.dtype(),
                    &stride
            );
            output.allocate();
        }

        int ret = bmcv_image_vpp_convert(
                handle_.data(),
                1,
                input.data(),
                &output.data()
        );

        return ret;
    }

    BMImage Bmcv::vpp_convert_format(BMImage &input) {
        BMImage output;
        vpp_convert_format(input, output);
        return std::move(output);
    }

    int Bmcv::convert_format(
            BMImage &input,
            BMImage &output
    ) {
        if (output.is_created()) {
            if(output.width() != input.width() || output.height() != input.height()) {
                SPDLOG_INFO("output will be reset to {}x{}",input.width(), input.height());
            }
            output.reset(input.width(), input.height());
        }

        if (!output.is_created()) {
            output.create(
                    handle_,
                    input.height(),
                    input.width(),
                    FORMAT_BGR_PLANAR, // force to this format
                    input.dtype()
            );
            output.allocate();
        }

        int ret = bmcv_image_storage_convert(
                handle_.data(),
                1,
                &input.data(),
                &output.data()
        );

        return ret;
    }

    BMImage Bmcv::convert_format(BMImage &input) {
        BMImage output;
        convert_format(input, output);
        return std::move(output);
    }

    int Bmcv::rectangle(
            const BMImage &input,
            int x0,
            int y0,
            int w,
            int h,
            const std::tuple<int, int, int> &color,
            int thickness
    ) {
        bmcv_rect rect = {x0, y0, w, h};
        int ret = bmcv_image_draw_rectangle(
                handle_.data(),
                input.data(),
                1,
                &rect,
                thickness,
                std::get<2>(color),  // R
                std::get<1>(color),  // G
                std::get<0>(color)); // B
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("bmcv_image_draw_rectangle() err={}", ret);
            return ret;
        }
        return BM_SUCCESS;
    }

    int Bmcv::rectangle_(
            const bm_image &input,
            int x0,
            int y0,
            int w,
            int h,
            const std::tuple<int, int, int> &color,
            int thickness
    ) {
        bmcv_rect rect = {x0, y0, w, h};
        int ret = bmcv_image_draw_rectangle(
                handle_.data(),
                input,
                1,
                &rect,
                thickness,
                std::get<2>(color),  // R
                std::get<1>(color),  // G
                std::get<0>(color)); // B
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("bmcv_image_draw_rectangle() err={}", ret);
            return ret;
        }
        return BM_SUCCESS;
    }

    int Bmcv::putText(
        const BMImage                   &image,
        const std::string               &text,
        int                             x,
        int                             y,
        const std::tuple<int, int, int> &color, // BGR
        float                           fontScale,
        int                             thickness
    ){
        if(image.format() != FORMAT_GRAY &&
            image.format() != FORMAT_YUV420P &&
            image.format() != FORMAT_YUV422P &&
            image.format() != FORMAT_NV12 &&
            image.format() != FORMAT_NV21 &&
            image.format() != FORMAT_NV16 &&
            image.format() != FORMAT_NV61){
            SPDLOG_ERROR("input format not supported!");
            print_image(image.data(),"input");
            return BM_ERR_FAILURE;
        }
        bmcv_point_t org = {x,y};
        bmcv_color_t color_put = {std::get<2>(color), std::get<1>(color), std::get<0>(color)};
        int ret = bmcv_image_put_text(
            handle_.data(), 
            image.data(), 
            text.c_str(),
            org,
            color_put,
            fontScale,
            thickness);
         if (BM_SUCCESS != ret) {
            print_image(image.data(),"input");
            SPDLOG_ERROR("bmcv_image_put_text() err={}", ret);
            return ret;
        }
        return BM_SUCCESS;
    }

    int Bmcv::putText_(
        const bm_image                  &image,
        const std::string               &text,
        int                             x,
        int                             y,
        const std::tuple<int, int, int> &color, // BGR
        float                           fontScale,
        int                             thickness
    ){
        if(image.image_format != FORMAT_GRAY &&
            image.image_format != FORMAT_YUV420P &&
            image.image_format != FORMAT_YUV422P &&
            image.image_format != FORMAT_NV12 &&
            image.image_format != FORMAT_NV21 &&
            image.image_format != FORMAT_NV16 &&
            image.image_format != FORMAT_NV61){
            SPDLOG_ERROR("input format not supported!");
            print_image(image,"input");
            return BM_ERR_FAILURE;
        }
        bmcv_point_t org = {x,y};
        bmcv_color_t color_put = {std::get<2>(color), std::get<1>(color), std::get<0>(color)};
        int ret = bmcv_image_put_text(
            handle_.data(), 
            image, 
            text.c_str(),
            org,
            color_put,
            fontScale,
            thickness);
         if (BM_SUCCESS != ret) {
            print_image(image,"input");
            SPDLOG_ERROR("bmcv_image_put_text() err={}", ret);
            return ret;
        }
        return BM_SUCCESS;
    }
    int Bmcv::image_add_weighted(
        BMImage           &input1,
        float             alpha,
        BMImage           &input2,
        float             beta,
        float             gamma,
        BMImage           &output){
        if (!input1.is_created()){
            SPDLOG_ERROR("input1 must be created before!");
            return BM_ERR_FAILURE;
        }
        if (!input2.is_created()){
            SPDLOG_ERROR("input2 must be created before!");
            return BM_ERR_FAILURE;
        }
        if (input1.dtype() != DATA_TYPE_EXT_1N_BYTE || input2.dtype() != DATA_TYPE_EXT_1N_BYTE){
            SPDLOG_ERROR("Input Dtype must be DATA_TYPE_EXT_1N_BYTE!");
            return BM_ERR_FAILURE;
        }
        if (input1.width() != input2.width() || input1.height() != input2.height() || input1.format() != input2.format()){
            SPDLOG_ERROR("The width, height and format of input2 must be consistent with that of input1!");
            return BM_ERR_FAILURE;
        }

        if (output.is_created()) {
            if (input1.width() != output.width() || input1.height() != output.height()){
                SPDLOG_ERROR("The width, height of output must be consistent with that of input1!");
                return BM_ERR_FAILURE;
            }
            if (output.format() != FORMAT_BGR_PLANAR || output.format() != FORMAT_RGB_PLANAR){
                SPDLOG_ERROR("The output format must be FORMAT_BGR_PLANAR!");
                return BM_ERR_FAILURE;
            }
            if (output.format() != input1.format())  {
                SPDLOG_ERROR("The output format must same as input1.format!");
                return BM_ERR_FAILURE;
            }
        }
        bm_image_format_ext temp_format = input1.format();
        bool convert_flag = false;
        if(temp_format != FORMAT_BGR_PLANAR && temp_format != FORMAT_RGB_PLANAR){
            temp_format = FORMAT_BGR_PLANAR;
            convert_flag = true;
        }
        if(!output.is_created()){ 
            int dtype_size = bm_image_data_type_size(input1.dtype());
            int stride = FFALIGN(input1.width() * dtype_size, SAIL_ALIGN); // ceiling to 64 * N

            output.create(
                    handle_,
                    input1.height(),
                    input1.width(),
                    temp_format, // force to this format
                    input1.dtype(),
                    &stride
            );
        }
        int ret = BM_SUCCESS;
        if(convert_flag){
            BMImage input1_temp = convert_format(input1);
            BMImage input2_temp = convert_format(input2);
            ret = bmcv_image_add_weighted(handle_.data(), input1_temp.data(), alpha, input2_temp.data(), beta, gamma, output.data());
        }else{
            ret = bmcv_image_add_weighted(handle_.data(), input1.data(), alpha, input2.data(), beta, gamma, output.data());
        }
        if (BM_SUCCESS != ret){
            SPDLOG_ERROR("bmcv_image_add_weighted err={}", ret);
        }
        return ret;
    }

    BMImage Bmcv::image_add_weighted(
      BMImage           &input1,
      float             alpha,
      BMImage           &input2,
      float             beta,
      float             gamma){
        BMImage output;
        int ret = image_add_weighted(input1,alpha,input2,beta, gamma,output);
        if (BM_SUCCESS != ret){
            exit(SAIL_ERR_BMCV_TRANS);
        }
        return std::move(output);
    }
    int Bmcv::imwrite(
            const std::string &filename,
            const BMImage &input
    ) {
        int ret;
 #if defined USE_OPENCV && defined USE_BMCV
        cv::Mat cv_img;
        bm_image bmImage = input.data();
        ret = cv::bmcv::toMAT((bm_image *) &bmImage, cv_img, true);
        if (ret != 0) {
            SPDLOG_ERROR("cv::bmcv::toMat() err={}", ret);
            return ret;
        }

        if (!cv::imwrite(filename, cv_img)) {
            SPDLOG_ERROR("cv::imwrite failed");
            return BM_ERR_FAILURE;
        }
#else
        ret = bm_image_write_to_bmp(input.data(), filename.c_str());
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("bm_image_write_to_bmp() err={}", ret);
            return ret;
        }
#endif
        return BM_SUCCESS;
    }

    int Bmcv::imwrite_(
            const std::string &filename,
            const bm_image &input
    ) {
#if defined USE_OPENCV && defined USE_BMCV
        cv::Mat cv_img;
        cv::bmcv::toMAT((bm_image *) &input, cv_img, true);
        cv::imwrite(filename, cv_img);
#else
        bm_image_write_to_bmp(input.data(), filename.c_str());
#endif
        return 0;
    }


    Handle Bmcv::get_handle() {
        return handle_;
    }

    bm_data_type_t
    Bmcv::get_bm_data_type(
            bm_image_data_format_ext fmt
    ) {
        return get_bm_data_type_sail(fmt);
    }

    bm_image_data_format_ext
    Bmcv::get_bm_image_data_format(
            bm_data_type_t dtype
    ) {
        return get_bm_image_data_format_sail(dtype);
    }
#endif //USE_BMCV

    bm_image Bmcv::crop_and_resize_padding(
      bm_image &input,
      int crop_x0,
      int crop_y0,
      int crop_w,
      int crop_h,
      int resize_w,
      int resize_h,
      PaddingAtrr &padding_in
    ){
        bm_image_format_ext image_format = FORMAT_BGR_PLANAR;
        if (input.image_format == FORMAT_RGB_PLANAR){
            image_format = FORMAT_RGB_PLANAR;
        }
        bm_image bm_image_result;
        int ret = bm_image_create(handle_.data(),
            resize_h, resize_w, 
            image_format, DATA_TYPE_EXT_1N_BYTE,
            &bm_image_result);

        float scale_w = (float)padding_in.dst_crop_w/crop_w;
        float scale_h = (float)padding_in.dst_crop_h/crop_h;
        int temp_image_w = padding_in.dst_crop_w;
        int temp_image_h = padding_in.dst_crop_h;
        if(scale_w < scale_h) temp_image_h = crop_h*scale_w;
        else temp_image_w = crop_w*scale_h;
        bm_image bm_image_temp;
        ret = bm_image_create(
            handle_.data(),
            temp_image_h, temp_image_w, 
            image_format, DATA_TYPE_EXT_1N_BYTE,
            &bm_image_temp);
        if (BM_SUCCESS != ret){
            SPDLOG_ERROR("bm_image_create err={}", ret);
            bm_image_destroy(bm_image_temp);
            return bm_image_result;
        }
        bmcv_resize_t attr_rt;
        attr_rt.start_x = crop_x0;
        attr_rt.start_y = crop_y0;
        attr_rt.in_width = crop_w;
        attr_rt.in_height = crop_h;
        attr_rt.out_width = temp_image_w;
        attr_rt.out_height = temp_image_h;
        
        bmcv_resize_image attr_ri;
        attr_ri.resize_img_attr = &attr_rt;
        attr_ri.roi_num = 1;
        attr_ri.stretch_fit = 1;
        attr_ri.interpolation = BMCV_INTER_NEAREST;

        ret = bmcv_image_resize(
            handle_.data(),
            1,
            &attr_ri,
            &input,
            &bm_image_temp);
        if (BM_SUCCESS != ret){
            SPDLOG_ERROR("bmcv_image_resize err={}", ret);
        }else{
            bmcv_copy_to_atrr_t copy_att;
            copy_att.start_x = padding_in.dst_crop_stx;
            copy_att.start_y = padding_in.dst_crop_sty;
            copy_att.padding_r = padding_in.padding_r;
            copy_att.padding_g = padding_in.padding_g;
            copy_att.padding_b = padding_in.padding_b;
            copy_att.if_padding = 1;
            ret = bmcv_image_copy_to(
                handle_.data(),
                copy_att,
                bm_image_temp,
                bm_image_result);
            if (BM_SUCCESS != ret){
                SPDLOG_ERROR("bmcv_image_resize err={}", ret);
            }
        }

        bm_image_destroy(bm_image_temp);
        return bm_image_result;
    }

    BMImage Bmcv::crop_and_resize_padding(
      BMImage &input,
      int crop_x0,
      int crop_y0,
      int crop_w,
      int crop_h,
      int resize_w,
      int resize_h,
      PaddingAtrr &padding_in
    ){
        bm_image bm_image_result = crop_and_resize_padding(
            input.data(),
            crop_x0, 
            crop_y0, 
            crop_w, 
            crop_h,
            resize_w, 
            resize_h, 
            padding_in);

        BMImage temp_img;
        temp_img = std::move(bm_image_result);
        return temp_img;
    }

    int Bmcv::image_copy_to(bm_image input, bm_image output, int start_x, int start_y)
    {
        if(input.width + start_x > output.width ){
            SPDLOG_ERROR("Input width add start_x must less than output width!");
            return 1;
        }
        if(input.height + start_y > output.height){
            SPDLOG_ERROR("Input height add start_y must less than output width!");
            return 1;
        }

        bmcv_copy_to_atrr_t copy_to_attr;
        copy_to_attr.start_x = start_x;
        copy_to_attr.start_y = start_y;
        copy_to_attr.if_padding = false;
        bm_status_t ret = bmcv_image_copy_to(handle_.data(),copy_to_attr,input,output);
        if(BM_SUCCESS != ret){
            SPDLOG_ERROR("bmcv_image_copy_to err {}!",ret);
            return ret;
        }
        return BM_SUCCESS;
    }

    int Bmcv::image_copy_to(BMImage &input, BMImage &output, int start_x, int start_y)
    {
        if(output.is_created()){
            if (input.format() != output.format()){
                SPDLOG_ERROR("Output Format must same as input!");
                exit(SAIL_ERR_BMCV_TRANS);
            }
        }else{
            SPDLOG_ERROR("Output has not created!");
            exit(SAIL_ERR_BMCV_TRANS);
        }
        int ret = image_copy_to(input.data(),output.data(),start_x,start_y);
        if(ret != BM_SUCCESS){
            exit(SAIL_ERR_BMCV_TRANS);
        }
        return ret;
    }

    int Bmcv::image_copy_to_padding(bm_image input, bm_image output,
        unsigned int padding_r, unsigned int padding_g, unsigned int padding_b,
        int start_x, int start_y)
    {
        if(input.width + start_x > output.width ){
            SPDLOG_ERROR("Input width add start_x must less than output width!");
            return 1;
        }
        if(input.height + start_y > output.height){
            SPDLOG_ERROR("Input height add start_y must less than output width!");
            return 1;
        }

        bmcv_copy_to_atrr_t copy_to_attr;
        copy_to_attr.start_x = start_x;
        copy_to_attr.start_y = start_y;
        copy_to_attr.padding_r = padding_r;
        copy_to_attr.padding_g = padding_g;
        copy_to_attr.padding_b = padding_b;
        copy_to_attr.if_padding = true;
        bm_status_t ret = bmcv_image_copy_to(handle_.data(),copy_to_attr,input,output);
        if(BM_SUCCESS != ret){
            SPDLOG_ERROR("bmcv_image_copy_to err {}!",ret);
            return ret;
        }
        return BM_SUCCESS;
    }

    int Bmcv::image_copy_to_padding(BMImage &input, BMImage &output,
        unsigned int padding_r, unsigned int padding_g, unsigned int padding_b,
        int start_x, int start_y)
    {
        if(output.is_created()){
            if (input.format() != output.format()){
                SPDLOG_ERROR("Output Format must same as input!");
                exit(SAIL_ERR_BMCV_TRANS);
            }
        }else{
            SPDLOG_ERROR("Output has not created!");
            exit(SAIL_ERR_BMCV_TRANS);
        }
        int ret = image_copy_to_padding(input.data(),output.data(),padding_r,padding_g,padding_b,start_x,start_y);
        if(ret != BM_SUCCESS){
            exit(SAIL_ERR_BMCV_TRANS);
        }
        return ret;      
    }

    nms_proposal_t* Bmcv::nms(face_rect_t *input_proposal, int proposal_size, float threshold)
    {
        nms_proposal_t *output_proposal = new nms_proposal_t;
        bmcv_nms(handle_.data(),
            bm_mem_from_system(input_proposal),
            proposal_size,
            threshold,
            bm_mem_from_system(output_proposal));
        return output_proposal;
    }

    BMImage Bmcv::warp_perspective(
        BMImage                     &input,
        const std::tuple<
            std::pair<int,int>,
            std::pair<int,int>,
            std::pair<int,int>,
            std::pair<int,int>>     &coordinate,
        int                         output_width,
        int                         output_height,
        bm_image_format_ext         format,
        bm_image_data_format_ext    dtype,
        int                         use_bilinear)
    {
        if (format != FORMAT_BGR_PLANAR && format != FORMAT_RGB_PLANAR){
            SPDLOG_ERROR("Output Format Error, Only support FORMAT_BGR_PLANAR and FORMAT_RGB_PLANAR!");
            exit(SAIL_ERR_BMCV_TRANS);
        }
        if (dtype != DATA_TYPE_EXT_1N_BYTE && dtype != DATA_TYPE_EXT_4N_BYTE){
            SPDLOG_ERROR("Output dtype Error, Only support DATA_TYPE_EXT_1N_BYTE and DATA_TYPE_EXT_4N_BYTE!");
            exit(SAIL_ERR_BMCV_TRANS);
        }
        BMImage output_image = sail::BMImage(handle_, output_height, output_width, format, dtype);
        bmcv_perspective_image_coordinate coord[4];  
        coord[0].coordinate_num = 1;
        bmcv_perspective_coordinate coordinate_temp;

        coordinate_temp.x[0] = std::get<0>(coordinate).first;
        coordinate_temp.y[0] = std::get<0>(coordinate).second;
        coordinate_temp.x[1] = std::get<1>(coordinate).first;
        coordinate_temp.y[1] = std::get<1>(coordinate).second;
        coordinate_temp.x[2] = std::get<2>(coordinate).first;
        coordinate_temp.y[2] = std::get<2>(coordinate).second;
        coordinate_temp.x[3] = std::get<3>(coordinate).first;
        coordinate_temp.y[3] = std::get<3>(coordinate).second;
        coord[0].coordinate = &coordinate_temp;

        int ret = BM_SUCCESS;
        if(input.format() == format && input.dtype() == dtype){
            ret = bmcv_image_warp_perspective_with_coordinate(
                handle_.data(),1,coord,&input.data(), &output_image.data(), use_bilinear);
            if(ret != BM_SUCCESS){
                SPDLOG_ERROR("bmcv_image_warp_perspective_with_coordinate err={}", ret);
                exit(SAIL_ERR_BMCV_TRANS);
            }
        }else{
            BMImage input_convert = sail::BMImage(handle_, input.height(), input.width(), format, dtype);
            if(input.width() % 16 == 0 && input.height() % 2 == 0){
                vpp_convert_format(input,input_convert); 
            }else{
                convert_format(input,input_convert); 
            }
            ret = bmcv_image_warp_perspective_with_coordinate(
                handle_.data(),1,coord,&input_convert.data(), &output_image.data(), use_bilinear);
            if(ret != BM_SUCCESS){
                SPDLOG_ERROR("bmcv_image_warp_perspective_with_coordinate err={}", ret);
                exit(SAIL_ERR_BMCV_TRANS);
            }   
        }
        return output_image;
    }
#ifdef PYTHON
    pybind11::array_t<float> Bmcv::nms(pybind11::array_t<float> input_proposal, float threshold)
    {
        pybind11::module np = pybind11::module::import("numpy");  // like 'import numpy as np'
        pybind11::buffer_info buf = input_proposal.request();
        if(buf.ndim != 2){
            SPDLOG_ERROR("Input proposal dims must be 2");
            exit(SAIL_ERR_BMCV_TRANS);
        }
        int proposal_size = buf.shape[0];
        if (proposal_size > 56000){
            SPDLOG_ERROR("Input proposal max size is 56000");
            exit(SAIL_ERR_BMCV_TRANS);
        }
        if(buf.shape[1] != 5){
            SPDLOG_ERROR("Input proposal shape error, proposal must be [left,top,right,bottom,score]!");
            exit(SAIL_ERR_BMCV_TRANS);
        }
        if(buf.itemsize !=4 || buf.format != "f"){
            SPDLOG_ERROR("Type of Input proposal must be float32!");
            exit(SAIL_ERR_BMCV_TRANS);
        }
        face_rect_t *proposal_rand = (face_rect_t *)buf.ptr;
        if (!pybind11::detail::check_flags(input_proposal.ptr(), pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_)) {
            pybind11::array_t<float> arr_c = np.attr("ascontiguousarray")(input_proposal, "dtype"_a="float32");
            proposal_rand = (face_rect_t*)arr_c.request().ptr;
        }
        nms_proposal_t *output_proposal = new nms_proposal_t;
        bmcv_nms(handle_.data(),
            bm_mem_from_system(proposal_rand),
            proposal_size,
            threshold,
            bm_mem_from_system(output_proposal));
        int output_size = output_proposal->size;
        pybind11::list shape_temp;
        shape_temp.append(output_size);
        shape_temp.append(5);
        pybind11::array_t<float> arr = np.attr("zeros")(shape_temp, "dtype"_a="float32");
        memcpy((void *)arr.request().ptr, (void *)output_proposal->face_rect, output_size*5*sizeof(float)); 
        delete output_proposal;

        return std::move(arr);
    }
#endif
}  // namespace sail
