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

#ifdef USE_OPENCV

#include "opencv2/opencv.hpp"

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

#ifdef USE_FFMPEG

// ref: https://ffmpeg.org/doxygen/trunk/demuxing_8c-source.html
    Decoder::Decoder(
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
            // frame buffer set
            av_dict_set(&opts, "extra_frame_buffer_num", "20", 0);
            // set tcp
            av_dict_set(&opts, "rtsp_transport", "tcp", 0);
            // set timeout
            av_dict_set(&opts, "stimeout", "10000000", 0);
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

    bool Decoder::is_opened() {
        return opened_;
    }

    std::vector<int> Decoder::get_frame_shape() {
        std::vector<int> shape(4);
        shape[0] = 1;
        shape[1] = 3;
        shape[2] = height_;
        shape[3] = width_;
        return shape;
    }

    Decoder::~Decoder() {
        if (!is_jpeg_file_) {
            av_free_packet(&pkt_);
            if (video_dec_ctx_) {
                avcodec_close(video_dec_ctx_);
            }
            avformat_close_input(&fmt_ctx_);
        }
    }

    bool Decoder::grab(Frame &frame) {
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

    int Decoder::decode_jpeg(Handle &handle, BMImage &image) {
        return decode_jpeg(handle, image.data());
    }

    int Decoder::decode_jpeg(Handle &handle, bm_image &image) {
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
          cv::Mat m1;
          m1.allocator = cv::hal::getAllocator();
          cv::imdecode(pic, cv::IMREAD_COLOR, &m1, handle.get_device_id());
          memset(&image, 0, sizeof(image));
          ret = cv::bmcv::toBMI(m1, &image);
          if (ret != BM_SUCCESS) {
            spdlog::error("cv::bmcv::toBMI() err {},{}", __FILE__, __LINE__);
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

    void Decoder::nv12_frame_to_image(Handle &handle, bm_image &image) {
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

    int Decoder::read(Handle &handle, BMImage &image) {
        bm_image img;
        int ret = read_(handle, img);
        image = std::move(img);
        return ret;
    }

    BMImage Decoder::read(Handle &handle) {
        BMImage image;
        read(handle, image);
        return std::move(image);
    }

    void Decoder::reset_decode(const std::string &file_path,
                               bool compressed,
                               int tpu_id) {

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
            // frame buffer set
            av_dict_set(&opts, "extra_frame_buffer_num", "20", 0);
            // set tcp
            av_dict_set(&opts, "rtsp_transport", "tcp", 0);
            // set timeout
            av_dict_set(&opts, "stimeout", "10000000", 0);
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


    int Decoder::read_(Handle &handle, bm_image &image) {
        handle_ = handle.data();
        if (is_jpeg_file_) {
            return decode_jpeg(handle, image);
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
        } else {
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

    bm_image Decoder::read_(Handle &handle) {
        bm_image image;
        read_(handle, image);
        return image;
    }

    void Decoder::convert_to_yuv420p() {
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

    float Decoder::get_fps() const {
        if (video_stream_) {
            return video_stream_->avg_frame_rate.num / (float) video_stream_->avg_frame_rate.den;
        } else
            return -1;
    }

#endif //USE_FFMPEG

#ifdef USE_BMCV

    BMImage::BMImage() : img_({}), need_to_free_(false) {
        img_.image_format = FORMAT_BGR_PLANAR;
        img_.data_type = DATA_TYPE_EXT_1N_BYTE;
        img_.width = 0;
        img_.height = 0;
    }

    BMImage::BMImage(bm_image &img) : img_(img), need_to_free_(false) {}

    BMImage::BMImage(
            Handle &handle,
            int h,
            int w,
            bm_image_format_ext format,
            bm_image_data_format_ext dtype
    ) : img_({}), need_to_free_(false) {
        create(handle, h, w, format, dtype);
        allocate();
    }

    BMImage::BMImage(
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

    BMImage::BMImage(BMImage &&other) : img_({}), need_to_free_(false) {
        *this = std::move(other);
    }

    BMImage &BMImage::operator=(BMImage &&other) {
        if (this != &other) {
            destroy();
            img_.width = other.img_.width;
            img_.height = other.img_.height;
            img_.image_format = other.img_.image_format;
            img_.data_type = other.img_.data_type;
            img_.image_private = other.img_.image_private;
            need_to_free_ = other.need_to_free_;
            other.img_.image_private = nullptr;
            other.need_to_free_ = false;
        }
        return *this;
    }

    BMImage &BMImage::operator=(bm_image &&other) {
        destroy();
        img_.width = other.width;
        img_.height = other.height;
        img_.image_format = other.image_format;
        img_.data_type = other.data_type;
        img_.image_private = other.image_private;
        need_to_free_ = true;
        other = {};
        return *this;
    }

    BMImage::~BMImage() {
        destroy();
    }

    bm_image &BMImage::data() {
        return img_;
    }

    bm_image BMImage::data() const {
        return img_;
    }

    int BMImage::width() const { return img_.width; }

    int BMImage::height() const { return img_.height; }

    bm_image_format_ext BMImage::format() const { return img_.image_format; }

    bm_image_data_format_ext BMImage::dtype() const { return img_.data_type; }

    int BMImage::empty_check() const {
        if (!img_.image_private)
            return 0;
        return 1;
    }

    int BMImage::get_plane_num() const {
        return bm_image_get_plane_num(img_);
    }

    void BMImage::create(
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

    void BMImage::destroy() {
        if (need_to_free_) {
            bm_image_destroy(img_);
            img_.image_private = nullptr;
            need_to_free_ = false;
        }
    }

    void BMImage::allocate() {
        bm_image_alloc_dev_mem_heap_mask(img_, 6);
        need_to_free_ = true;
    }

    bool BMImage::is_created() const {
        return img_.image_private != nullptr;
    }

    void BMImage::reset(int w, int h)
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

    Bmcv::Bmcv(Handle &handle) : handle_(handle) {}

    Bmcv::~Bmcv() {}



#if defined(USE_BMCV) && defined(USE_OPENCV)

    int Bmcv::mat_to_bm_image(cv::Mat &mat, BMImage &img) {
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
        bm_memcpy_d2s_partial(get_handle().data(), sys_data,addr, addr.size);
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
            BMImage &input,
            BMImage &output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h
    ) {
        int input_num = 1;
        int output_num = 1; // should equal to size of crop windows

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

        if (output.is_created()) {
            output.reset(resize_w, resize_h);
        }

        if (!output.is_created()) {
            /* vpp limitation: 64-aligned */
            int dtype_size = bm_image_data_type_size(input.dtype());

            int stride = FFALIGN(resize_w * dtype_size, SAIL_ALIGN); // ceiling to 64 * N
            output.create(
                    handle_,
                    resize_h,
                    resize_w,
                    FORMAT_BGR_PLANAR, // force to this format
                    input.dtype(),
                    &stride);
            output.allocate();
        }

        int ret = bmcv_image_resize(
                handle_.data(),
                input_num,
                &attr0,
                &input.data(),
                &output.data()
        );

        return ret;
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
            BMImage &input,
            BMImage &output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h
    ) {
        int input_num = 1;
        int output_num = 1; // should equal to size of crop windows

        bmcv_rect rect;
        rect.start_x = crop_x0;
        rect.start_y = crop_y0;
        rect.crop_w = crop_w;
        rect.crop_h = crop_h;

        if (output.is_created()) {
            output.reset(resize_w, resize_h);
        }

        if (!output.is_created()) {
            /* vpp limitation: 64-aligned */
            int dtype_size = bm_image_data_type_size(input.dtype());

            int stride = FFALIGN(resize_w * dtype_size, SAIL_ALIGN); // ceiling to 64 * N
            output.create(
                    handle_,
                    resize_h,
                    resize_w,
                    FORMAT_BGR_PLANAR, // force to this format
                    input.dtype(),
                    &stride
            );
            output.allocate();
        }

        int ret = bmcv_image_vpp_convert(
                handle_.data(),
                output_num,
                input.data(),
                &output.data(),
                &rect
        );

        if (ret == BM_NOT_SUPPORTED) {
            // vpp not support, try tpu resize
            bmcv_resize_t roi_attr[1];
            bmcv_resize_image resize_attr[1];
            memset(resize_attr, 0, sizeof(resize_attr));
            resize_attr[0].roi_num = 1;
            resize_attr[0].stretch_fit =1 ;
            roi_attr[0].start_x = 0;
            roi_attr[0].start_y = 0;
            roi_attr[0].in_width = input.width();
            roi_attr[0].in_height = input.height();
            roi_attr[0].out_width = output.width();
            roi_attr[0].out_height = output.height();
            resize_attr[0].resize_img_attr = &roi_attr[0];
            ret = bmcv_image_resize(handle_.data(), input_num, resize_attr, &input.data(), &output.data());
            if (BM_SUCCESS != ret) {
                SPDLOG_ERROR("bmcv_image_resize err={}", ret);
                print_image(input.data(), " src:");
                print_image(output.data(), " dst:");
            }
        }else {
            if (BM_SUCCESS != ret) {
                SPDLOG_ERROR("bmcv_image_vpp_convert err={}", ret);
                print_image(input.data(), " src:");
                print_image(output.data(), " dst:");
            }
        }


        return ret;
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
            BMImage &input,
            BMImage &output,
            int crop_x0,
            int crop_y0,
            int crop_w,
            int crop_h,
            int resize_w,
            int resize_h,
            PaddingAtrr &padding_in) {
        int input_num = 1;
        int output_num = 1; // should equal to size of crop windows

        bmcv_rect rect;
        rect.start_x = crop_x0;
        rect.start_y = crop_y0;
        rect.crop_w = crop_w;
        rect.crop_h = crop_h;

        if (output.is_created()) {
            output.reset(resize_w, resize_h);
        }

        if (!output.is_created()) {
            /* vpp limitation: 64-aligned */
            int dtype_size = bm_image_data_type_size(input.dtype());

            int stride = FFALIGN(resize_w * dtype_size, SAIL_ALIGN); // ceiling to 64 * N
            output.create(
                    handle_,
                    resize_h,
                    resize_w,
                    FORMAT_BGR_PLANAR, // force to this format
                    input.dtype(),
                    &stride
            );
            output.allocate();
        }

        // filling output
        bmcv_padding_atrr_t padding;
        padding.dst_crop_stx = padding_in.dst_crop_stx;
        padding.dst_crop_sty = padding_in.dst_crop_sty;
        padding.dst_crop_w = padding_in.dst_crop_w;
        padding.dst_crop_h = padding_in.dst_crop_h;
        padding.if_memset = 0;

        int width = output.data().width;
        int height = output.data().height;
        int stride = 0;
        int ret = bm_image_get_stride(output.data(), &stride);
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("bm_image_get_stride err={}", ret);
        }

        if (output.format() == FORMAT_BGR_PLANAR) {
            char color[4] = {0};
            bm_device_mem_t mem[4];
            // output.data().image_format = FORMAT_BGRP_SEPARATE;
            bm_status_t ret = bm_image_get_device_mem(output.data(), mem);
            output.data().image_format = FORMAT_BGR_PLANAR;

            color[0] = padding_in.padding_b;
            bm_memset_device_ext(handle_.data(), (void *) color, 1, mem[0]);

            color[0] = padding_in.padding_g;
            mem[0].u.device.device_addr += height * stride;
            bm_memset_device_ext(handle_.data(), (void *) color, 1, mem[0]);

            color[0] = padding_in.padding_r;
            mem[0].u.device.device_addr += height * stride;
            bm_memset_device_ext(handle_.data(), (void *) color, 1, mem[0]);
        } else if (output.format() == FORMAT_BGR_PACKED) {
            char color[4] = {0};
            color[0] = padding_in.padding_b;
            color[1] = padding_in.padding_g;
            color[2] = padding_in.padding_r;

            bm_device_mem_t mem;
            bm_image_get_device_mem(output.data(), &mem);

            bm_memset_device_ext(handle_.data(), (void *) color, 3, mem);
        } else if (output.format() == FORMAT_RGB_PLANAR) {
            char color[4] = {0};
            bm_device_mem_t mem[4];
            bm_image_get_device_mem(output.data(), mem);

            color[0] = padding_in.padding_r;
            bm_memset_device_ext(handle_.data(), (void *) color, 1, mem[0]);

            mem[0].u.device.device_addr += height * stride;
            color[0] = padding_in.padding_g;
            bm_memset_device_ext(handle_.data(), (void *) color, 1, mem[0]);

            color[0] = padding_in.padding_b;
            mem[0].u.device.device_addr += height * stride;
            bm_memset_device_ext(handle_.data(), (void *) color, 1, mem[0]);
        } else if (output.format() == FORMAT_RGB_PACKED) {
            char color[4] = {0};
            color[0] = padding_in.padding_r;
            color[1] = padding_in.padding_g;
            color[2] = padding_in.padding_b;
            bm_device_mem_t addr;
            bm_image_get_device_mem(output.data(), &addr);
            bm_memset_device_ext(handle_.data(), (void *) color, 3, addr);
        } else {
            spdlog::error(
                    "Only support image format FORMAT_BGR_PLANAR or FORMAT_RGB_PLANAR or FORMAT_BGR_PACKED or FORMAT_RGB_PACKED. Please convert it first.");
            return -1;
        }

        ret = bmcv_image_vpp_convert_padding(
                handle_.data(),
                output_num,
                input.data(),
                &output.data(),
                &padding,
                &rect
        );
        if (BM_SUCCESS != ret) {
            SPDLOG_ERROR("bmcv_image_vpp_convert_padding err={}", ret);
            print_image(input.data(), " src:");
            print_image(output.data(), " dst:");
        }

        return ret;
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
            BMImage &input,
            BMImage &output,
            const std::pair<
                    std::tuple<float, float, float>,
                    std::tuple<float, float, float>> &matrix
    ) {
        if (input.data().image_format != FORMAT_RGB_PLANAR && input.data().image_format != FORMAT_BGR_PLANAR) {
            spdlog::error(
                    "Only support image format FORMAT_BGR_PLANAR or FORMAT_RGB_PLANAR. Please convert it first, now is %d",
                    input.data().image_format);
            printf("format is %d\n", input.data().image_format);
            exit(SAIL_ERR_BMCV_FUNC);
        }

        int input_num = 1;
        int output_num = 1;

        bmcv_warp_matrix attr1;
        attr1.m[0] = std::get<0>(matrix.first);
        attr1.m[1] = std::get<1>(matrix.first);
        attr1.m[2] = std::get<2>(matrix.first);
        attr1.m[3] = std::get<0>(matrix.second);
        attr1.m[4] = std::get<1>(matrix.second);
        attr1.m[5] = std::get<2>(matrix.second);

        bmcv_warp_image_matrix attr0;
        attr0.matrix = &attr1;
        attr0.matrix_num = 1;

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

        int ret = bmcv_image_warp_affine(handle_.data(), input_num, &attr0, &input.data(), &output.data());

        return ret;
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
            BMImage &input,
            BMImage &output,
            const std::tuple<
                    std::pair<float, float>,
                    std::pair<float, float>,
                    std::pair<float, float>> &alpha_beta
    ) {
        int input_num = 1;
        int output_num = 1;

        bmcv_convert_to_attr attr;
        attr.alpha_0 = std::get<0>(alpha_beta).first;
        attr.beta_0 = std::get<0>(alpha_beta).second;
        attr.alpha_1 = std::get<1>(alpha_beta).first;
        attr.beta_1 = std::get<1>(alpha_beta).second;
        attr.alpha_2 = std::get<2>(alpha_beta).first;
        attr.beta_2 = std::get<2>(alpha_beta).second;

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

        int ret = bmcv_image_convert_to(handle_.data(), input_num, attr, &input.data(), &output.data());
        if (ret != BM_SUCCESS) {
            printf("bmcv_image_convert_to error, src.format=%d, dst.format=%d", input.format(), output.format());
        }

        return ret;
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

    bm_image_data_format_ext
    Bmcv::get_bm_image_data_format(
            bm_data_type_t dtype
    ) {
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

#endif //USE_BMCV

}  // namespace sail
