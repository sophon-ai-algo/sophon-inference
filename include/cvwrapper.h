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

/** @file     cvwrapper.h
 *  @brief    Header file of BMCV and BMDECODE
 *  @author   bitmain
 *  @version  2.0.3
 *  @date     2019-12-27
 */

#pragma once
#ifdef USE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavdevice/avdevice.h>
#include <libswscale/swscale.h>
}
#else
#ifndef FFALIGN
#define FFALIGN(x, a)(((x) + (a)-1) & ~((a)-1))
#endif
#endif

#ifdef USE_BMCV
#include <bmruntime_interface.h>
#include <bmlib_runtime.h>
#include <bmcv_api.h>
#include <bmcv_api_ext.h>
#endif

#ifdef PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#endif

#include <string>
#include <vector>
#include "tensor.h"
#include <iostream>
using namespace std;

namespace cv { class Mat; }

/// Namespace containing all symbols from the sail library.
namespace sail {

/**
 * @brief Set Decoder environment, must set befor Decoder Constructor, else use default values
 * refcounted_frames,extra_frame_buffer_num,rtsp_transport,stimeout,
 * rtsp_flags, buffer_size, max_delay, probesize, analyzeduration
 */
int DECL_EXPORT set_decoder_env(std::string env_name, std::string env_value);

class PaddingAtrr {
public:
    explicit PaddingAtrr(){};
    explicit PaddingAtrr(unsigned int crop_start_x,
        unsigned int crop_start_y,
        unsigned int crop_width,
        unsigned int crop_height,
        unsigned char padding_value_r,
        unsigned char padding_value_g,
        unsigned char padding_value_b);
    PaddingAtrr(const PaddingAtrr& other);

    void set_stx(unsigned int stx);
    void set_sty(unsigned int sty);
    void set_w(unsigned int w);
    void set_h(unsigned int h);
    void set_r(unsigned int r);
    void set_g(unsigned int g);
    void set_b(unsigned int b);

    unsigned int    dst_crop_stx;
    unsigned int    dst_crop_sty;
    unsigned int    dst_crop_w;
    unsigned int    dst_crop_h;
    unsigned char   padding_r;
    unsigned char   padding_g;
    unsigned char   padding_b;
};

#ifdef USE_OPENCV
/**
 * @brief Convert bm_data_type_t to opencv depth.
 *
 * @param dtype bm_data_type_t
 * @return opencv depth
 */
int get_cv_depth(bm_data_type_t dtype);

/**
 * @brief Convert data from cv::Mat to sail::Tensor.
 *
 * @param mat    Data with type cv:Mat
 * @param tensor Data with type sail::Tensor
 */
void mat_to_tensor(cv::Mat& mat, Tensor& tensor);

/**
 * @brief Convert data from vector of cv::Mat to sail::Tensor.
 *
 * @param mat    Data with type vector of cv:Mat
 * @param tensor Data with type sail::Tensor
 */
void mat_to_tensor(std::vector<cv::Mat>& mats, Tensor& tensor);
#endif

#ifdef USE_BMCV

class BMImage;

/**
 * @brief The wrapper of bmcv bm_image in sail for python api.
 */
class DECL_EXPORT BMImage {
 public:
  /**
   * @brief The default Constructor.
   */
  BMImage();
  /**
   * @brief Construct BMImage with bm_image.
   *
   * @param img Init input bm_image
   */
  BMImage(bm_image &img);
  /**
   * @brief The BMImage Constructor.
   *
   * @param handle A Handle instance
   * @param h      Image width
   * @param w      Image height
   * @param format Image format
   * @param dtype  Data type
   */
  BMImage(
      Handle&                  handle,
      int                      h,
      int                      w,
      bm_image_format_ext      format,
      bm_image_data_format_ext dtype);
  /**
   * @brief The BMImage Constructor.
   *
   * @param handle A Handle instance
   * @param h      Image width
   * @param w      Image height
   * @param format Image format
   * @param dtype  Data type
   */
  BMImage(
      Handle&                  handle,
      int                      h,
      int                      w,
      bm_image_format_ext      format,
      bm_image_data_format_ext dtype,
      int                      *stride);
  /**
   * @brief The copy constructor of BMImage.
   */
  BMImage(BMImage &&other);
  /**
   * @brief The assignment function of BMImage.
   */
  BMImage& operator=(BMImage &&other);
  BMImage& operator=(bm_image &&other);
  virtual ~BMImage();
  /**
   * @brief Get inner bm_image
   *
   * @return The inner bm_image
   */
  bm_image& data();
  bm_image data() const;
  /**
   * @brief Get the img width.
   *
   * @return the width of img
   */
  int width() const;
  /**
   * @brief Get the img height.
   *
   * @return the height of img
   */
  int height() const;
  /**
   * @brief Get the img format.
   *
   * @return the format of img
   */
  bm_image_format_ext format() const;
  /**
   * @brief Get the img data type.
   *
   * @return the data type of img
   */
  bm_image_data_format_ext dtype() const;

  /**
   * @brief Get device id of this image.
   *
   * @return Device id.
   */
  int get_device_id() const;

  bool need_to_free() const;
  int empty_check() const;
  int get_plane_num() const;
 protected:
  /// inner bm_image
  void reset(int w, int h);

 private:
  class BMImage_CC;
  class BMImage_CC* const _impl;

  BMImage(const BMImage &other) = delete;
  BMImage& operator=(const BMImage &other) = delete;

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

  friend class Bmcv;
};

template<std::size_t N>
class BMImageArray : public std::array<bm_image, N> {
 public:
  BMImageArray();
  BMImageArray(
      Handle                   &handle,
      int                      h,
      int                      w,
      bm_image_format_ext      format,
      bm_image_data_format_ext dtype);
  BMImageArray(
      Handle                   &handle,
      int                      h,
      int                      w,
      bm_image_format_ext      format,
      bm_image_data_format_ext dtype,
      int                      *stride);

  // BMImageArray(const std::vector<BMImage> &data);
  // int copy_from(Handle &handle, int i, BMImage &data);
  // int attach_from(Handle &handle, int i, BMImage &data);

  int copy_from(int i, BMImage &data);
  int attach_from(int i, BMImage &data);
  virtual ~BMImageArray();

  BMImageArray(BMImageArray &&other);
  BMImageArray& operator=(BMImageArray &&other);
  bool check_need_free() {return need_to_free_; };
  void set_need_free(bool value){need_to_free_ = value;};
  void create(
            Handle                   &handle,
            int                      h,
            int                      w,
            bm_image_format_ext      format,
            bm_image_data_format_ext dtype);

  void to_tensor(Tensor &tensor);

  /**
   * @brief Get device id of this image array.
   *
   * @return Device id.
   */
  int get_device_id();

 private:
  BMImageArray(const BMImageArray&) = delete;
  BMImageArray& operator=(const BMImageArray&) = delete;

  void create(
      Handle                   &handle,
      int                      h,
      int                      w,
      bm_image_format_ext      format,
      bm_image_data_format_ext dtype,
      int                      *stride);
  void destroy();
  //void allocate();
  bool is_created() const;
  bm_image_format_ext format(int index) const;
  void reset(int h, int w);

  bool need_to_free_;

  friend class Bmcv;
};

template class BMImageArray<1>;     //特化
template class BMImageArray<2>;     //特化
template class BMImageArray<3>;     //特化
template class BMImageArray<4>;     //特化
template class BMImageArray<8>;     //特化
template class BMImageArray<16>;    //特化
template class BMImageArray<32>;    //特化
template class BMImageArray<64>;    //特化
template class BMImageArray<128>;   //特化
template class BMImageArray<256>;   //特化

#endif

#ifdef USE_FFMPEG
/**
 * @brief A class of image frame read by FFMPEG.
 *        It is an inner class used by Decoder.
 */
class Frame {
 public:
  /**
   * @brief Constructor.
   */
  Frame() {
    frame_ = av_frame_alloc();
  }
  ~Frame() {
    av_frame_free(&frame_);
  }
  /**
   * @brief Get the pointer of AVFrame instance.
   *
   * @return Pointer of AVFrame instance.
   */
  AVFrame* get() {
    return frame_;
  }
  /**
   * @brief Get height of the frame.
   *
   * @return Height of the frame
   */
  int get_height() {
    return frame_->height;
  }
  /**
   * @brief Get width of the frame.
   *
   * @return Width of the frame
   */
  int get_width() {
    return frame_->width;
  }

  Frame& operator =(const Frame& other){
      if (this == &other) {
          return *this;
      }

      if (this->frame_ != nullptr) {
          av_frame_free(&frame_);
      }
      this->frame_ = other.frame_;
      return *this;
  }

  void set_frame(AVFrame *frame) {
      if (this->frame_ != nullptr) {
          av_frame_free(&frame_);
      }
      this->frame_ = frame;
  }

 private:
  /// Pointer to AVFrame instance
  AVFrame* frame_;
};

/**
 * @brief Decoder by VPU.
 *
 * Only format of AV_PIX_FMT_NV12 is supported.
 */
class DECL_EXPORT Decoder {
 public:
  /**
   * @brief Constructor.
   *
   * @param file_path  Path or rtsp url to the video/image file.
   * @param compressed Whether the format of decoded output is compressed NV12.
   * @param tpu_id     ID of TPU, there may be more than one TPU for PCIE mode.
   */
  explicit Decoder(
      const std::string& file_path,
      bool               compressed = true,
      int                tpu_id = 0);

  /**
   * @brief Destructor.
   */
  ~Decoder();

  /**
   * @brief Judge if the source is opened successfully.
   *
   * @return True if the source is opened successfully
   */
  bool is_opened();
  /**
   * @brief Get frame shape in the Decoder.
   *
   * @return Frame shape in the Decoder, [1, C, H, W]
   */
  std::vector<int> get_frame_shape();
  /**
   * @brief Read a BMImage from the image file.
   *
   * @param handle A bm_handle_t instance
   * @param image Reference of BMImage to be read to
   */
  int decode_jpeg(Handle& handle, BMImage& image);
  /**
   * @brief Read a bm_image from the image file.
   *
   * @param handle A bm_handle_t instance
   * @param image Reference of bm_image to be read to
   */
  int decode_jpeg(Handle& handle, bm_image& image);
  /**
   * @brief Read a BMImage from the Decoder.
   *
   * @param handle A bm_handle_t instance
   * @param image Reference of BMImage to be read to
   * @return 0 for success and 1 for failure
   */
  int read(Handle& handle, BMImage& image);
  /**
   * @brief Read a BMImage from the Decoder.
   *
   * @param handle A bm_handle_t instance
   * @return BMImage instance to be read to
   */
  BMImage read(Handle& handle);
  /**
   * @brief Read a bm_image from the Decoder.
   *
   * @param handle A bm_handle_t instance
   * @param image Reference of bm_image to be read to
   * @return 0 for success and 1 for failure
   */
  int read_(Handle& handle, bm_image& image);
  /**
   * @brief Read a bm_image from the Decoder.
   *
   * @param handle A bm_handle_t instance
   * @return bm_image instance to be read to
   */
  bm_image read_(Handle& handle);

  /**
   *  @brief Get the fps of the Video
   *  @return the fps of the video
   */
  float get_fps() const;

  /**
   * @brief Release Decoder.
   */
  void release();

  /**
   * @brief Reconnect Decoder.
   */
  int reconnect();

 private:
  class Decoder_CC;
  class Decoder_CC* const _impl;
};
#endif

#ifdef USE_BMCV

/**
 * @brief A class for image processing by VPP/TPU.
 */
class DECL_EXPORT Bmcv {
 public:
  /**
   * @brief Constructor.
   *
   * @param handle A Handle instance
   */
  explicit Bmcv(Handle &handle);
  ~Bmcv();

#if defined(USE_BMCV) && defined(USE_OPENCV)
  static int     mat_to_bm_image (cv::Mat &mat, BMImage &img);
  static BMImage mat_to_bm_image (cv::Mat &mat);

  static int     bm_image_to_mat(BMImage &img, cv::Mat &mat);
  static cv::Mat bm_image_to_mat(BMImage &img);
#endif

  /**
   * @brief Convert BMImage to tensor.
   *
   * @param img      Input image
   * @param tensor   Output tensor
   */
  void   bm_image_to_tensor (BMImage &img, Tensor &tensor);
  Tensor bm_image_to_tensor (BMImage &img);

  template<std::size_t N> void   bm_image_to_tensor (BMImageArray<N> &imgs, Tensor &tensor);
  template<std::size_t N> Tensor bm_image_to_tensor (BMImageArray<N> &imgs);

  /**
   * @brief Convert tensor to BMImage.
   *
   * @param tensor   Input tensor
   * @param img      Output image
   * @param bgr2rgb  swap color channel
   */
  void    tensor_to_bm_image (Tensor &tensor, BMImage &img, bool bgr2rgb=false);
  BMImage tensor_to_bm_image (Tensor &tensor, bool bgr2rgb=false);

  template<std::size_t N> void            tensor_to_bm_image (Tensor &tensor, BMImageArray<N> &imgs, bool bgr2rgb=false);
  template<std::size_t N> BMImageArray<N> tensor_to_bm_image (Tensor &tensor, bool bgr2rgb=false);

  /**
   * @brief Crop then resize an image.
   *
   * @param input    Input image
   * @param output   Output image
   * @param crop_x0  Start point x of the crop window
   * @param crop_y0  Start point y of the crop window
   * @param crop_w   Width of the crop window
   * @param crop_h   Height of the crop window
   * @param resize_w Target width
   * @param resize_h Target height
   * @return 0 for success and other for failure
   */
  int crop_and_resize(
      bm_image                     *input,
      bm_image                     *output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      int                          input_num = 1);

  int crop_and_resize(
      BMImage                      &input,
      BMImage                      &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h);

  BMImage crop_and_resize(
      BMImage                      &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h);

  template<std::size_t N>
  int crop_and_resize(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h);

  template<std::size_t N>
  BMImageArray<N> crop_and_resize(
      BMImageArray<N>              &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h);

  bm_image crop_and_resize_padding(
      bm_image                     &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in);

  BMImage crop_and_resize_padding(
      BMImage                     &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in);

  /**
   * @brief Crop an image with given window.
   *
   * @param input    Input image
   * @param output   Output image
   * @param crop_x0  Start point x of the crop window
   * @param crop_y0  Start point y of the crop window
   * @param crop_w   Width of the crop window
   * @param crop_h   Height of the crop window
   * @return 0 for success and other for failure
   */
  int crop(
      BMImage                      &input,
      BMImage                      &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h);

  BMImage crop(
      BMImage                      &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h);

  template<std::size_t N>
  int crop(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h);

  template<std::size_t N>
  BMImageArray<N> crop(
      BMImageArray<N>              &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h);

  /**
   * @brief Resize an image with interpolation of INTER_NEAREST.
   *
   * @param input    Input image
   * @param output   Output image
   * @param resize_w Target width
   * @param resize_h Target height
   * @return 0 for success and other for failure
   */
  int resize(
      BMImage                      &input,
      BMImage                      &output,
      int                          resize_w,
      int                          resize_h);

  BMImage resize(
      BMImage                      &input,
      int                          resize_w,
      int                          resize_h);

  template<std::size_t N>
  int resize(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      int                          resize_w,
      int                          resize_h);

  template<std::size_t N>
  BMImageArray<N> resize(
      BMImageArray<N>              &input,
      int                          resize_w,
      int                          resize_h);

  /**
   * @brief Crop then resize an image using vpp.
   *
   * @param input    Input image
   * @param output   Output image
   * @param crop_x0  Start point x of the crop window
   * @param crop_y0  Start point y of the crop window
   * @param crop_w   Width of the crop window
   * @param crop_h   Height of the crop window
   * @param resize_w Target width
   * @param resize_h Target height
   * @return 0 for success and other for failure
   */
  int vpp_crop_and_resize(
      bm_image                     *input,
      bm_image                     *output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      int                          input_num = 1);

  int vpp_crop_and_resize(
      BMImage                      &input,
      BMImage                      &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h);

  BMImage vpp_crop_and_resize(
      BMImage                      &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h);

  int vpp_crop_and_resize_padding(
      bm_image                     *input,
      bm_image                     *output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in,
      int                          input_num = 1);

  int vpp_crop_and_resize_padding(
      BMImage                      &input,
      BMImage                      &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in);

  BMImage vpp_crop_and_resize_padding(
      BMImage                      &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in);


  template<std::size_t N>
  int vpp_crop_and_resize(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h);

  template<std::size_t N>
  BMImageArray<N> vpp_crop_and_resize(
      BMImageArray<N>              &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h);
  template<std::size_t N>
  int vpp_crop_and_resize_padding(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in);

  template<std::size_t N>
  BMImageArray<N> vpp_crop_and_resize_padding(
      BMImageArray<N>              &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in);

  /**
   * @brief Crop an image with given window using vpp.
   *
   * @param input    Input image
   * @param output   Output image
   * @param crop_x0  Start point x of the crop window
   * @param crop_y0  Start point y of the crop window
   * @param crop_w   Width of the crop window
   * @param crop_h   Height of the crop window
   * @return 0 for success and other for failure
   */
  int vpp_crop(
      BMImage                      &input,
      BMImage                      &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h);

  BMImage vpp_crop(
      BMImage                      &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h);

  int vpp_crop_padding(
      BMImage                      &input,
      BMImage                      &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      PaddingAtrr                  &padding_in);

  BMImage vpp_crop_padding(
      BMImage                      &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      PaddingAtrr                  &padding_in);

  template<std::size_t N>
  int vpp_crop(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h);

  template<std::size_t N>
  BMImageArray<N> vpp_crop(
      BMImageArray<N>              &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h);

  template<std::size_t N>
  int vpp_crop_padding(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      PaddingAtrr                  &padding_in);

  template<std::size_t N>
  BMImageArray<N> vpp_crop_padding(
      BMImageArray<N>              &input,
      int                          crop_x0,
      int                          crop_y0,
      int                          crop_w,
      int                          crop_h,
      PaddingAtrr                  &padding_in);

  /**
   * @brief Resize an image with interpolation of INTER_NEAREST using vpp.
   *
   * @param input    Input image
   * @param output   Output image
   * @param resize_w Target width
   * @param resize_h Target height
   * @return 0 for success and other for failure
   */
  int vpp_resize(
      BMImage                      &input,
      BMImage                      &output,
      int                          resize_w,
      int                          resize_h);

  BMImage vpp_resize(
      BMImage                      &input,
      int                          resize_w,
      int                          resize_h);

  int vpp_resize_padding(
      BMImage                      &input,
      BMImage                      &output,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in);

  BMImage vpp_resize_padding(
      BMImage                      &input,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in);

  template<std::size_t N>
  int vpp_resize(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      int                          resize_w,
      int                          resize_h);

  template<std::size_t N>
  BMImageArray<N> vpp_resize(
      BMImageArray<N>              &input,
      int                          resize_w,
      int                          resize_h);

  template<std::size_t N>
  int vpp_resize_padding(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in);

  template<std::size_t N>
  BMImageArray<N> vpp_resize_padding(
      BMImageArray<N>              &input,
      int                          resize_w,
      int                          resize_h,
      PaddingAtrr                  &padding_in);

  /**
   * @brief Applies an affine transformation to an image.
   *
   * @param input    Input image
   * @param output   Output image
   * @param matrix   2x3 transformation matrix
   * @return 0 for success and other for failure
   */
  int warp(
      bm_image *input,
      bm_image *output,
      const std::pair<
        std::tuple<float, float, float>,
        std::tuple<float, float, float>> *matrix,
      int input_num = 1);

  int warp(
      BMImage                            &input,
      BMImage                            &output,
      const std::pair<
        std::tuple<float, float, float>,
        std::tuple<float, float, float>> &matrix);

  BMImage warp(
      BMImage                            &input,
      const std::pair<
        std::tuple<float, float, float>,
        std::tuple<float, float, float>> &matrix);

  template<std::size_t N>
  int warp(
      BMImageArray<N>                          &input,
      BMImageArray<N>                          &output,
      const std::array<
        std::pair<
          std::tuple<float, float, float>,
          std::tuple<float, float, float>>, N> &matrix);

  template<std::size_t N>
  BMImageArray<N> warp(
      BMImageArray<N>                          &input,
      const std::array<
        std::pair<
          std::tuple<float, float, float>,
          std::tuple<float, float, float>>, N> &matrix);

  /**
   * @brief Applies a linear transformation to an image.
   *
   * @param input        Input image
   * @param output       Output image
   * @param alpha_beta   (a0, b0), (a1, b1), (a2, b2) factors
   * @return 0 for success and other for failure
   */
  int convert_to(
      bm_image *input,
      bm_image *output,
      const std::tuple<
        std::pair<float, float>,
        std::pair<float, float>,
        std::pair<float, float>>   &alpha_beta,
        int input_num = 1);

  int convert_to(
      BMImage                      &input,
      BMImage                      &output,
      const std::tuple<
        std::pair<float, float>,
        std::pair<float, float>,
        std::pair<float, float>>   &alpha_beta);

  BMImage convert_to(
      BMImage                      &input,
      const std::tuple<
        std::pair<float, float>,
        std::pair<float, float>,
        std::pair<float, float>>   &alpha_beta);

  template<std::size_t N>
  int convert_to(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output,
      const std::tuple<
        std::pair<float, float>,
        std::pair<float, float>,
        std::pair<float, float>>   &alpha_beta);

  template<std::size_t N>
  BMImageArray<N> convert_to(
      BMImageArray<N>              &input,
      const std::tuple<
        std::pair<float, float>,
        std::pair<float, float>,
        std::pair<float, float>>   &alpha_beta);

  /**
   * @brief Convert an image from YUV to BGR.
   *
   * @param input    Input image
   * @param output   Output image
   * @return 0 for success and other for failure
   */
  int yuv2bgr(
      BMImage                      &input,
      BMImage                      &output);

  BMImage yuv2bgr(
      BMImage                      &input);

  template<std::size_t N>
  int yuv2bgr(
      BMImageArray<N>              &input,
      BMImageArray<N>              &output);

  template<std::size_t N>
  BMImageArray<N> yuv2bgr(
      BMImageArray<N>              &input);

  /**
   * @brief Convert an image to BGR PLANAR format using vpp.
   output.create
   * @param input    Input image
   * @param output   Output image
   * @return 0 for success and other for failure
   */
  int vpp_convert_format(
      BMImage          &input,
      BMImage          &output
  );

  BMImage vpp_convert_format(
      BMImage          &input
  );

  template<std::size_t N>
  int vpp_convert_format(
      BMImageArray<N>  &input,
      BMImageArray<N>  &output
  );

  template<std::size_t N>
  BMImageArray<N> vpp_convert_format(
      BMImageArray<N>  &input
  );

  /**
   * @brief Convert an image to BGR PLANAR format.
   *
   * @param input    Input image
   * @param output   Output image
   * @return 0 for success and other for failure
   */
  int convert_format(
      BMImage          &input,
      BMImage          &output
  );

  BMImage convert_format(
      BMImage          &input
  );

  template<std::size_t N>
  int convert_format(
      BMImageArray<N>  &input,
      BMImageArray<N>  &output
  );

  template<std::size_t N>
  BMImageArray<N> convert_format(
      BMImageArray<N>  &input
  );

  /**
   * @brief Draw a rectangle on input image.
   *
   * @param image      Input image
   * @param x0         Start point x of rectangle
   * @param y0         Start point y of rectangle
   * @param w          Width of rectangle
   * @param h          Height of rectangle
   * @param color      Color of rectangle
   * @param thickness  Thickness of rectangle
   * @return 0 for success and other for failure
   */
  int rectangle(
      const BMImage                   &image,
      int                             x0,
      int                             y0,
      int                             w,
      int                             h,
      const std::tuple<int, int, int> &color, // BGR
      int                             thickness=1
  );

  int rectangle_(
      const bm_image                  &image,
      int                             x0,
      int                             y0,
      int                             w,
      int                             h,
      const std::tuple<int, int, int> &color, // BGR
      int                             thickness=1
  );

  /**
   * @brief put text on input image
   * 
   * @param image     Input image
   * @param text      Text string to be drawn
   * @param x         Start x
   * @param y         Start y
   * @param color     Color of text
   * @param fontScale Font scale factor that is multiplied by the font-specific base size
   * @param thickness Thickness of the lines used to draw a text
   * @return int 
   */
  int putText(
      const BMImage                   &image,
      const std::string               &text,
      int                             x,
      int                             y,
      const std::tuple<int, int, int> &color, // BGR
      float                           fontScale,
      int                             thickness=1
  );

  int putText_(
      const bm_image                  &image,
      const std::string               &text,
      int                             x,
      int                             y,
      const std::tuple<int, int, int> &color, // BGR
      float                           fontScale,
      int                             thickness=1
  );

  /** @brief output = input1 * alpha + input2 * beta + gamma
   */
  int image_add_weighted(
      BMImage           &input1,
      float             alpha,
      BMImage           &input2,
      float             beta,
      float             gamma,
      BMImage           &output
  );

  BMImage image_add_weighted(
      BMImage           &input1,
      float             alpha,
      BMImage           &input2,
      float             beta,
      float             gamma
  );

  /**@brief Copy input image to output
   * @param input   Input image
   * @param output  Output image
   * @start_x       Target starting point x
   * @start_y       Target starting point y
   */
  int image_copy_to(bm_image input, bm_image output, int start_x, int start_y);

  int image_copy_to(BMImage &input, BMImage &output, int start_x = 0, int start_y = 0);

  template<std::size_t N>
  int image_copy_to(BMImageArray<N> &input, BMImageArray<N> &output, int start_x = 0, int start_y = 0);

  /**@brief Copy input image to output with padding
   * @param input   Input image
   * @param output  Output image
   * @param start_x       Target starting point x
   * @param start_y       Target starting point y
   * @param padding_r     padding value of r
   * @param padding_g     padding value of g
   * @param padding_b     padding value of b
   */
  int image_copy_to_padding(bm_image input, bm_image output,
    unsigned int padding_r, unsigned int padding_g, unsigned int padding_b,
    int start_x, int start_y);

  int image_copy_to_padding(BMImage &input, BMImage &output,
    unsigned int padding_r, unsigned int padding_g, unsigned int padding_b,
    int start_x = 0, int start_y = 0);
  
  template<std::size_t N>
  int image_copy_to_padding(BMImageArray<N> &input, BMImageArray<N> &output, 
    unsigned int padding_r, unsigned int padding_g, unsigned int padding_b,
    int start_x = 0, int start_y = 0);

  /**
   * @brief Save the image to the specified file.
   *
   * @param filename   Name of the file
   * @param image      Image to be saved
   * @return 0 for success and other for failure
   */
  int imwrite(
      const std::string &filename,
      const BMImage     &image
  );

  int imwrite_(
          const std::string &filename,
          const bm_image     &image);
  /**
   * @brief Get Handle instance.
   *
   * @return Handle instance
   */
  Handle get_handle();

  /**
   * @brief Do nms use tpu
   * 
   * @param input_proposal input proposal objects
   * @param threshold      nms threshold
   * @param proposal_size  proposal size
   * @return result boxes [for c++, result memory should free by user]
   */
  nms_proposal_t* nms(face_rect_t *input_proposal,int proposal_size, float threshold);

  /**
   * @brief Applies a perspective transformation to an image.
   * 
   * @param input         Input image
   * @param coordinate    coordinate of left_top, right_top, left_bottom, right_bottom 
   * @param output_width  Output width
   * @param output_height Output height
   * @param format        Output format, only FORMAT_BGR_PLANAR,FORMAT_RGB_PLANAR
   * @param dtype         Output dtype, only DATA_TYPE_EXT_1N_BYTE,DATA_TYPE_EXT_4N_BYTE
   * @param use_bilinear  Bilinear use flag
   * @return Output image
   */

  BMImage warp_perspective(
    BMImage                     &input,
    const std::tuple<
      std::pair<int,int>,
      std::pair<int,int>,
      std::pair<int,int>,
      std::pair<int,int>>       &coordinate,
    int                         output_width,
    int                         output_height,
    bm_image_format_ext         format = FORMAT_BGR_PLANAR,
    bm_image_data_format_ext    dtype = DATA_TYPE_EXT_1N_BYTE,
    int                         use_bilinear = 0);


#ifdef PYTHON
  pybind11::array_t<float> nms(pybind11::array_t<float> input_proposal, float threshold);
#endif

  bm_data_type_t           get_bm_data_type(bm_image_data_format_ext fmt);
  bm_image_data_format_ext get_bm_image_data_format(bm_data_type_t dtype);

 private:
  Handle handle_;

  template<std::size_t N>
  void check_create(BMImageArray<N>& image,int height, int width, bm_image_data_format_ext dtype, bool reset = true,
    bm_image_format_ext fmt = FORMAT_BGR_PLANAR, int *stride = nullptr);
  
  void check_create(BMImage& image,int height, int width, bm_image_data_format_ext dtype, 
    bm_image_format_ext fmt = FORMAT_BGR_PLANAR, int *stride = nullptr);
};

template<std::size_t N>
void Bmcv::check_create(BMImageArray<N>& image, int height, int width, bm_image_data_format_ext dtype, bool reset,
  bm_image_format_ext fmt, int *stride){
  if (reset && image.is_created()) {
      image.reset(height, width);
  }
  if(!image.is_created()){
    image.create(handle_, height, width, fmt, dtype, stride);
  }
}

template<std::size_t N>
void Bmcv::bm_image_to_tensor (BMImageArray<N> &imgs, Tensor &tensor)
{
  return imgs.to_tensor(tensor);
}

template<std::size_t N>
Tensor Bmcv::bm_image_to_tensor (BMImageArray<N> &imgs)
{
  Tensor tensor(get_handle());
  bm_image_to_tensor(imgs, tensor);
  return std::move(tensor);
}

template<std::size_t N>
void Bmcv::tensor_to_bm_image (Tensor &tensor, BMImageArray<N> &imgs, bool bgr2rgb)
{
  auto shape = tensor.shape();
  int n = shape[0];
  if (n != N) {
    spdlog::error("Batch size mis-matched.");
    return;
  }

  int h = shape[2];
  int w = shape[3];


  if (imgs.is_created()) {
      imgs.destroy();
  }

  bm_image_data_format_ext dtype = get_bm_image_data_format(tensor.dtype());

  check_create(imgs, h, w, dtype, false, bgr2rgb ? FORMAT_RGB_PLANAR : FORMAT_BGR_PLANAR);

  bm_device_mem_t mem = tensor.dev_data();
  bm_image_attach_contiguous_mem(imgs.size(), imgs.data(), mem);
}

template<std::size_t N>
BMImageArray<N> Bmcv::tensor_to_bm_image (Tensor &tensor, bool bgr2rgb)
{
  BMImageArray<N> imgs;
  tensor_to_bm_image(tensor, imgs, bgr2rgb);
  return std::move(imgs);
}

template<std::size_t N>
int Bmcv::crop_and_resize(
  BMImageArray<N> &input,
  BMImageArray<N> &output,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h,
  int             resize_w,
  int             resize_h
) {
  check_create(output, resize_h, resize_w, input[0].data_type);
  int ret = 0;
  for(int i = 0; i < N; ++i) {
    ret = crop_and_resize(&input.at(i), &output.at(i),
          crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h);
    if (ret != 0){
      break;
    }
  }
  return ret;
}

template<std::size_t N>
BMImageArray<N> Bmcv::crop_and_resize(
  BMImageArray<N> &input,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h,
  int             resize_w,
  int             resize_h
) {
  BMImageArray<N> output;
  crop_and_resize(input, output, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::crop(
  BMImageArray<N>              &input,
  BMImageArray<N>              &output,
  int                          crop_x0,
  int                          crop_y0,
  int                          crop_w,
  int                          crop_h
) {
  return crop_and_resize(input, output, crop_x0, crop_y0, crop_w, crop_h, crop_w, crop_h);
}

template<std::size_t N>
BMImageArray<N> Bmcv::crop(
  BMImageArray<N>              &input,
  int                          crop_x0,
  int                          crop_y0,
  int                          crop_w,
  int                          crop_h
) {
  BMImageArray<N> output;
  crop(input, output, crop_x0, crop_y0, crop_w, crop_h);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::resize(
  BMImageArray<N>              &input,
  BMImageArray<N>              &output,
  int                          resize_w,
  int                          resize_h
) {
  return crop_and_resize(input, output, 0, 0, input[0].width, input[0].height, resize_w, resize_h);
}

template<std::size_t N>
BMImageArray<N> Bmcv::resize(
  BMImageArray<N>              &input,
  int                          resize_w,
  int                          resize_h
) {
  BMImageArray<N> output;
  resize(input, output, resize_w, resize_h);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::vpp_crop_and_resize(
  BMImageArray<N> &input,
  BMImageArray<N> &output,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h,
  int             resize_w,
  int             resize_h
) {

  check_create(output, resize_h, resize_w, input[0].data_type);
  int ret = 0;
  for (int i = 0; i < N; ++i){
    ret = vpp_crop_and_resize(&input.at(i), &output.at(i),
          crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h);
    if (ret != 0){
      break;
    }
  }
  return ret;
}

template<std::size_t N>
BMImageArray<N> Bmcv::vpp_crop_and_resize(
  BMImageArray<N> &input,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h,
  int             resize_w,
  int             resize_h
) {
  BMImageArray<N> output;
  vpp_crop_and_resize(input, output, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::vpp_crop_and_resize_padding(
  BMImageArray<N> &input,
  BMImageArray<N> &output,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h,
  int             resize_w,
  int             resize_h,
  PaddingAtrr     &padding_in
) {
  check_create(output, resize_h, resize_w, input[0].data_type);
 
  return vpp_crop_and_resize_padding(input.data(), output.data(), crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h,padding_in,N);
}

template<std::size_t N>
BMImageArray<N> Bmcv::vpp_crop_and_resize_padding(
  BMImageArray<N> &input,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h,
  int             resize_w,
  int             resize_h,
  PaddingAtrr     &padding_in
) {
  BMImageArray<N> output;
  vpp_crop_and_resize_padding(input, output, crop_x0, crop_y0, crop_w, crop_h, resize_w, resize_h,padding_in);
  return std::move(output);
}
template<std::size_t N>
int Bmcv::vpp_crop(
  BMImageArray<N> &input,
  BMImageArray<N> &output,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h
) {
  return vpp_crop_and_resize(input, output, crop_x0, crop_y0, crop_w, crop_h, crop_w, crop_h);
}

template<std::size_t N>
BMImageArray<N> Bmcv::vpp_crop(
  BMImageArray<N> &input,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h
) {
  BMImageArray<N> output;
  vpp_crop(input, output, crop_x0, crop_y0, crop_w, crop_h);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::vpp_crop_padding(
  BMImageArray<N> &input,
  BMImageArray<N> &output,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h,
  PaddingAtrr     &padding_in
) {
  return vpp_crop_and_resize_padding(input, output, crop_x0, crop_y0, crop_w, crop_h, crop_w, crop_h, padding_in);
}

template<std::size_t N>
BMImageArray<N> Bmcv::vpp_crop_padding(
  BMImageArray<N> &input,
  int             crop_x0,
  int             crop_y0,
  int             crop_w,
  int             crop_h,
  PaddingAtrr     &padding_in
) {
  BMImageArray<N> output;
  vpp_crop_padding(input, output, crop_x0, crop_y0, crop_w, crop_h, padding_in);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::vpp_resize(
  BMImageArray<N> &input,
  BMImageArray<N> &output,
  int             resize_w,
  int             resize_h
) {
  return vpp_crop_and_resize(input, output, 0, 0, input[0].width, input[0].height, resize_w, resize_h);
}

template<std::size_t N>
BMImageArray<N> Bmcv::vpp_resize(
  BMImageArray<N> &input,
  int             resize_w,
  int             resize_h
) {
  BMImageArray<N> output;
  vpp_resize(input, output, resize_w, resize_h);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::vpp_resize_padding(
  BMImageArray<N> &input,
  BMImageArray<N> &output,
  int             resize_w,
  int             resize_h,
  PaddingAtrr     &padding_in
) {
  return vpp_crop_and_resize_padding(input, output, 0, 0, input[0].width, input[0].height, resize_w, resize_h, padding_in);
}

template<std::size_t N>
BMImageArray<N> Bmcv::vpp_resize_padding(
  BMImageArray<N> &input,
  int             resize_w,
  int             resize_h,
  PaddingAtrr     &padding_in
) {
  BMImageArray<N> output;
  vpp_resize_padding(input, output, resize_w, resize_h, padding_in);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::warp(
  BMImageArray<N>                          &input,
  BMImageArray<N>                          &output,
  const std::array<
    std::pair<
      std::tuple<float, float, float>,
      std::tuple<float, float, float>>, N> &matrix
) {
  check_create(output, input[0].height, input[0].width, input[0].data_type,input[0].image_format);

  int ret = warp(input.data(), output.data(), matrix.data(), N);
  return ret;
}

template<std::size_t N>
BMImageArray<N> Bmcv::warp(
  BMImageArray<N>                          &input,
  const std::array<
    std::pair<
      std::tuple<float, float, float>,
      std::tuple<float, float, float>>, N> &matrix
) {
  BMImageArray<N> output;
  warp(input, output, matrix);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::convert_to(
  BMImageArray<N>            &input,
  BMImageArray<N>            &output,
  const std::tuple<
    std::pair<float, float>,
    std::pair<float, float>,
    std::pair<float, float>> &alpha_beta
) {
  check_create(output, input[0].height, input[0].width, input[0].data_type);

  return convert_to(input.data(), output.data(), alpha_beta, N);
}

template<std::size_t N>
BMImageArray<N> Bmcv::convert_to(
  BMImageArray<N>            &input,
  const std::tuple<
    std::pair<float, float>,
    std::pair<float, float>,
    std::pair<float, float>> &alpha_beta
) {
  BMImageArray<N> output;
  convert_to(input, output, alpha_beta);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::yuv2bgr(
    BMImageArray<N> &input,
  BMImageArray<N> &output
) {
 
  check_create(output, input[0].height, input[0].width, input[0].data_type, false);
  
  int ret = bmcv_image_yuv2bgr_ext(handle_.data(), N, input.data(), output.data());

  return ret;
}

template<std::size_t N>
BMImageArray<N> Bmcv::yuv2bgr(
  BMImageArray<N> &input
) {
  BMImageArray<N> output;
  yuv2bgr(input, output);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::vpp_convert_format(
  BMImageArray<N> &input,
  BMImageArray<N> &output
) {
 
  check_create(output, input[0].height, input[0].width, input[0].data_type, false);

  int ret = 0;
  for (int i = 0; i < N; i ++) {
    ret = bmcv_image_vpp_convert(
      handle_.data(),
      1,
      input[i],
      &output[i]
    );
    if (ret != 0) break;
  }

  return ret;
}

template<std::size_t N>
BMImageArray<N> Bmcv::vpp_convert_format(
  BMImageArray<N> &input
) {
  BMImageArray<N> output;
  vpp_convert_format(input, output);
  return std::move(output);
}

template<std::size_t N>
int Bmcv::convert_format(
  BMImageArray<N>  &input,
  BMImageArray<N>  &output
) {
  check_create(output,input[0].height,input[0].width,input[0].data_type, false);
  int ret = bmcv_image_storage_convert(
    handle_.data(),
    N,
    input.data(),
    output.data()
  );

  return ret;
}

template<std::size_t N>
BMImageArray<N> Bmcv::convert_format(
  BMImageArray<N>  &input
) {
  BMImageArray<N> output;
  convert_format(input, output);
  return std::move(output);
}

  template<std::size_t N>
  int Bmcv::image_copy_to(BMImageArray<N> &input, BMImageArray<N> &output, int start_x, int start_y)
  {
    if (!output.is_created() || !input.is_created()){
      SPDLOG_ERROR("input or output must be created before!");
      exit(SAIL_ERR_BMCV_TRANS);
    }
    for(int i = 0; i < N; ++i) {
      int ret = image_copy_to(input.at(i), output.at(i), start_x, start_y);
      if (ret != 0){
        exit(SAIL_ERR_BMCV_TRANS);
      }
    }
    return BM_SUCCESS;
  }

  template<std::size_t N>
  int Bmcv::image_copy_to_padding(BMImageArray<N> &input, BMImageArray<N> &output, 
    unsigned int padding_r, unsigned int padding_g, unsigned int padding_b,
    int start_x, int start_y)
  {
    if (!output.is_created() || !input.is_created()){
      SPDLOG_ERROR("input or output must be created before!");
      exit(SAIL_ERR_BMCV_TRANS);
    }
    for(int i = 0; i < N; ++i) {
      int ret = image_copy_to_padding(input.at(i), output.at(i),padding_r, padding_g, padding_b, start_x, start_y);
      if (ret != 0){
        exit(SAIL_ERR_BMCV_TRANS);
      }
    }
    return BM_SUCCESS;
  }
#endif

}  // namespace sail
