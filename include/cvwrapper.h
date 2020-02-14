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
}
#endif

#ifdef USE_BMCV
#include <bmruntime_interface.h>
#include <bmlib_runtime.h>
#include <op_code.h>
#include <bmcv_api.h>
#include <bmcv_api_ext.h>
#endif

#include <string>
#include <vector>
#include "tensor.h"

namespace cv { class Mat; };

/// Namespace containing all symbols from the sail library.
namespace sail {

#ifndef USE_PCIE
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
class BMImage {
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

 protected:
  /// inner bm_image
  bm_image img_;

 private:
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

  bool need_to_free_;

  friend class Bmcv;
  friend bm_image move(BMImage  &img);
  friend BMImage  move(bm_image &img);
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

  virtual ~BMImageArray();

  BMImageArray(BMImageArray &&other);
  BMImageArray& operator=(BMImageArray &&other);

 private:
  BMImageArray(const BMImageArray&) = delete;
  BMImageArray& operator=(const BMImageArray&) = delete;

  void create(
      Handle                   &handle,
      int                      h,
      int                      w,
      bm_image_format_ext      format,
      bm_image_data_format_ext dtype,
      int                      *stride=NULL);
  void destroy();
  void allocate();
  bool is_created() const;

  bool need_to_free_;

  friend class Bmcv;
};

template<std::size_t N>
BMImageArray<N>::BMImageArray() : need_to_free_(false) {}

template<std::size_t N>
BMImageArray<N>::BMImageArray(
    Handle                   &handle,
    int                      h,
    int                      w,
    bm_image_format_ext      format,
    bm_image_data_format_ext dtype
) : need_to_free_(false) {
  create(handle, h, w, format, dtype);
  allocate();
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
  allocate();
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
void BMImageArray<N>::create(
  Handle                   &handle,
  int                      h,
  int                      w,
  bm_image_format_ext      format,
  bm_image_data_format_ext dtype,
  int                      *stride
) {
  for (size_t i = 0; i < N; i ++) {
    bm_image_create(handle.data(), h, w, format, dtype, &this->at(i), stride);
  }
}

template<std::size_t N>
void BMImageArray<N>::destroy() {
  if (need_to_free_) {
    bm_image_free_contiguous_mem(N, this->data());
    need_to_free_ = false;
  }
  for (size_t i = 0; i < N; i ++) {
    bm_image_destroy(this->at(i));
  }
}

template<std::size_t N>
void BMImageArray<N>::allocate() {
  bm_image_alloc_contiguous_mem(N, this->data());
  need_to_free_ = true;
}

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

 private:
  /// Pointer to AVFrame instance
  AVFrame* frame_;
};

/**
 * @brief Decoder by VPU.
 *
 * Only format of AV_PIX_FMT_NV12 is supported.
 */
class Decoder {
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
  void decode_jpeg(Handle& handle, BMImage& image);
  /**
   * @brief Read a bm_image from the image file.
   *
   * @param handle A bm_handle_t instance
   * @param image Reference of bm_image to be read to
   */
  void decode_jpeg(Handle& handle, bm_image& image);
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

 private:
  /**
   * @brief Read a frame from the Decoder.
   *
   * @param frame Reference of frame to be read to
   * @return 0 for success and 1 for failure
   */
  int read(Frame& frame);
  /**
   * @brief Convert frame with format of AV_PIX_FMT_NV12 to bm_image.
   *
   * @param image Reference of BMImage to convert to
   */
  void nv12_frame_to_image(Handle& handle, bm_image& image);

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
  /// fps of the Decoder.
  int fps_;
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
  /// Indicator of whether the input source is image file.
  bool is_image_;
  /// Flag of whether to read frame from buffer.
  bool flush_;
};
#endif

#ifdef USE_BMCV

/**
 * @brief A class for image processing by VPP/TPU.
 */
class Bmcv {
 public:
  /**
   * @brief Constructor.
   *
   * @param handle A Handle instance
   */
  explicit Bmcv(Handle &handle);
  ~Bmcv();

#ifndef USE_PCIE
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

  /**
   * @brief Applies an affine transformation to an image.
   *
   * @param input    Input image
   * @param output   Output image
   * @param matrix   2x3 transformation matrix
   * @return 0 for success and other for failure
   */
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

  /**
   * @brief Get Handle instance.
   *
   * @return Handle instance
   */
  Handle get_handle();

  bm_data_type_t           get_bm_data_type(bm_image_data_format_ext fmt);
  bm_image_data_format_ext get_bm_image_data_format(bm_data_type_t dtype);

 private:
  Handle handle_;
};

template<std::size_t N>
void Bmcv::bm_image_to_tensor (BMImageArray<N> &imgs, Tensor &tensor)
{
  if (imgs[0].image_format != FORMAT_BGR_PLANAR) {
    spdlog::error("Only support image format FORMAT_BGR_PLANAR. Please convert it first.");
    return;
  }

  bm_data_type_t dtype = get_bm_data_type(imgs[0].data_type);
  bm_device_mem_t addr;
  bm_image_get_contiguous_device_mem(imgs.size(), imgs.data(), &addr);

  int h = imgs[0].height;
  int w = imgs[0].width;

  tensor.reset({imgs.size(), 3, h, w}, dtype);
  tensor.reset_dev_data(addr);
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

  bm_image_data_format_ext dtype = get_bm_image_data_format(tensor.dtype());

  imgs.create(
    handle_,
    h,
    w,
    bgr2rgb ? FORMAT_RGB_PLANAR : FORMAT_BGR_PLANAR,
    dtype
  );

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
  bmcv_resize_t attr1;
  attr1.start_x    = crop_x0;
  attr1.start_y    = crop_y0;
  attr1.in_width   = crop_w;
  attr1.in_height  = crop_h;
  attr1.out_width  = resize_w;
  attr1.out_height = resize_h;

  bmcv_resize_image attr0;
  attr0.resize_img_attr = &attr1;
  attr0.roi_num         = 1;
  attr0.stretch_fit     = 1;
  attr0.interpolation   = BMCV_INTER_NEAREST;

  if (!output.is_created()) {
    output.create(
      handle_,
      resize_h,
      resize_w,
      FORMAT_BGR_PLANAR,
      input[0].data_type
    );
    output.allocate();
  }

  int ret = bmcv_image_resize(
    handle_.data(),
    N,
    &attr0,
    input.data(),
    output.data()
  );

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
  bmcv_rect_t rect;
  rect.start_x = crop_x0;
  rect.start_y = crop_y0;
  rect.crop_w  = crop_w;
  rect.crop_h  = crop_h;

  if (!output.is_created()) {
    /* vpp limitation: 64-aligned */
    int stride = ((resize_w + (64 - 1)) >> 6) << 6; // ceiling to 64 * N

    output.create(
      handle_,
      resize_h,
      resize_w,
      FORMAT_BGR_PLANAR,
      input[0].data_type,
      &stride
    );
    output.allocate();
  }

  int ret = 0;
  for (size_t i = 0; i < N; i ++) {
    ret = bmcv_image_vpp_convert(
      handle_.data(),
      1,
      input[i],
      &output[i],
      &rect
    );
    if (ret != 0) break;
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
int Bmcv::warp(
  BMImageArray<N>                          &input,
  BMImageArray<N>                          &output,
  const std::array<
    std::pair<
      std::tuple<float, float, float>,
      std::tuple<float, float, float>>, N> &matrix
) {
  for (int i = 0; i < N; i ++) {
    if (input[i].image_format != FORMAT_RGB_PLANAR && input[i].image_format != FORMAT_BGR_PLANAR) {
      spdlog::error("Only support image format FORMAT_BGR_PLANAR or FORMAT_RGB_PLANAR. Please convert it first.");
      return -1;
    }
  }

  bmcv_warp_image_matrix attr0[N];
  for (int i = 0; i < N; i ++) {
    bmcv_warp_matrix attr1;
    attr1.m[0] = std::get<0>(matrix[i].first);
    attr1.m[1] = std::get<1>(matrix[i].first);
    attr1.m[2] = std::get<2>(matrix[i].first);
    attr1.m[3] = std::get<0>(matrix[i].second);
    attr1.m[4] = std::get<1>(matrix[i].second);
    attr1.m[5] = std::get<2>(matrix[i].second);

    attr0[N].matrix = &attr1;
    attr0[N].matrix_num = 1;
  }

  if (!output.is_created()) {
    output.create(
      handle_,
      input[0].height,
      input[0].width,
      input[0].image_format,
      input[0].data_type
    );
    output.allocate();
  }

  int ret = bmcv_image_warp(handle_.data(), N, attr0, input.data(), output.data());

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
  bmcv_convert_to_attr attr;
  attr.alpha_0 = std::get<0>(alpha_beta).first;
  attr.beta_0  = std::get<0>(alpha_beta).second;
  attr.alpha_1 = std::get<1>(alpha_beta).first;
  attr.beta_1  = std::get<1>(alpha_beta).second;
  attr.alpha_2 = std::get<2>(alpha_beta).first;
  attr.beta_2  = std::get<2>(alpha_beta).second;

  if (!output.is_created()) {
    output.create(
      handle_,
      input[0].height,
      input[0].width,
      FORMAT_BGR_PLANAR, // force to this format
      input[0].data_type
    );
    output.allocate();
  }

  int ret = bmcv_image_convert_to(handle_.data(), N, attr, input.data(), output.data());

  return ret;
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
  if (!output.is_created()) {
    output.create(
      handle_,
      input[0].height,
      input[0].width,
      FORMAT_BGR_PLANAR, // force to this format
      input[0].data_type
    );
    output.allocate();
  }

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
  if (!output.is_created()) {
    /* vpp limitation: 64-aligned */
    int stride = ((input.data().width + (64 - 1)) >> 6) << 6; // ceiling to 64 * N

    output.create(
      handle_,
      input[0].height,
      input[0].width,
      FORMAT_BGR_PLANAR, // force to this format
      input[0].data_type,
      &stride
    );
    output.allocate();
  }

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
  if (!output.is_created()) {
    output.create(
      handle_,
      input[0].height,
      input[0].width,
      FORMAT_BGR_PLANAR, // force to this format
      input[0].data_type
    );
    output.allocate();
  }

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

#endif

}  // namespace sail
