/* Copyright 2016-2022 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.  */

#include <fstream>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include "tensor.h"
#include "engine.h"
#include "cvwrapper.h"
#include "tools.h"
#include "base64.h"
#include "internal.h"
#include "engine_multi.h"

using namespace sail;
namespace py = pybind11;

#ifdef USE_BMCV
template<std::size_t N>
static void declareBMImageArray(py::module &m) {
  std::stringstream ss; ss << "BMImageArray" << N << "D";
  py::class_<BMImageArray<N>>(m, ss.str().c_str())
    .def(py::init<>())
    .def(py::init<Handle&, int, int, bm_image_format_ext, bm_image_data_format_ext>())
    
    .def("__len__", &BMImageArray<N>::size)
    .def("__getitem__",
         [](BMImageArray<N> &v, size_t i) -> bm_image & {
           if (i >= v.size()) throw py::index_error();
           return v[i];
         }, py::return_value_policy::reference_internal)
    .def("__setitem__",
         [](BMImageArray<N> &v, size_t i, const bm_image &t) {
           if (i >= v.size()) throw py::index_error();
           if (v.check_need_free()) {
               if (v[0].width != t.width || v[0].height != t.height || v[0].image_format != t.image_format ||
                   v[0].data_type != t.data_type) {
                   printf("ERROR:__setitem__:  requires src image's format is same as dst\n");
                   printf("src(w=%d,h=%d, format=%s, dtype=%s\n", t.width, t.height,
                           bm_image_format_desc(t.image_format),
                           bm_image_data_type_desc(t.data_type));
                   printf("dst(w=%d,h=%d, format=%s, dtype=%s\n", v[i].width, v[i].height,
                          bm_image_format_desc(v[i].image_format),
                          bm_image_data_type_desc(v[i].data_type));
                   throw py::value_error();
               }
               bm_handle_t  handle = bm_image_get_handle(&v[0]);
               bmcv_copy_to_atrr_t attr;
               memset(&attr, 0, sizeof(attr));
               int ret = bmcv_image_copy_to(handle, attr, t, v[i]);
               if (BM_SUCCESS != ret) {
                   SPDLOG_ERROR("bmcv_image_copy_to err={}", ret);
                   throw py::value_error();
               }
           }else{
               //printf("__setitem__:\n");
               //print_image(t, "src:");
               int stride[3]={0};
               bm_handle_t  handle = bm_image_get_handle((bm_image*)&t);
               if (handle != nullptr) {
                   bm_image_get_stride(t, stride);
                  for(int temp_idx = 0; temp_idx < v.size(); ++temp_idx){
                    int ret = bm_image_create(handle, t.height, t.width, t.image_format, 
                      t.data_type, &v[temp_idx], stride);
                    if (BM_SUCCESS != ret) {
                      SPDLOG_ERROR("bm_image_create err={}", ret);
                      throw py::value_error();
                    }
                    // ret = bm_image_alloc_dev_mem_heap_mask(v[temp_idx], 6);
                    // if (BM_SUCCESS != ret) {
                    //   SPDLOG_ERROR("bm_image_alloc_dev_mem_heap_mask err={}", ret);
                    //   throw py::value_error();
                    // }
                  }
                  bm_image_alloc_contiguous_mem(N, v.data());
                  //  bm_device_mem_t dev_mem[3];
                  //  bm_image_get_device_mem(t, dev_mem);
                  //  bm_image_attach(v[i], dev_mem);
                  bmcv_copy_to_atrr_t attr;
                  memset(&attr, 0, sizeof(attr));
                  int ret = bmcv_image_copy_to(handle, attr, t, v[0]);
                  if (BM_SUCCESS != ret) {
                    SPDLOG_ERROR("bmcv_image_copy_to err={}", ret);
                    throw py::value_error();
                  }    
                  v.set_need_free(true);
               }else{
                   SPDLOG_ERROR("src image handle=nullptr");
                   throw py::value_error();
               }
           }
         }
    )
    .def("check_need_free", (bool (BMImageArray<N>::*) ()) &BMImageArray<N>::check_need_free)
    .def("set_need_free", (void (BMImageArray<N>::*) (bool))   &BMImageArray<N>::set_need_free)
    .def("create", (void (BMImageArray<N>::*)(Handle&, int, int, bm_image_format_ext, bm_image_data_format_ext)) &BMImageArray<N>::create)
    .def("copy_from",        &BMImageArray<N>::copy_from)
    .def("attach_from",      &BMImageArray<N>::attach_from);
}

template<std::size_t N>
static void registerBMImageArrayFunctions(py::class_<Bmcv> &cls) {
  cls.def("bm_image_to_tensor",  (void            (Bmcv::*)(BMImageArray<N>&, Tensor&))       &Bmcv::bm_image_to_tensor)
     .def("bm_image_to_tensor",  (Tensor          (Bmcv::*)(BMImageArray<N>&))                &Bmcv::bm_image_to_tensor)
     .def("tensor_to_bm_image",  (void            (Bmcv::*)(Tensor&, BMImageArray<N>&, bool)) &Bmcv::tensor_to_bm_image, py::arg("tensor"), py::arg("img"), py::arg("bgr2rgb")=false)
     .def("tensor_to_bm_image",  (BMImageArray<N> (Bmcv::*)(Tensor&, bool))                   &Bmcv::tensor_to_bm_image, py::arg("tensor"), py::arg("bgr2rgb")=false)
     .def("crop_and_resize",     (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int, int, int, int, int)) &Bmcv::crop_and_resize)
     .def("crop",                (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int, int, int))           &Bmcv::crop)
     .def("resize",              (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int))                     &Bmcv::resize)
     .def("vpp_resize",          (int             (Bmcv::*)(BMImageArray<N>&, BMImageArray<N>&, int, int))   &Bmcv::vpp_resize)
     .def("vpp_crop_and_resize", (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int, int, int, int, int)) &Bmcv::vpp_crop_and_resize)
     .def("vpp_crop",            (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int, int, int))           &Bmcv::vpp_crop)
     .def("vpp_resize",          (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int))                     &Bmcv::vpp_resize)
     .def("vpp_crop_and_resize_padding", (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int, int, int, int, int,PaddingAtrr&)) &Bmcv::vpp_crop_and_resize_padding)
     .def("vpp_crop_padding",    (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int, int, int, PaddingAtrr&))  &Bmcv::vpp_crop_padding)
     .def("vpp_resize_padding",  (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int, PaddingAtrr&))  &Bmcv::vpp_resize_padding)
     .def("yuv2bgr",             (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&))                               &Bmcv::yuv2bgr)
     .def("warp",                (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, const std::array<std::pair<std::tuple<float, float, float>, std::tuple<float, float, float>>, N>&))              &Bmcv::warp)
     .def("convert_to",          (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, const std::tuple<std::pair<float, float>, std::pair<float, float>, std::pair<float, float>>&))                   &Bmcv::convert_to)
     .def("convert_to",          (int             (Bmcv::*)(BMImageArray<N>&, BMImageArray<N>&, const std::tuple<std::pair<float, float>, std::pair<float, float>, std::pair<float, float>>&)) &Bmcv::convert_to)
     .def("image_copy_to",          (int          (Bmcv::*)(BMImageArray<N>&, BMImageArray<N>&, int, int))  &Bmcv::image_copy_to,
        py::arg("input"),py::arg("output"), py::arg("start_x")=0, py::arg("start_y")=0)
     .def("image_copy_to_padding",  (int          (Bmcv::*)(BMImageArray<N>&, BMImageArray<N>&, unsigned int, unsigned int, unsigned int, int, int))  &Bmcv::image_copy_to_padding,
        py::arg("input"),py::arg("output"), py::arg("padding_r"), py::arg("padding_g"), py::arg("padding_b"),py::arg("start_x")=0, py::arg("start_y")=0);
}
#endif

PYBIND11_MODULE(sail, m) {

  char temp[64] = {0};
  get_sail_version(temp);
  using namespace pybind11::literals;
  m.attr("__version__") = temp;
  m.doc() = "sophon inference module";

  m.def("get_available_tpu_num", &get_available_tpu_num);
  m.def("set_print_flag", &set_print_flag);
  m.def("set_dump_io_flag", &set_dump_io_flag);
  m.def("set_decoder_env",&set_decoder_env);
  m.def("_dryrun", &model_dryrun);
  m.def("_perf",   &multi_tpu_perf);

  py::enum_<bm_data_type_t>(m, "Dtype")
    .value("BM_FLOAT32", bm_data_type_t::BM_FLOAT32)
    .value("BM_INT8", bm_data_type_t::BM_INT8)
    .value("BM_UINT8", bm_data_type_t::BM_UINT8)
    .value("BM_INT32", bm_data_type_t::BM_INT32)
    .value("BM_UINT32", bm_data_type_t::BM_UINT32)
    //.value("BM_INT64", bm_data_type_t::BM_INT64)
    .export_values();

  py::enum_<IOMode>(m, "IOMode")
    .value("SYSI", IOMode::SYSI)
    .value("SYSO", IOMode::SYSO)
    .value("SYSIO", IOMode::SYSIO)
    .value("DEVIO", IOMode::DEVIO)
    .export_values();

  py::class_<Handle>(m, "Handle")
    .def(py::init<int>())
    .def("get_device_id", &Handle::get_device_id)
    .def("get_sn",        &Handle::get_sn);

  py::class_<Tensor>(m, "Tensor")
    .def(py::init<Handle, const std::vector<int>&, bm_data_type_t, bool, bool>(),
      py::arg("handle"),py::arg("shape"),py::arg("dtype")=BM_FLOAT32,py::arg("own_sys_data")=false,py::arg("own_dev_data")=false)
    .def(py::init<Handle, py::array_t<float>&>())
    .def(py::init<Handle, py::array_t<int8_t>&>())
    .def(py::init<Handle, py::array_t<uint8_t>&>())
    .def(py::init<Handle, py::array_t<int32_t>&>())
    .def(py::init<Handle, py::array_t<float>&, bool>())
    .def(py::init<Handle, py::array_t<int8_t>&, bool>())
    .def(py::init<Handle, py::array_t<uint8_t>&, bool>())
    .def(py::init<Handle, py::array_t<int32_t>&, bool>())
    .def("shape",                 &Tensor::shape)
    .def("reshape",               &Tensor::reshape)
    .def("own_sys_data",          &Tensor::own_sys_data)
    .def("own_dev_data",          &Tensor::own_dev_data)
    .def("asnumpy",               (py::object (Tensor::*)()) &Tensor::asnumpy)
    .def("pysys_data",            (py::array_t<long> (Tensor::*)()) &Tensor::pysys_data)
    .def("asnumpy",               (py::object (Tensor::*)(const std::vector<int>&)) &Tensor::asnumpy)
    .def("update_data",           (void (Tensor::*)(pybind11::array_t<float>&)) &Tensor::update_data)
    .def("update_data",           (void (Tensor::*)(pybind11::array_t<int8_t>&)) &Tensor::update_data)
    .def("update_data",           (void (Tensor::*)(pybind11::array_t<uint8_t>&)) &Tensor::update_data)
    .def("update_data",           (void (Tensor::*)(pybind11::array_t<int32_t>&))   &Tensor::update_data)
    .def("scale_from",            (void (Tensor::*)(pybind11::array_t<float>&, float)) &Tensor::scale_from)
    //.def("scale_from",            (void (Tensor::*)(pybind11::array_t<int32_t>&, float)) &Tensor::scale_from)
    .def("scale_to",              (py::array_t<float> (Tensor::*)(float)) &Tensor::scale_to)
    .def("scale_to",              (py::array_t<float> (Tensor::*)(float, const std::vector<int>&)) &Tensor::scale_to)
    //.def("scale_to",              (py::array_t<int32_t> (Tensor::*)(float)) &Tensor::scale_to)
    //.def("scale_to",              (py::array_t<int32_t> (Tensor::*)(float, const std::vector<int>&)) &Tensor::scale_to)
    .def("sync_s2d",              (void (Tensor::*)()) &Tensor::sync_s2d, "move all data from system to device")
    .def("sync_s2d",              (void (Tensor::*)(int)) &Tensor::sync_s2d, "move size data from system to device")
    .def("sync_d2s",              (void (Tensor::*)()) &Tensor::sync_d2s, "move all data from device to system")
    .def("sync_d2s",              (void (Tensor::*)(int)) &Tensor::sync_d2s, "move size data from device to system");

  py::class_<Engine>(m, "Engine")
    .def(py::init<int>())
    .def(py::init<const Handle&>())
    .def(py::init<const std::string&, int, IOMode>())
    .def(py::init<py::bytes&, int, int, IOMode>())
    .def("load", (bool (Engine::*)(const std::string&))&Engine::load)
    .def("load", (bool (Engine::*)(py::bytes&, int))&Engine::load)
    .def("get_handle",            (Handle& (Engine::*)())&Engine::get_handle)
    .def("get_device_id",         &Engine::get_device_id)
    .def("get_graph_names",       &Engine::get_graph_names)
    .def("set_io_mode",           &Engine::set_io_mode)
    .def("get_input_names",       &Engine::get_input_names)
    .def("get_output_names",      &Engine::get_output_names)
    .def("get_max_input_shapes",  &Engine::get_max_input_shapes)
    .def("get_input_shape",       &Engine::get_input_shape)
    .def("get_max_output_shapes", &Engine::get_max_output_shapes)
    .def("get_output_shape",      &Engine::get_output_shape)
    .def("get_input_dtype",       &Engine::get_input_dtype)
    .def("get_output_dtype",      &Engine::get_output_dtype)
    .def("get_input_scale",       &Engine::get_input_scale)
    .def("get_output_scale",      &Engine::get_output_scale)
    .def("create_input_tensors_map",  (std::map<std::string, Tensor&> (Engine::*) (const std::string&, int)) &Engine::create_input_tensors_map,
      py::arg("graph_name"), py::arg("create_mode")=-1)
    .def("create_output_tensors_map", (std::map<std::string, Tensor&> (Engine::*) (const std::string&, int)) &Engine::create_output_tensors_map,
      py::arg("graph_name"), py::arg("create_mode")=-1)
    .def("process", (void (Engine::*)(const std::string&, std::map<std::string, Tensor&>&, std::map<std::string, Tensor&>&)) &Engine::process)
    .def("process", (void (Engine::*)(const std::string&, std::map<std::string, Tensor&>&, std::map<std::string, std::vector<int>>&, std::map<std::string, Tensor&>&)) &Engine::process)
    .def("process", (std::map<std::string, pybind11::array_t<float>> (Engine::*)(const std::string&, std::map<std::string, pybind11::array_t<float>>&)) &Engine::process);

#ifdef USE_FFMPEG
  py::class_<Frame>(m, "Frame")
    .def(py::init<>())
    .def("get_height",           &Frame::get_height)
    .def("get_width",            &Frame::get_width);

  py::class_<Decoder>(m, "Decoder")
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, bool>())
    .def(py::init<const std::string&, bool, int>())
    .def("is_opened",            &Decoder::is_opened)
    .def("get_frame_shape",      &Decoder::get_frame_shape)
    .def("read",                 (BMImage  (Decoder::*)(Handle&))            &Decoder::read)
    .def("read",                 (int      (Decoder::*)(Handle&, BMImage&))  &Decoder::read)
    // .def("read_",                (bm_image (Decoder::*)(Handle&))            &Decoder::read_)
    .def("read_",                (int      (Decoder::*)(Handle&, bm_image&)) &Decoder::read_)
    .def("get_fps",              (float      (Decoder::*)() const)             &Decoder::get_fps)
    .def("release",              &Decoder::release)
    .def("reconnect",            &Decoder::reconnect);
#endif

#ifdef USE_BMCV
  py::enum_<bm_image_format_ext>(m, "Format")
    .value("FORMAT_YUV420P",       bm_image_format_ext::FORMAT_YUV420P)
    .value("FORMAT_YUV422P",       bm_image_format_ext::FORMAT_YUV422P)
    .value("FORMAT_YUV444P",       bm_image_format_ext::FORMAT_YUV444P)
    .value("FORMAT_NV12",          bm_image_format_ext::FORMAT_NV12)
    .value("FORMAT_NV21",          bm_image_format_ext::FORMAT_NV21)
    .value("FORMAT_NV16",          bm_image_format_ext::FORMAT_NV16)
    .value("FORMAT_NV61",          bm_image_format_ext::FORMAT_NV61)
    .value("FORMAT_NV24",          bm_image_format_ext::FORMAT_NV24)
    .value("FORMAT_RGB_PLANAR",    bm_image_format_ext::FORMAT_RGB_PLANAR)
    .value("FORMAT_BGR_PLANAR",    bm_image_format_ext::FORMAT_BGR_PLANAR)
    .value("FORMAT_RGB_PACKED",    bm_image_format_ext::FORMAT_RGB_PACKED)
    .value("FORMAT_BGR_PACKED",    bm_image_format_ext::FORMAT_BGR_PACKED)
    .value("FORMAT_RGBP_SEPARATE", bm_image_format_ext::FORMAT_RGBP_SEPARATE)
    .value("FORMAT_BGRP_SEPARATE", bm_image_format_ext::FORMAT_BGRP_SEPARATE)
    .value("FORMAT_GRAY",          bm_image_format_ext::FORMAT_GRAY)
    .value("FORMAT_COMPRESSED",    bm_image_format_ext::FORMAT_COMPRESSED)
    .export_values();

  py::enum_<bm_image_data_format_ext>(m, "ImgDtype")
    .value("DATA_TYPE_EXT_FLOAT32",        bm_image_data_format_ext::DATA_TYPE_EXT_FLOAT32)
    .value("DATA_TYPE_EXT_1N_BYTE",        bm_image_data_format_ext::DATA_TYPE_EXT_1N_BYTE)
    .value("DATA_TYPE_EXT_4N_BYTE",        bm_image_data_format_ext::DATA_TYPE_EXT_4N_BYTE)
    .value("DATA_TYPE_EXT_1N_BYTE_SIGNED", bm_image_data_format_ext::DATA_TYPE_EXT_1N_BYTE_SIGNED)
    .value("DATA_TYPE_EXT_4N_BYTE_SIGNED", bm_image_data_format_ext::DATA_TYPE_EXT_4N_BYTE_SIGNED)
    .export_values();

  /* cannot be instantiated in python, the only use case is: Decoder.read_(handle, BMImageArray[i]) */
  py::class_<bm_image>(m, "bm_image")
    .def("width",                [](bm_image &img) -> int { return img.width;  })
    .def("height",               [](bm_image &img) -> int { return img.height; })
    .def("format",               [](bm_image &img) -> bm_image_format_ext { return img.image_format; })
    .def("dtype",                [](bm_image &img) -> bm_image_data_format_ext { return img.data_type; });
    
  py::class_<BMImage>(m, "BMImage")
    .def(py::init<>())
    .def(py::init<bm_image&>())
    .def(py::init<Handle&, int, int, bm_image_format_ext, bm_image_data_format_ext>())
    .def("width",                &BMImage::width)
    .def("height",               &BMImage::height)
    .def("format",               &BMImage::format)
    .def("dtype",                &BMImage::dtype)
    .def("data",                 (bm_image (BMImage::*)() const)      &BMImage::data )
    .def("get_plane_num",        (int (BMImage::*)() const)           &BMImage::get_plane_num)
    .def("need_to_free",         (bool (BMImage::*)() const)          &BMImage::need_to_free)
    .def("empty_check",          (int (BMImage::*)() const)           &BMImage::empty_check);
    
  declareBMImageArray<2>(m); // BMImageArray2D
  declareBMImageArray<3>(m); // BMImageArray3D
  declareBMImageArray<4>(m); // BMImageArray4D
  declareBMImageArray<8>(m); // BMImageArray8D
  declareBMImageArray<16>(m); // BMImageArray16D
  declareBMImageArray<32>(m); // BMImageArray32D
  declareBMImageArray<64>(m); // BMImageArray64D
  declareBMImageArray<128>(m); // BMImageArray128D
  declareBMImageArray<256>(m); // BMImageArray256D

  py::class_<PaddingAtrr>(m, "PaddingAtrr")
    .def(py::init<>())
    .def(py::init<unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int>())
    .def("set_stx",              (void (PaddingAtrr::*)(unsigned int))   &PaddingAtrr::set_stx)
    .def("set_sty",              (void (PaddingAtrr::*)(unsigned int))   &PaddingAtrr::set_sty)
    .def("set_w",                (void (PaddingAtrr::*)(unsigned int))   &PaddingAtrr::set_w)
    .def("set_h",                (void (PaddingAtrr::*)(unsigned int))   &PaddingAtrr::set_h)
    .def("set_r",                (void (PaddingAtrr::*)(unsigned int))   &PaddingAtrr::set_r)
    .def("set_g",                (void (PaddingAtrr::*)(unsigned int))   &PaddingAtrr::set_g)
    .def("set_b",                (void (PaddingAtrr::*)(unsigned int))   &PaddingAtrr::set_b);

  auto cls = py::class_<Bmcv>(m, "Bmcv")
    .def(py::init<Handle&>())
    .def("bm_image_to_tensor",  (void    (Bmcv::*)(BMImage&, Tensor&))       &Bmcv::bm_image_to_tensor)
    .def("bm_image_to_tensor",  (Tensor  (Bmcv::*)(BMImage&))                &Bmcv::bm_image_to_tensor)
    .def("tensor_to_bm_image",  (void    (Bmcv::*)(Tensor&, BMImage&, bool)) &Bmcv::tensor_to_bm_image, py::arg("tensor"), py::arg("img"), py::arg("bgr2rgb")=false)
    .def("tensor_to_bm_image",  (BMImage (Bmcv::*)(Tensor&, bool))           &Bmcv::tensor_to_bm_image, py::arg("tensor"), py::arg("bgr2rgb")=false)
    .def("crop_and_resize",     (BMImage (Bmcv::*)(BMImage&, int, int, int, int, int, int)) &Bmcv::crop_and_resize)
    .def("crop",                (BMImage (Bmcv::*)(BMImage&, int, int, int, int))           &Bmcv::crop)
    .def("resize",              (BMImage (Bmcv::*)(BMImage&, int, int))                     &Bmcv::resize)
    .def("vpp_crop_and_resize", (BMImage (Bmcv::*)(BMImage&, int, int, int, int, int, int)) &Bmcv::vpp_crop_and_resize)
    .def("vpp_crop",            (BMImage (Bmcv::*)(BMImage&, int, int, int, int))           &Bmcv::vpp_crop)
    .def("vpp_resize",          (BMImage (Bmcv::*)(BMImage&, int, int))                     &Bmcv::vpp_resize)
    .def("vpp_resize",          (int     (Bmcv::*)(BMImage&, BMImage&, int, int))           &Bmcv::vpp_resize)
    .def("vpp_crop_and_resize_padding", (BMImage (Bmcv::*)(BMImage&, int, int, int, int, int, int,PaddingAtrr&)) &Bmcv::vpp_crop_and_resize_padding)
    .def("vpp_crop_padding",    (BMImage (Bmcv::*)(BMImage&, int, int, int, int, PaddingAtrr&))   &Bmcv::vpp_crop_padding)
    .def("vpp_resize_padding",  (BMImage (Bmcv::*)(BMImage&, int, int, PaddingAtrr&))   &Bmcv::vpp_resize_padding)
    .def("yuv2bgr",             (BMImage (Bmcv::*)(BMImage&))                               &Bmcv::yuv2bgr)
    .def("warp",                (BMImage (Bmcv::*)(BMImage&, const std::pair<std::tuple<float, float, float>, std::tuple<float, float, float>>&))                     &Bmcv::warp)
    .def("warp_perspective",    (BMImage (Bmcv::*)(BMImage&, const std::tuple<std::pair<int, int>,std::pair<int, int>,std::pair<int, int>,std::pair<int, int>>&, int, int, bm_image_format_ext, bm_image_data_format_ext, int))  &Bmcv::warp_perspective,
      py::arg("input"),py::arg("coordinate"),py::arg("output_width"),py::arg("output_height"),py::arg("format")=FORMAT_BGR_PLANAR,py::arg("dtype")=DATA_TYPE_EXT_1N_BYTE,py::arg("use_bilinear")=0)
    .def("convert_to",          (BMImage (Bmcv::*)(BMImage&, const std::tuple<std::pair<float, float>, std::pair<float, float>, std::pair<float, float>>&))           &Bmcv::convert_to)
    .def("convert_to",          (int     (Bmcv::*)(BMImage&, BMImage&, const std::tuple<std::pair<float, float>, std::pair<float, float>, std::pair<float, float>>&)) &Bmcv::convert_to)
    .def("rectangle",           &Bmcv::rectangle)
    .def("rectangle_",          &Bmcv::rectangle_)
    .def("putText",             &Bmcv::putText)
    .def("putText_",            &Bmcv::putText_)
    .def("imwrite",             &Bmcv::imwrite)
    .def("imwrite_",            &Bmcv::imwrite_)
    .def("get_handle",          &Bmcv::get_handle)
    .def("get_bm_data_type",         (bm_data_type_t (Bmcv::*)(bm_image_data_format_ext)) &Bmcv::get_bm_data_type)
    .def("get_bm_image_data_format", (bm_image_data_format_ext (Bmcv::*)(bm_data_type_t)) &Bmcv::get_bm_image_data_format)
    .def("vpp_convert_format",          (BMImage (Bmcv::*)(BMImage&))                     &Bmcv::vpp_convert_format)
    .def("vpp_convert_format",          (int     (Bmcv::*)(BMImage&, BMImage&))           &Bmcv::vpp_convert_format)
    .def("convert_format",              (BMImage (Bmcv::*)(BMImage&))                     &Bmcv::convert_format)
    .def("convert_format",              (int     (Bmcv::*)(BMImage&, BMImage&))           &Bmcv::convert_format)
    .def("crop_and_resize_padding",     (BMImage (Bmcv::*)(BMImage&, int, int, int, int, int, int, PaddingAtrr&))   &Bmcv::crop_and_resize_padding)
    .def("image_add_weighted",          (int     (Bmcv::*)(BMImage&, float, BMImage&, float, float, BMImage&))      &Bmcv::image_add_weighted)
    .def("image_add_weighted",          (int     (Bmcv::*)(BMImage&, float, BMImage&, float, float, BMImage&))      &Bmcv::image_add_weighted)
    .def("image_copy_to",               (int     (Bmcv::*)(BMImage&, BMImage&, int, int))  &Bmcv::image_copy_to,
      py::arg("input"),py::arg("output"), py::arg("start_x")=0, py::arg("start_y")=0)
    .def("image_copy_to_padding",       (int     (Bmcv::*)(BMImage&, BMImage&, unsigned int, unsigned int, unsigned int, int, int))  &Bmcv::image_copy_to_padding,
      py::arg("input"),py::arg("output"), py::arg("padding_r"), py::arg("padding_g"), py::arg("padding_b"),py::arg("start_x")=0, py::arg("start_y")=0)
    .def("nms",  (pybind11::array_t<float> (Bmcv::*)(pybind11::array_t<float>, float)) &Bmcv::nms);

  registerBMImageArrayFunctions<2>(cls);
  registerBMImageArrayFunctions<3>(cls);
  registerBMImageArrayFunctions<4>(cls);
  registerBMImageArrayFunctions<8>(cls);
  // registerBMImageArrayFunctions<16>(cls);
  // registerBMImageArrayFunctions<32>(cls);
  // registerBMImageArrayFunctions<64>(cls);
  // registerBMImageArrayFunctions<128>(cls);
  // registerBMImageArrayFunctions<256>(cls);

  m.def("base64_encode", [](Handle& handle, py::bytes a) {
         std::string str1 = (std::string)a;
         std::string str2;
         base64_enc(handle, str1.data(), str1.size(), str2);
         return py::bytes(str2);
      }, "Bitmain base64 encoder");

  m.def("base64_decode", [](Handle& handle, py::bytes s) {
      std::string input = (std::string)s;
      uint32_t input_size = input.size();
      uint32_t output_size = input_size/4*3;
      uint32_t real_size = 0;
      uint8_t* out_data = new uint8_t[output_size];
      base64_dec(handle, input.data(), input_size, out_data, &real_size);
      auto ob= pybind11::bytes((char*)out_data, real_size);
      delete [] out_data;
      return ob;
      }, "Bitmain base64 decoder");

    m.def("base64_encode_array", [](Handle& handle, py::array a) {
        std::string str2;
        base64_enc(handle, a.data(), a.size(), str2);
        return py::bytes(str2);
    }, "Bitmain base64 encoder");

    m.def("base64_decode_asarray", [](Handle& handle, py::bytes s) {
        std::string input = (std::string)s;
        uint32_t input_size = input.size();
        uint32_t output_size = input_size/4*3;
        uint8_t *buf = new uint8_t[output_size];
        uint32_t real_size = 0;
        base64_dec(handle, input.data(), input_size, buf, &real_size);
        assert(real_size <= output_size);
        std::vector<ssize_t> shape;
        shape.push_back(real_size);
        auto ndarray = pybind11::array_t<uint8_t>(shape);
        auto data = ndarray.mutable_data();
        memcpy(data, buf, real_size);
        delete []buf;
        return ndarray;
    }, "Bitmain base64 decoder");

#endif
    py::class_<MultiEngine>(m, "MultiEngine")
    .def(py::init<const std::string&, std::vector<int>, bool, int>(),
      py::arg("bmodel_path"),py::arg("tpu_ids"), py::arg("sys_out")=true, py::arg("graph_idx")=0)
    .def("get_device_ids",        &MultiEngine::get_device_ids)
    .def("get_graph_names",       &MultiEngine::get_graph_names)
    .def("get_input_names",       &MultiEngine::get_input_names)
    .def("get_output_names",      &MultiEngine::get_output_names)
    .def("get_input_shape",       &MultiEngine::get_input_shape)
    .def("get_output_shape",      &MultiEngine::get_output_shape)
    .def("set_print_flag",        &MultiEngine::set_print_flag)
    .def("set_print_time",        &MultiEngine::set_print_time)
    .def("process", (std::vector<std::map<std::string, Tensor*>> (MultiEngine::*)(std::vector<std::map<std::string, Tensor*>>&)) &MultiEngine::process)
    .def("process", (std::map<std::string, pybind11::array_t<float>> (MultiEngine::*)(std::map<std::string, pybind11::array_t<float>>&)) &MultiEngine::process);
}
