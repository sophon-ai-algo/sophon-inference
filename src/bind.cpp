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
           v[i] = t;
         }
    );
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
     .def("vpp_crop_and_resize", (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int, int, int, int, int)) &Bmcv::vpp_crop_and_resize)
     .def("vpp_crop",            (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int, int, int))           &Bmcv::vpp_crop)
     .def("vpp_resize",          (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, int, int))                     &Bmcv::vpp_resize)
     .def("yuv2bgr",             (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&))                               &Bmcv::yuv2bgr)
     .def("warp",                (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, const std::array<std::pair<std::tuple<float, float, float>, std::tuple<float, float, float>>, N>&))              &Bmcv::warp)
     .def("convert_to",          (BMImageArray<N> (Bmcv::*)(BMImageArray<N>&, const std::tuple<std::pair<float, float>, std::pair<float, float>, std::pair<float, float>>&))                   &Bmcv::convert_to)
     .def("convert_to",          (int             (Bmcv::*)(BMImageArray<N>&, BMImageArray<N>&, const std::tuple<std::pair<float, float>, std::pair<float, float>, std::pair<float, float>>&)) &Bmcv::convert_to);
}
#endif

PYBIND11_MODULE(sail, m) {
  using namespace pybind11::literals;
  m.attr("__version__") = "2.0.3";
  m.doc() = "sophon inference module";

  m.def("get_available_tpu_num", &get_available_tpu_num);
  m.def("_dryrun", &model_dryrun);

  py::enum_<bm_data_type_t>(m, "Dtype")
    .value("BM_FLOAT32", bm_data_type_t::BM_FLOAT32)
    .value("BM_INT8", bm_data_type_t::BM_INT8)
    .value("BM_UINT8", bm_data_type_t::BM_UINT8)
    .export_values();

  py::enum_<IOMode>(m, "IOMode")
    .value("SYSI", IOMode::SYSI)
    .value("SYSO", IOMode::SYSO)
    .value("SYSIO", IOMode::SYSIO)
    .value("DEVIO", IOMode::DEVIO)
    .export_values();

  py::class_<Handle>(m, "Handle")
    .def(py::init<int>());

  py::class_<Tensor>(m, "Tensor")
    .def(py::init<Handle, const std::vector<int>&, bm_data_type_t, bool, bool>())
    .def(py::init<Handle, py::array_t<float>&>())
    .def(py::init<Handle, py::array_t<int8_t>&>())
    .def(py::init<Handle, py::array_t<uint8_t>&>())
    .def("shape",                 &Tensor::shape)
    .def("reshape",               &Tensor::reshape)
    .def("own_sys_data",          &Tensor::own_sys_data)
    .def("own_dev_data",          &Tensor::own_dev_data)
    .def("asnumpy",               (py::object (Tensor::*)()) &Tensor::asnumpy)
    .def("asnumpy",               (py::object (Tensor::*)(const std::vector<int>&)) &Tensor::asnumpy)
    .def("update_data",           (void (Tensor::*)(pybind11::array_t<float>&)) &Tensor::update_data)
    .def("update_data",           (void (Tensor::*)(pybind11::array_t<int8_t>&)) &Tensor::update_data)
    .def("update_data",           (void (Tensor::*)(pybind11::array_t<uint8_t>&)) &Tensor::update_data)
    .def("scale_from",            &Tensor::scale_from)
    .def("scale_to",              (py::array_t<float> (Tensor::*)(float)) &Tensor::scale_to)
    .def("scale_to",              (py::array_t<float> (Tensor::*)(float, const std::vector<int>&)) &Tensor::scale_to)
    .def("sync_s2d",              (void (Tensor::*)()) &Tensor::sync_s2d, "move all data from system to device")
    .def("sync_s2d",              (void (Tensor::*)(int)) &Tensor::sync_s2d, "move size data from system to device")
    .def("sync_d2s",              (void (Tensor::*)()) &Tensor::sync_d2s, "move all data from device to system")
    .def("sync_d2s",              (void (Tensor::*)(int)) &Tensor::sync_d2s, "move size data from device to system");

  py::class_<Engine>(m, "Engine")
    .def(py::init<int>())
    .def(py::init<const std::string&, int, IOMode>())
    .def(py::init<py::bytes&, int, int, IOMode>())
    .def("load", (bool (Engine::*)(const std::string&))&Engine::load)
    .def("load", (bool (Engine::*)(py::bytes&, int))&Engine::load)
    .def("get_handle",            (Handle& (Engine::*)())&Engine::get_handle)
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
    .def("read_",                (bm_image (Decoder::*)(Handle&))            &Decoder::read_)
    .def("read_",                (int      (Decoder::*)(Handle&, bm_image&)) &Decoder::read_);
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

  /* cannot be instantiated in python, the only use case is: BMImageArray[i] = Decoder.read_() */
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
    .def("dtype",                &BMImage::dtype);

  declareBMImageArray<2>(m); // BMImageArray2D
  declareBMImageArray<3>(m); // BMImageArray3D
  declareBMImageArray<4>(m); // BMImageArray4D

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
    .def("yuv2bgr",             (BMImage (Bmcv::*)(BMImage&))                               &Bmcv::yuv2bgr)
    .def("warp",                (BMImage (Bmcv::*)(BMImage&, const std::pair<std::tuple<float, float, float>, std::tuple<float, float, float>>&))                     &Bmcv::warp)
    .def("convert_to",          (BMImage (Bmcv::*)(BMImage&, const std::tuple<std::pair<float, float>, std::pair<float, float>, std::pair<float, float>>&))           &Bmcv::convert_to)
    .def("convert_to",          (int     (Bmcv::*)(BMImage&, BMImage&, const std::tuple<std::pair<float, float>, std::pair<float, float>, std::pair<float, float>>&)) &Bmcv::convert_to)
    .def("rectangle",           &Bmcv::rectangle)
    .def("imwrite",             &Bmcv::imwrite)
    .def("get_handle",          &Bmcv::get_handle)
    .def("get_bm_data_type",         (bm_data_type_t (Bmcv::*)(bm_image_data_format_ext)) &Bmcv::get_bm_data_type)
    .def("get_bm_image_data_format", (bm_image_data_format_ext (Bmcv::*)(bm_data_type_t)) &Bmcv::get_bm_image_data_format);

  registerBMImageArrayFunctions<2>(cls);
  registerBMImageArrayFunctions<3>(cls);
  registerBMImageArrayFunctions<4>(cls);

#endif
}
