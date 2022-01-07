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

#include "frame_provider.h"

FFMpegFrameProvider::FFMpegFrameProvider(
    sail::Bmcv&   bmcv,
    const string& filename,
    int           tpu_id)
  : bmcv_(bmcv), hdl_(bmcv.get_handle()), decoder_(filename, false, tpu_id) {
}

int FFMpegFrameProvider::get(sail::BMImage& input) {
  int ret = decoder_.read(hdl_, input);
  return ret;
}

PreProcessorBmcv::PreProcessorBmcv(
    sail::Bmcv& bmcv,
    float       scale,
    int         height,
    int         width)
    : bmcv_(bmcv),
    height_(height), width_(width) {
  
    ab_[0] = 0.003922;
    ab_[1] = 0.0;
    ab_[2] = 0.003922;
    ab_[3] = 0.0;
    ab_[4] = 0.003922;
    ab_[5] = 0.0;

  for (int i = 0; i < 6; i++) {
    ab_[i] *= scale;
  }
}

void PreProcessorBmcv::process(sail::BMImage& input, sail::BMImage& output) {
  sail::Handle handle;
  handle = bmcv_.get_handle();
  sail::BMImage imgtemp(handle, height_, width_,FORMAT_RGB_PLANAR, input.dtype());

  // resize: bgr-packed -> bgr-planar
  bmcv_.vpp_resize(input, imgtemp, width_, height_);
  // linear: bgr-planar -> rgb-planar
  bmcv_.convert_to(imgtemp,
                   output,
                   std::make_tuple(
                     std::make_pair(ab_[0], ab_[1]),
                     std::make_pair(ab_[2], ab_[3]),
                     std::make_pair(ab_[4], ab_[5])
                     ));
}
