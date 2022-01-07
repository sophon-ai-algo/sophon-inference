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

#pragma once
#include <string>
#include "cvwrapper.h"

using namespace std;

class FFMpegFrameProvider {
public:
  FFMpegFrameProvider(sail::Bmcv &bmcv, const string &filename, int tpu_id);
  int get(sail::BMImage& input);
protected:
  sail::Bmcv &bmcv_;
  sail::Handle hdl_;
  sail::Decoder decoder_;
};

/**
 * @brief The preprocess class with bmcv
 */
class PreProcessorBmcv {
 public:
  PreProcessorBmcv(sail::Bmcv& bmcv, float scale, int height, int width);
  void process(sail::BMImage& input, sail::BMImage& output);
 private:
  sail::Bmcv& bmcv_;
  float ab_[6];
  int height_;
  int width_;
};
