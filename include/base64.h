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

/** @file     base64.h
 *  @brief    base64 encoder and decoder
 *  @author   bitmain
 *  @version  2.3.2
 *  @date     2021-04-06
 */

#ifndef SAIL_BASE64_H
#define SAIL_BASE64_H

#include <iostream>
#include <string>
#include "tensor.h"

#ifdef USE_BMCV
#include <bmlib_runtime.h>
#include <bmcv_api.h>
#include <bmcv_api_ext.h>
#endif

namespace sail {
    int base64_enc(Handle& handle, const void *data, uint32_t dlen, std::string& encoded);
    int base64_dec(Handle& handle, const void *data, uint32_t dlen, uint8_t* p_outbuf, uint32_t *p_size);
}


#endif //!SAIL_BASE64_H