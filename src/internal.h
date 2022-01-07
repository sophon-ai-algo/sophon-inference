//
// Created by yuan on 6/15/21.
//

#ifndef SOPHON_INFERENCE_INTERNAL_H
#define SOPHON_INFERENCE_INTERNAL_H

#include "limits.h"
#include <sstream>
#include <numeric>
#include "bmlib_runtime.h"
#include "bmcv_api_ext.h"
// using simd intrinsic accelarate scale op
#if defined(__amd64__) || defined(__x86_64__)
#include <x86intrin.h>
#elif defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(__amd64__) || defined(__x86_64__)
#define USE_ASM_SSE 1
#elif defined(__arm__) || defined(__aarch64__)
#define USE_ASM_SSE 1
#endif



namespace sail {
#define SAIL_MIN(a, b)  (a) <= (b) ? (a):(b)
#define SAIL_MAX(a, b)  (a) >= (b) ? (a):(b)
#define SAIL_ALIGN 1

//#define USE_TRACE_POINT 1
#if USE_TRACE_POINT
#define TRACE_POINT printf("call %s() %s:%d\n", __FUNCTION__, __FILE__, __LINE__)
#else
#define TRACE_POINT void()
#endif

#define TRANSFER_INT(src, src_type, dst, dst_type, scale, size, dmax, dmin) for (int i = 0; i < size; ++i) { \
    src_type *src1 = (src_type*)src; \
    dst_type *dst1 = (dst_type*)dst; \
    auto v = std::round(src1[i] * scale); \
    dst1[i] = (dst_type)(SAIL_MIN(SAIL_MAX(v, dmin), dmax));\
}

#define TRANSFER_FLOAT(src, src_type, dst, dst_type, scale, size) for (int i = 0; i < size; ++i) { \
    src_type *src1 = (src_type*)src; \
    dst_type *dst1 = (dst_type*)dst; \
    dst1[i] = (dst_type) (src1[i] * scale);\
}


    static void AnyScale(void *src, bm_data_type_t src_dtype, void *dst, bm_data_type_t dst_dtype,  float scale, int size) {
        if (nullptr == src || nullptr == dst) {
            SPDLOG_ERROR("AnyScale param err, src={}, dst={}", src, dst);
            return;
        }
        if (src_dtype == BM_INT8) {
            if (dst_dtype == BM_INT8) {
                TRANSFER_INT(src, int8_t, dst, int8_t, scale, size, 127, -127)
            } else if (dst_dtype == BM_UINT8) {
                TRANSFER_INT(src, int8_t, dst, uint8_t, scale, size, UINT8_MAX, 0)
            } else if (dst_dtype == BM_UINT16) {
                TRANSFER_INT(src, int8_t, dst, uint16_t, scale, size, UINT16_MAX, 0)
            } else if (dst_dtype == BM_INT16) {
                TRANSFER_INT(src, int8_t, dst, int16_t, scale, size, INT16_MAX, INT16_MIN)
            } else if (dst_dtype == BM_UINT32) {
                TRANSFER_INT(src, int8_t, dst, uint32_t, scale, size, UINT32_MAX, (0U))
            } else if (dst_dtype == BM_INT32) {
                TRANSFER_INT(src, int8_t, dst, int32_t, scale, size, INT32_MAX, INT32_MIN)
            } else if (dst_dtype == BM_FLOAT32) {
                TRANSFER_FLOAT(src, int8_t, dst, float, scale, size)
            } else {
                printf("ERROR: don't support dtype=%d now!", dst_dtype);
                exit(0);
            }
        } else if (src_dtype == BM_UINT8) {
            if (dst_dtype == BM_INT8) {
                TRANSFER_INT(src, uint8_t, dst, int8_t, scale, size, 127, -127);
            } else if (dst_dtype == BM_UINT8) {
                TRANSFER_INT(src, uint8_t, dst, uint8_t, scale, size, UINT8_MAX, 0);
            } else if (dst_dtype == BM_UINT16) {
                TRANSFER_INT(src, uint8_t, dst, uint16_t, scale, size, UINT16_MAX, 0);
            } else if (dst_dtype == BM_INT16) {
                TRANSFER_INT(src, uint8_t, dst, int16_t, scale, size, INT16_MAX, INT16_MIN);
            } else if (dst_dtype == BM_UINT32) {
                TRANSFER_INT(src, uint8_t, dst, uint32_t, scale, size, UINT32_MAX, (0U));
            } else if (dst_dtype == BM_INT32) {
                TRANSFER_INT(src, uint8_t, dst, int32_t, scale, size, INT32_MAX, INT32_MIN);
            } else if (dst_dtype == BM_FLOAT32) {
                TRANSFER_FLOAT(src, uint8_t, dst, float, scale, size)
            } else {
                printf("ERROR: don't support dtype=%d now!", dst_dtype);
                exit(0);
            }
        } else if (src_dtype == BM_UINT16) {
            if (dst_dtype == BM_INT8) {
                TRANSFER_INT(src, uint16_t, dst, int8_t, scale, size, 127, -127);
            } else if (dst_dtype == BM_UINT8) {
                TRANSFER_INT(src, uint16_t, dst, uint8_t, scale, size, UINT8_MAX, 0);
            } else if (dst_dtype == BM_UINT16) {
                TRANSFER_INT(src, uint16_t, dst, uint16_t, scale, size, UINT16_MAX, 0);
            } else if (dst_dtype == BM_INT16) {
                TRANSFER_INT(src, uint16_t, dst, int16_t, scale, size, INT16_MAX, INT16_MIN);
            } else if (dst_dtype == BM_UINT32) {
                TRANSFER_INT(src, uint16_t, dst, uint32_t, scale, size, UINT32_MAX, (0U));
            } else if (dst_dtype == BM_INT32) {
                TRANSFER_INT(src, uint16_t, dst, int32_t, scale, size, INT32_MAX, INT32_MIN);
            } else if (dst_dtype == BM_FLOAT32) {
                TRANSFER_FLOAT(src, uint16_t, dst, float, scale, size)
            } else {
                printf("ERROR: don't support dtype=%d now!", dst_dtype);
                exit(0);
            }
        } else if (src_dtype == BM_INT16) {
            if (dst_dtype == BM_INT8) {
                TRANSFER_INT(src, int16_t, dst, int8_t, scale, size, 127, -127);
            } else if (dst_dtype == BM_UINT8) {
                TRANSFER_INT(src, int16_t, dst, uint8_t, scale, size, UINT8_MAX, 0);
            } else if (dst_dtype == BM_UINT16) {
                TRANSFER_INT(src, int16_t, dst, uint16_t, scale, size, UINT16_MAX, 0);
            } else if (dst_dtype == BM_INT16) {
                TRANSFER_INT(src, int16_t, dst, int16_t, scale, size, INT16_MAX, INT16_MIN);
            } else if (dst_dtype == BM_UINT32) {
                TRANSFER_INT(src, int16_t, dst, uint32_t, scale, size, UINT32_MAX, (0U));
            } else if (dst_dtype == BM_INT32) {
                TRANSFER_INT(src, int16_t, dst, int32_t, scale, size, INT32_MAX, INT32_MIN);
            } else if (dst_dtype == BM_FLOAT32) {
                TRANSFER_FLOAT(src, int16_t, dst, float, scale, size)
            } else {
                printf("ERROR: don't support dtype=%d now!", dst_dtype);
                exit(0);
            }
        } else if (src_dtype == BM_UINT32) {
            if (dst_dtype == BM_INT8) {
                TRANSFER_INT(src, uint32_t, dst, int8_t, scale, size, 127, -127);
            } else if (dst_dtype == BM_UINT8) {
                TRANSFER_INT(src, uint32_t, dst, uint8_t, scale, size, UINT8_MAX, 0);
            } else if (dst_dtype == BM_UINT16) {
                TRANSFER_INT(src, uint32_t, dst, uint16_t, scale, size, UINT16_MAX, 0);
            } else if (dst_dtype == BM_INT16) {
                TRANSFER_INT(src, uint32_t, dst, int16_t, scale, size, INT16_MAX, INT16_MIN);
            } else if (dst_dtype == BM_UINT32) {
                TRANSFER_INT(src, uint32_t, dst, uint32_t, scale, size, UINT32_MAX, (0U));
            } else if (dst_dtype == BM_INT32) {
                TRANSFER_INT(src, uint32_t, dst, int32_t, scale, size, INT32_MAX, INT32_MIN);
            } else if (dst_dtype == BM_FLOAT32) {
                TRANSFER_FLOAT(src, uint32_t, dst, float, scale, size)
            } else {
                printf("ERROR: don't support dtype=%d now!", dst_dtype);
                exit(0);
            }
        } else if (src_dtype == BM_INT32) {
            if (dst_dtype == BM_INT8) {
                TRANSFER_INT(src, int32_t, dst, int8_t, scale, size, 127, -127);
            } else if (dst_dtype == BM_UINT8) {
                TRANSFER_INT(src, int32_t, dst, uint8_t, scale, size, UINT8_MAX, 0);
            } else if (dst_dtype == BM_UINT16) {
                TRANSFER_INT(src, int32_t, dst, uint16_t, scale, size, UINT16_MAX, 0);
            } else if (dst_dtype == BM_INT16) {
                TRANSFER_INT(src, int32_t, dst, int16_t, scale, size, INT16_MAX, INT16_MIN);
            } else if (dst_dtype == BM_UINT32) {
                TRANSFER_INT(src, int32_t, dst, uint32_t, scale, size, UINT32_MAX, (0U));
            } else if (dst_dtype == BM_INT32) {
                TRANSFER_INT(src, int32_t, dst, int32_t, scale, size, INT32_MAX, INT32_MIN);
            } else if (dst_dtype == BM_FLOAT32) {
                TRANSFER_FLOAT(src, int32_t, dst, float, scale, size)
            } else {
                printf("ERROR: don't support dtype=%d now!", dst_dtype);
                exit(0);
            }
        } else if (src_dtype == BM_FLOAT32) {
            if (dst_dtype == BM_INT8) {
                TRANSFER_INT(src, float, dst, int8_t, scale, size, 127, -127);
            } else if (dst_dtype == BM_UINT8) {
                TRANSFER_INT(src, float, dst, uint8_t, scale, size, UINT8_MAX, 0);
            } else if (dst_dtype == BM_UINT16) {
                TRANSFER_INT(src, float, dst, uint16_t, scale, size, UINT16_MAX, 0);
            } else if (dst_dtype == BM_INT16) {
                TRANSFER_INT(src, float, dst, int16_t, scale, size, INT16_MAX, INT16_MIN);
            } else if (dst_dtype == BM_UINT32) {
                TRANSFER_INT(src, float, dst, uint32_t, scale, size, UINT32_MAX, (0U));
            } else if (dst_dtype == BM_INT32) {
                TRANSFER_INT(src, float, dst, int32_t, scale, size, INT32_MAX, INT32_MIN);
            } else if (dst_dtype == BM_FLOAT32) {
                TRANSFER_FLOAT(src, float, dst, float, scale, size)
            } else {
                printf("ERROR: don't support dtype=%d now!", dst_dtype);
                exit(0);
            }
        } else {
            printf("ERROR: don't support dtype=%d now!", src_dtype);
            exit(0);
        }
    }

#if USE_ASM_SSE

    static void scale_fp32_to_int8(float *src, int8_t *dst, float scale, int size) {
        int i = 0;
#if defined(__amd64__) || defined(__x86_64__)
        __m128 vec4fp32;
        __m128i vec4int;
        __m128 vec4scales = _mm_set1_ps(scale);
        for (i = 0; i < size - 3; i += 4) {
            vec4fp32 = _mm_load_ps(src + i);
            vec4fp32 = _mm_mul_ps(vec4fp32, vec4scales);
            vec4int = _mm_cvtps_epi32(vec4fp32);
            vec4int = _mm_packs_epi32(vec4int, vec4int);
            vec4int = _mm_packs_epi16(vec4int, vec4int);
            *reinterpret_cast<int *>(dst + i) = _mm_cvtsi128_si32(vec4int);
        }


#elif defined(__arm__) || defined(__aarch64__)
        int8x8_t  vec_s8x8; // target 8 x int8
        int32x4_t vec_s32x4_l, vec_s32x4_h;
        int32x4_t min_val = vdupq_n_s32(-128);
        int32x4_t max_val = vdupq_n_s32(127);
        float32x4_t vec_f32x4_l, vec_f32x4_h;

        for (i = 0; i < size-7; i+=8) {
            vec_f32x4_l = vld1q_f32(src+i);
            vec_f32x4_h = vld1q_f32(src+i+4);
            vec_f32x4_l = vmulq_n_f32(vec_f32x4_l, scale);
            vec_f32x4_h = vmulq_n_f32(vec_f32x4_h, scale);
            vec_s32x4_l = vcvtnq_s32_f32(vec_f32x4_l);
            vec_s32x4_h = vcvtnq_s32_f32(vec_f32x4_h);
            vec_s32x4_l = vminq_s32(vmaxq_s32(vec_s32x4_l, min_val), max_val);
            vec_s32x4_h = vminq_s32(vmaxq_s32(vec_s32x4_h, min_val), max_val);
            vec_s8x8 = vmovn_s16(vcombine_s16(vmovn_s32(vec_s32x4_l), vmovn_s32(vec_s32x4_h)));
            vst1_s8(dst+i, vec_s8x8);
        }
#else
#error "ERROR:unknown arch, only support arm/x86"
#endif

        for (; i < size; i++){
            int v = std::round(src[i] * scale);
            dst[i] = (int8_t)std::min(std::max(v, -127), 127);
        }
    }

    static void scale_int8_to_fp32(int8_t *src, float *dst, float scale, int size)
    {
        int i = 0;
#if defined(__amd64__) || defined(__x86_64__)
        __m128  vec4scales = _mm_set1_ps(scale);
        __m128i vec4int;
        __m128  vec4fp32;

        for (i = 0; i < size-3; i+=4) {
            vec4int = _mm_cvtsi32_si128(*reinterpret_cast<int*>(src+i));
            vec4int = _mm_unpacklo_epi8(vec4int,  _mm_cmplt_epi8(vec4int, _mm_setzero_si128()));
            vec4int = _mm_unpacklo_epi16(vec4int, _mm_cmplt_epi8(vec4int, _mm_setzero_si128()));
            vec4fp32 = _mm_cvtepi32_ps(vec4int);
            vec4fp32 = _mm_mul_ps(vec4fp32, vec4scales);
            _mm_store_ps(dst+i, vec4fp32);
        }


#elif defined(__arm__) || defined(__aarch64__)
        int8x8_t    vec_s8x8;
        int32x4_t   vec_s32x4;
        float32x4_t vec_f32x4;

        for (i = 0; i < size-3; i+=4) {vec_s8x8  = vld1_s8(src+i);
            vec_s8x8  = vget_low_s8(vcombine_s8(vec_s8x8, vcreate_s8(0)));
            vec_s32x4 = vmovl_s16(vget_low_s16(vmovl_s8(vec_s8x8)));
            vec_f32x4 = vcvtq_f32_s32(vec_s32x4);
            vec_f32x4 = vmulq_n_f32(vec_f32x4, scale);
            vst1q_f32(dst+i, vec_f32x4);
        }
#else
#error "ERROR:unknown arch, only support arm/x86"
#endif
        for (; i < size; i++)
            dst[i] = (float)src[i]*scale;

    }

    static void scale_fp32_to_uint8(float *src, uint8_t *dst, float scale, int size) {
        int i = 0;
#if defined(__amd64__) || defined(__x86_64__)
        __m128  vec4fp32;
        __m128i vec4int;
        __m128  vec4scales = _mm_set1_ps(scale);

        for (i = 0; i < size-3; i+=4) {
            vec4fp32 = _mm_load_ps(src+i);
            vec4fp32 = _mm_mul_ps(vec4fp32, vec4scales);
            vec4int = _mm_cvtps_epi32(vec4fp32);
            vec4int = _mm_packus_epi32(vec4int, vec4int);
            vec4int = _mm_packus_epi16(vec4int, vec4int);
            *reinterpret_cast<int*>(dst+i) = _mm_cvtsi128_si32(vec4int);
        }


#elif defined(__arm__) || defined(__aarch64__)
        uint8x8_t  vec_u8x8; // target 8 x int8
        uint32x4_t vec_u32x4_l, vec_u32x4_h;
        uint32x4_t min_val = vdupq_n_u32(0);
        uint32x4_t max_val = vdupq_n_u32(255);
        float32x4_t vec_f32x4_l, vec_f32x4_h;

        for (i = 0; i < size-7; i+=8) {
          vec_f32x4_l = vld1q_f32(src+i);
          vec_f32x4_h = vld1q_f32(src+i+4);
          vec_f32x4_l = vmulq_n_f32(vec_f32x4_l, scale);
          vec_f32x4_h = vmulq_n_f32(vec_f32x4_h, scale);
          vec_u32x4_l = vcvtnq_u32_f32(vec_f32x4_l);
          vec_u32x4_h = vcvtnq_u32_f32(vec_f32x4_h);
          vec_u32x4_l = vminq_u32(vmaxq_u32(vec_u32x4_l, min_val), max_val);
          vec_u32x4_h = vminq_u32(vmaxq_u32(vec_u32x4_h, min_val), max_val);
          vec_u8x8 = vmovn_u16(vcombine_u16(vmovn_u32(vec_u32x4_l), vmovn_u32(vec_u32x4_h)));
          vst1_u8(dst+i, vec_u8x8);
        }
#else
#error "ERROR:unknown arch, only support arm/x86"
#endif
        for (; i < size; i++){
            int v = std::round(src[i] * scale);
            dst[i] = (uint8_t)std::min(std::max(v, 0), 255);
        }

    }

    static void scale_uint8_to_fp32(uint8_t *src, float *dst, float scale, int size)
    {
        int i = 0;
#if defined(__amd64__) || defined(__x86_64__)
        __m128  vec4scales = _mm_set1_ps(scale);
        __m128i vec4int;
        __m128  vec4fp32;

        for (i = 0; i < size-3; i+=4) {
            vec4int = _mm_cvtsi32_si128(*reinterpret_cast<int*>(src+i));
            vec4int = _mm_unpacklo_epi8(vec4int, _mm_setzero_si128());
            vec4int = _mm_unpacklo_epi16(vec4int, _mm_setzero_si128());
            vec4fp32 = _mm_cvtepi32_ps(vec4int);
            vec4fp32 = _mm_mul_ps(vec4fp32, vec4scales);
            _mm_store_ps(dst+i, vec4fp32);
        }



#elif defined(__arm__) || defined(__aarch64__)
        uint8x8_t   vec_u8x8;
        uint32x4_t  vec_u32x4;
        float32x4_t vec_f32x4;

        for (i = 0; i < size-3; i+=4) {
          vec_u8x8  = vld1_u8(src+i);
          vec_u8x8  = vget_low_u8(vcombine_u8(vec_u8x8, vcreate_u8(0)));
          vec_u32x4 = vmovl_u16(vget_low_u16(vmovl_u8(vec_u8x8)));
          vec_f32x4 = vcvtq_f32_u32(vec_u32x4);
          vec_f32x4 = vmulq_n_f32(vec_f32x4, scale);
          vst1q_f32(dst+i, vec_f32x4);
        }
#else
#error "ERROR:unknown arch, only support arm/x86"
#endif
        for (; i < size; i++)
            dst[i] = (float)src[i]*scale;
    }

    static void AnyScale_SSE(void *src, bm_data_type_t src_dtype, void *dst, bm_data_type_t dst_dtype,  float scale, int size)
    {
        if (nullptr == src || nullptr == dst) {
            SPDLOG_ERROR("AnyScale_SSE param err, src={}, dst={}", src, dst);
            return;
        }

        if (src_dtype == BM_INT8) {
            if (dst_dtype == BM_INT8) {
                TRANSFER_INT(src, int8_t, dst, int8_t, scale, size, 127, -127)
            } else if (dst_dtype == BM_UINT8) {
                TRANSFER_INT(src, int8_t, dst, uint8_t, scale, size, UINT8_MAX, 0)
            } else if (dst_dtype == BM_UINT16) {
                TRANSFER_INT(src, int8_t, dst, uint16_t, scale, size, UINT16_MAX, 0)
            } else if (dst_dtype == BM_INT16) {
                TRANSFER_INT(src, int8_t, dst, int16_t, scale, size, INT16_MAX, INT16_MIN)
            } else if (dst_dtype == BM_UINT32) {
                TRANSFER_INT(src, int8_t, dst, uint32_t, scale, size, UINT32_MAX, (0U))
            } else if (dst_dtype == BM_INT32) {
                TRANSFER_INT(src, int8_t, dst, int32_t, scale, size, INT32_MAX, INT32_MIN)
            } else if (dst_dtype == BM_FLOAT32) {
                scale_int8_to_fp32((int8_t*)src, (float*)dst, scale, size);
            } else {
                printf("ERROR: don't support dtype=%d now!", dst_dtype);
                exit(0);
            }
        } else if (src_dtype == BM_UINT8) {
            if (dst_dtype == BM_INT8) {
                TRANSFER_INT(src, uint8_t, dst, int8_t, scale, size, 127, -127);
            } else if (dst_dtype == BM_UINT8) {
                TRANSFER_INT(src, uint8_t, dst, uint8_t, scale, size, UINT8_MAX, 0);
            } else if (dst_dtype == BM_UINT16) {
                TRANSFER_INT(src, uint8_t, dst, uint16_t, scale, size, UINT16_MAX, 0);
            } else if (dst_dtype == BM_INT16) {
                TRANSFER_INT(src, uint8_t, dst, int16_t, scale, size, INT16_MAX, INT16_MIN);
            } else if (dst_dtype == BM_UINT32) {
                TRANSFER_INT(src, uint8_t, dst, uint32_t, scale, size, UINT32_MAX, (0U));
            } else if (dst_dtype == BM_INT32) {
                TRANSFER_INT(src, uint8_t, dst, int32_t, scale, size, INT32_MAX, INT32_MIN);
            } else if (dst_dtype == BM_FLOAT32) {
                scale_uint8_to_fp32((uint8_t*)src, (float*)dst, scale, size);
            } else {
                printf("ERROR: don't support dtype=%d now!", dst_dtype);
                exit(0);
            }
        } else if (src_dtype == BM_UINT16) {
            if (dst_dtype == BM_INT8) {
                TRANSFER_INT(src, uint16_t, dst, int8_t, scale, size, 127, -127);
            } else if (dst_dtype == BM_UINT8) {
                TRANSFER_INT(src, uint16_t, dst, uint8_t, scale, size, UINT8_MAX, 0);
            } else if (dst_dtype == BM_UINT16) {
                TRANSFER_INT(src, uint16_t, dst, uint16_t, scale, size, UINT16_MAX, 0);
            } else if (dst_dtype == BM_INT16) {
                TRANSFER_INT(src, uint16_t, dst, int16_t, scale, size, INT16_MAX, INT16_MIN);
            } else if (dst_dtype == BM_UINT32) {
                TRANSFER_INT(src, uint16_t, dst, uint32_t, scale, size, UINT32_MAX, (0U));
            } else if (dst_dtype == BM_INT32) {
                TRANSFER_INT(src, uint16_t, dst, int32_t, scale, size, INT32_MAX, INT32_MIN);
            } else if (dst_dtype == BM_FLOAT32) {
                TRANSFER_FLOAT(src, uint16_t, dst, float, scale, size)
            } else {
                printf("ERROR: don't support dtype=%d now!", dst_dtype);
                exit(0);
            }
        } else if (src_dtype == BM_INT16) {
            if (dst_dtype == BM_INT8) {
                TRANSFER_INT(src, int16_t, dst, int8_t, scale, size, 127, -127);
            } else if (dst_dtype == BM_UINT8) {
                TRANSFER_INT(src, int16_t, dst, uint8_t, scale, size, UINT8_MAX, 0);
            } else if (dst_dtype == BM_UINT16) {
                TRANSFER_INT(src, int16_t, dst, uint16_t, scale, size, UINT16_MAX, 0);
            } else if (dst_dtype == BM_INT16) {
                TRANSFER_INT(src, int16_t, dst, int16_t, scale, size, INT16_MAX, INT16_MIN);
            } else if (dst_dtype == BM_UINT32) {
                TRANSFER_INT(src, int16_t, dst, uint32_t, scale, size, UINT32_MAX, (0U));
            } else if (dst_dtype == BM_INT32) {
                TRANSFER_INT(src, int16_t, dst, int32_t, scale, size, INT32_MAX, INT32_MIN);
            } else if (dst_dtype == BM_FLOAT32) {
                TRANSFER_FLOAT(src, int16_t, dst, float, scale, size)
            } else {
                printf("ERROR: don't support dtype=%d now!", dst_dtype);
                exit(0);
            }
        } else if (src_dtype == BM_UINT32) {
            if (dst_dtype == BM_INT8) {
                TRANSFER_INT(src, uint32_t, dst, int8_t, scale, size, 127, -127);
            } else if (dst_dtype == BM_UINT8) {
                TRANSFER_INT(src, uint32_t, dst, uint8_t, scale, size, UINT8_MAX, 0);
            } else if (dst_dtype == BM_UINT16) {
                TRANSFER_INT(src, uint32_t, dst, uint16_t, scale, size, UINT16_MAX, 0);
            } else if (dst_dtype == BM_INT16) {
                TRANSFER_INT(src, uint32_t, dst, int16_t, scale, size, INT16_MAX, INT16_MIN);
            } else if (dst_dtype == BM_UINT32) {
                TRANSFER_INT(src, uint32_t, dst, uint32_t, scale, size, UINT32_MAX, (0U));
            } else if (dst_dtype == BM_INT32) {
                TRANSFER_INT(src, uint32_t, dst, int32_t, scale, size, INT32_MAX, INT32_MIN);
            } else if (dst_dtype == BM_FLOAT32) {
                TRANSFER_FLOAT(src, uint32_t, dst, float, scale, size)
            } else {
                printf("ERROR: don't support dtype=%d now!", dst_dtype);
                exit(0);
            }
        } else if (src_dtype == BM_INT32) {
            if (dst_dtype == BM_INT8) {
                TRANSFER_INT(src, int32_t, dst, int8_t, scale, size, 127, -127);
            } else if (dst_dtype == BM_UINT8) {
                TRANSFER_INT(src, int32_t, dst, uint8_t, scale, size, UINT8_MAX, 0);
            } else if (dst_dtype == BM_UINT16) {
                TRANSFER_INT(src, int32_t, dst, uint16_t, scale, size, UINT16_MAX, 0);
            } else if (dst_dtype == BM_INT16) {
                TRANSFER_INT(src, int32_t, dst, int16_t, scale, size, INT16_MAX, INT16_MIN);
            } else if (dst_dtype == BM_UINT32) {
                TRANSFER_INT(src, int32_t, dst, uint32_t, scale, size, UINT32_MAX, (0U));
            } else if (dst_dtype == BM_INT32) {
                TRANSFER_INT(src, int32_t, dst, int32_t, scale, size, INT32_MAX, INT32_MIN);
            } else if (dst_dtype == BM_FLOAT32) {
                TRANSFER_FLOAT(src, int32_t, dst, float, scale, size)
            } else {
                printf("ERROR: don't support dtype=%d now!", dst_dtype);
                exit(0);
            }
        } else if (src_dtype == BM_FLOAT32) {
            if (dst_dtype == BM_INT8) {
                scale_fp32_to_int8((float*)src, (int8_t*)dst, scale, size);
            } else if (dst_dtype == BM_UINT8) {
                scale_fp32_to_uint8((float*)src, (uint8_t*)dst, scale, size);
            } else if (dst_dtype == BM_UINT16) {
                TRANSFER_INT(src, float, dst, uint16_t, scale, size, UINT16_MAX, 0)
            } else if (dst_dtype == BM_INT16) {
                TRANSFER_INT(src, float, dst, int16_t, scale, size, INT16_MAX, INT16_MIN)
            } else if (dst_dtype == BM_UINT32) {
                TRANSFER_INT(src, float, dst, uint32_t, scale, size, UINT32_MAX, (0U))
            } else if (dst_dtype == BM_INT32) {
                TRANSFER_INT(src, float, dst, int32_t, scale, size, INT32_MAX, INT32_MIN)
            } else if (dst_dtype == BM_FLOAT32) {
                TRANSFER_FLOAT(src, float, dst, float, scale, size)
            } else {
                printf("ERROR: don't support dtype=%d now!", dst_dtype);
                exit(0);
            }
        } else {
            printf("ERROR: don't support dtype=%d now!", src_dtype);
            exit(0);
        }
    }
#endif

static __inline const char* bm_image_format_desc(bm_image_format_ext fmt) {
    switch (fmt) {
        case FORMAT_YUV420P:
            return "FORMAT_YUV420P";
        case FORMAT_YUV422P:
            return "FORMAT_YUV422P";
        case FORMAT_YUV444P:
            return "FORMAT_YUV444P";
        case FORMAT_NV12:
            return "FORMAT_NV12";
        case FORMAT_NV21:
            return "FORMAT_NV21";
        case FORMAT_NV16:
            return "FORMAT_NV16";
        case FORMAT_NV61:
            return "FORMAT_NV61";
        case FORMAT_NV24:
            return "FORMAT_NV24";
        case FORMAT_RGB_PLANAR:
            return "FORMAT_RGB_PLANAR";
        case FORMAT_BGR_PLANAR:
            return "FORMAT_BGR_PLANAR";
        case FORMAT_RGB_PACKED:
            return "FORMAT_RGB_PACKED";
        case FORMAT_BGR_PACKED:
            return "FORMAT_BGR_PACKED";
        case FORMAT_RGBP_SEPARATE:
            return "FORMAT_RGBP_SEPARATE";
        case FORMAT_BGRP_SEPARATE:
            return "FORMAT_BGRP_SEPARATE";
        case FORMAT_GRAY:
            return "FORMAT_GRAY";
        case FORMAT_COMPRESSED:
            return "FORMAT_COMPRESSED";
    }

    return "unknown bm_image_format";
}

static __inline const char* bm_image_data_type_desc(bm_image_data_format_ext dtype)
{
    switch(dtype) {
        case DATA_TYPE_EXT_FLOAT32: return "DATA_TYPE_EXT_FLOAT32";
        case DATA_TYPE_EXT_1N_BYTE: return "DATA_TYPE_EXT_1N_BYTE";
        case DATA_TYPE_EXT_4N_BYTE: return "DATA_TYPE_EXT_4N_BYTE";
        case DATA_TYPE_EXT_1N_BYTE_SIGNED: return "DATA_TYPE_EXT_1N_BYTE_SIGNED";
        case DATA_TYPE_EXT_4N_BYTE_SIGNED: return "DATA_TYPE_EXT_4N_BYTE_SIGNED";
    }

    return "unknown bm_image dtype";
}

static __inline void print_image(const bm_image &img, const char *prefix="") {
     spdlog::info("{} bmimage w={},h={}, format={}, dtype={}",prefix, img.width,
             img.height, bm_image_format_desc(img.image_format),
             bm_image_data_type_desc(img.data_type));
}

static __inline int bm_image_data_type_size(bm_image_data_format_ext dtype)
{
    switch (dtype) {
        case DATA_TYPE_EXT_FLOAT32: return 4;
        case DATA_TYPE_EXT_1N_BYTE: return 1;
        case DATA_TYPE_EXT_4N_BYTE: return 4;
        case DATA_TYPE_EXT_1N_BYTE_SIGNED: return 1;
        case DATA_TYPE_EXT_4N_BYTE_SIGNED: return 4;
    }

    return -1;
}

// shape to string
static __inline std::string shape_to_str(std::vector<int> shape) {
    std::stringstream ss;
    ss << "[" ;
    for(int i = 0; i < shape.size(); ++i) {
        ss << shape[i];
        if (i != shape.size()-1) {
            ss << ",";
        }
    }
    ss << "]";
    return ss.str();
}

// get shape size
static __inline int shape_size(const std::vector<int>& shape) {
    return  std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

static __inline const char* io_mode_str(int iomode) {
    switch(iomode) {
        case SYSIO: return "SYSIO";
        case SYSI: return "SYSI";
        case SYSO: return "SYSO";
        case DEVIO: return "DEVIO";
        default:
            return "None";
    }
}

} // end of ns(sail)

#endif //SOPHON_INFERENCE_INTERNAL_H
