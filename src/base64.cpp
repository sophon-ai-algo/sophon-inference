#include "base64.h"

namespace sail {
    //
    // if success, return 0, else return -1
    //
    int base64_enc(Handle& handle, const void *data, uint32_t dlen, std::string& encoded) {
#if USE_BMCV
        if (data == nullptr) return -1;
        bm_handle_t bmHandle = (bm_handle_t) handle.data();
        unsigned long lens[2];
        lens[0] = dlen;
        int out_size = (dlen + 2) / 3 * 4;
        encoded.resize(out_size);
        bm_status_t ret = bmcv_base64_enc(bmHandle, bm_mem_from_system((void *) data),
                                          bm_mem_from_system((char *) encoded.data()), lens);
        if (BM_SUCCESS != ret) {
            return -1;
        }

        return 0;
#else
        return -1;
#endif
    }

    //
    // if success, return 0, else return -1
    //
    int base64_dec(Handle& handle, const void *data, uint32_t dlen, uint8_t* p_outbuf, uint32_t *p_size)
    {
#if USE_BMCV
        bm_handle_t bmHandle = (bm_handle_t)handle.data();
        unsigned long lens[2];
        lens[0] = dlen;
        int out_size = dlen/4*3;
        if (nullptr == p_outbuf && nullptr != p_size) {
            *p_size = out_size;
            return -1;
        }

        if (nullptr == data || nullptr == p_size) return -1;
        *p_size = out_size;

        bm_status_t ret = bmcv_base64_dec(bmHandle, bm_mem_from_system((void*)data), bm_mem_from_system(p_outbuf), lens);
        if (BM_SUCCESS != ret) {
            return -1;
        }

        *p_size = lens[1];
        return 0;
#else
        return -1;
#endif
    }
}