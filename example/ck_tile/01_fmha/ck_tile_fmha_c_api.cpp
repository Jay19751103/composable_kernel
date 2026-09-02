// C ABI shim implementation for ck_tile FMHA forward.
// Compiled as HIP for the target gfx arch and linked with the generated fmha_fwd
// instances + fmha_fwd_api.cpp (which defines fmha_fwd()).
//
// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
#include <string>

#include "fmha_fwd.hpp"
#include "mask.hpp"
#include "bias.hpp"
#include "ck_tile_fmha_c_api.h"

namespace {

inline void fill_traits(const ck_fa_params* p, fmha_fwd_traits& t) {
    t.hdim_q              = p->hdim_q;
    t.hdim_v              = p->hdim_v;
    t.data_type           = (p->dtype == CK_FA_BF16) ? std::string("bf16") : std::string("fp16");
    t.is_group_mode       = false;
    t.is_v_rowmajor       = true;
    t.has_logits_soft_cap = false;
    t.mask_type           = static_cast<mask_enum>(p->mask_mode);
    t.bias_type           = (p->bias_mode == CK_FA_BIAS_ELEMENTWISE) ? bias_enum::elementwise_bias
                                                                     : bias_enum::no_bias;
    t.has_lse             = (p->has_lse != 0);
    t.has_dropout         = false;
    t.qscale_type         = quant_scale_enum::no_scale;
    t.skip_min_seqlen_q   = false;
    t.has_sink            = (p->sink_ptr != nullptr);
}

inline void fill_args(const ck_fa_params* p, fmha_fwd_args& a) {
    a.q_ptr    = p->q_ptr;
    a.k_ptr    = p->k_ptr;
    a.v_ptr    = p->v_ptr;
    a.bias_ptr = p->bias_ptr;
    a.o_ptr    = p->o_ptr;
    a.lse_ptr  = p->lse_ptr;
    a.sink_ptr = p->sink_ptr;

    a.q_descale_ptr = nullptr;
    a.k_descale_ptr = nullptr;
    a.v_descale_ptr = nullptr;
    a.rand_val_ptr  = nullptr;

    a.seqstart_q_ptr           = nullptr;
    a.seqstart_k_ptr           = nullptr;
    a.seqlen_q_ptr             = nullptr;
    a.seqlen_k_ptr             = nullptr;
    a.cu_seqlen_q_ptr          = nullptr;
    a.cu_seqlen_k_ptr          = nullptr;
    a.block_scale_seqstart_q_ptr = nullptr;
    a.block_scale_seqstart_k_ptr = nullptr;
    a.seqstart_v_scale_ptr     = nullptr;

    a.seqlen_q         = p->seqlen_q;
    a.seqlen_k         = p->seqlen_k;
    a.batch            = p->batch;
    a.max_seqlen_q     = p->max_seqlen_q;
    a.hdim_q           = p->hdim_q;
    a.hdim_v           = p->hdim_v;
    a.nhead_q          = p->nhead_q;
    a.nhead_k          = p->nhead_k;
    a.num_head_q_total = p->nhead_q;
    a.head_start       = 0;

    a.scale_s         = p->scale_s;
    a.logits_soft_cap = 0.0f;

    a.stride_q    = p->stride_q;
    a.stride_k    = p->stride_k;
    a.stride_v    = p->stride_v;
    a.stride_bias = p->stride_bias;
    a.stride_randval = 0;
    a.stride_o    = p->stride_o;
    a.stride_q_descale = 0;
    a.stride_k_descale = 0;
    a.stride_v_descale = 0;

    a.nhead_stride_q    = p->nhead_stride_q;
    a.nhead_stride_k    = p->nhead_stride_k;
    a.nhead_stride_v    = p->nhead_stride_v;
    a.nhead_stride_bias = p->nhead_stride_bias;
    a.nhead_stride_randval = 0;
    a.nhead_stride_lse  = p->nhead_stride_lse;
    a.nhead_stride_o    = p->nhead_stride_o;
    a.nhead_stride_q_descale = 0;
    a.nhead_stride_k_descale = 0;
    a.nhead_stride_v_descale = 0;

    a.batch_stride_q    = p->batch_stride_q;
    a.batch_stride_k    = p->batch_stride_k;
    a.batch_stride_v    = p->batch_stride_v;
    a.batch_stride_bias = p->batch_stride_bias;
    a.batch_stride_randval = 0;
    a.batch_stride_lse  = p->batch_stride_lse;
    a.batch_stride_o    = p->batch_stride_o;
    a.batch_stride_q_descale = 0;
    a.batch_stride_k_descale = 0;
    a.batch_stride_v_descale = 0;

    // mask window: for plain causal use (-1, 0); no_mask uses (-1,-1).
    if (p->mask_mode == CK_FA_MASK_NONE) {
        a.window_size_left  = -1;
        a.window_size_right = -1;
    } else {
        a.window_size_left  = (p->window_left  >= 0) ? p->window_left  : -1;
        a.window_size_right = (p->window_right >= 0) ? p->window_right : 0;
    }
    a.sink_size    = 0;
    a.mask_type    = p->mask_mode;
    a.min_seqlen_q = 0;
}

} // namespace

extern "C" int ck_tile_fmha_fwd(const ck_fa_params* p, void* stream) {
    fmha_fwd_traits traits;
    fill_traits(p, traits);

    fmha_fwd_args args;
    fill_args(p, args);

    ck_tile::stream_config cfg;
    cfg.stream_id_   = reinterpret_cast<hipStream_t>(stream);
    cfg.time_kernel_ = false;
    cfg.log_level_   = 0;

    const float r = fmha_fwd(traits, args, cfg);
    return (r >= 0.0f) ? 0 : -1;
}

extern "C" int ck_tile_fmha_fwd_supported(const ck_fa_params* p) {
    const bool dt_ok  = (p->dtype == CK_FA_FP16 || p->dtype == CK_FA_BF16);
    const bool hd_ok  = (p->hdim_q == 64  && p->hdim_v == 64)  ||
                        (p->hdim_q == 128 && p->hdim_v == 128) ||
                        (p->hdim_q == 256 && p->hdim_v == 256);
    return (dt_ok && hd_ok) ? 1 : 0;
}
