/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#ifndef M_LOG2E
#define M_LOG2E 1.44269504088896340736f
#endif

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include "selective_scan.h"
#include "selective_scan_common.h"
#include "static_switch.h"

template<int kNThreads_, int kNItems_, bool kIsEvenLen_, bool kDeltaSoftplus_, typename input_t_, typename weight_t_, typename output_t_>
struct Selective_Scan_fwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    using output_t = output_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kNItems = kNItems_;
    static constexpr int MaxDState = MAX_DSTATE;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : std::min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsEvenLen = bool(kIsEvenLen_);
    static constexpr bool kDeltaSoftplus = bool(kDeltaSoftplus_);
    static constexpr int kMinBlocks = kNThreads == 128 && 3;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = float2;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = cub::BlockStore<output_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    static constexpr int kSmemIOSize = std::max({sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadVecT::TempStorage),
                                                 sizeof(typename BlockLoadWeightT::TempStorage),
                                                 sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_fwd_kernel(SSMParamsBase params) {
    constexpr bool kDeltaSoftplus = Ktraits::kDeltaSoftplus;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using output_t = typename Ktraits::output_t;
    using scan_t = typename Ktraits::scan_t;

    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id * params.delta_d_stride;
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * params.A_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    float D_val = params.D_ptr == nullptr ? 0 : reinterpret_cast<float *>(params.D_ptr)[dim_id];
    float delta_bias = params.delta_bias_ptr == nullptr ? 0 : reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id];
    output_t *out = reinterpret_cast<output_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + dim_id * params.out_d_stride;
    scan_t *x = params.x_ptr == nullptr
        ? nullptr
        : reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id) * (params.n_chunks) * params.dstate;

    constexpr int kChunkSize = kNThreads * kNItems;
    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        input_t u_vals[kNItems];
        input_t delta_vals_load[kNItems];
        __syncthreads();
        load_input<Ktraits>(u, u_vals, smem_load, params.seqlen - chunk * kChunkSize);
        __syncthreads();
        load_input<Ktraits>(delta, delta_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
        u += kChunkSize;
        delta += kChunkSize;

        float delta_vals[kNItems], out_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            delta_vals[i] = float(delta_vals_load[i]) + delta_bias;
            if constexpr (kDeltaSoftplus) {
                delta_vals[i] = delta_vals[i] <= 20.f ? log1pf(expf(delta_vals[i])) : delta_vals[i];
            }
            out_vals[i] = D_val * float(u_vals[i]);
        }

        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            constexpr float kLog2e = M_LOG2E;
            weight_t A_val = A[state_idx * params.A_dstate_stride];
            weight_t A_scaled = A_val * kLog2e;
            weight_t B_vals[kNItems], C_vals[kNItems];
            load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    smem_load_weight, (params.seqlen - chunk * kChunkSize));
            load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight, (params.seqlen - chunk * kChunkSize));

            scan_t thread_data[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                thread_data[i] = make_float2(exp2f(delta_vals[i] * A_scaled), delta_vals[i] * float(u_vals[i]) * B_vals[i]);
            }

            scan_t running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx] : make_float2(1.f, 0.f);
            SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
            Ktraits::BlockScanT(smem_scan).InclusiveScan(
                thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
            );
            if (threadIdx.x == 0) { smem_running_prefix[state_idx] = prefix_op.running_prefix; }
            if (x != nullptr && threadIdx.x % 32 == 0) {
                x[chunk * params.dstate + state_idx] = prefix_op.running_prefix;
            }

            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                out_vals[i] += thread_data[i].y * C_vals[i];
            }
        }

        output_t *out_cur = out + chunk * kChunkSize;
        __syncthreads();
        store_output<Ktraits>(out_cur, out_vals, smem_store, params.seqlen - chunk * kChunkSize);
        Bvar += kChunkSize;
        Cvar += kChunkSize;
    }
}

template<int kNThreads, int kNItems, bool kIsEvenLen, bool kDeltaSoftplus, typename input_t, typename weight_t, typename output_t>
inline void selective_scan_fwd_launch_impl(SSMParamsBase &params, cudaStream_t stream) {
    using Ktraits = Selective_Scan_fwd_kernel_traits<kNThreads, kNItems, kIsEvenLen, kDeltaSoftplus, input_t, weight_t, output_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize + Ktraits::MaxDState * sizeof(typename Ktraits::scan_t);
    dim3 grid(params.batch, params.dim);
    auto kernel = &selective_scan_fwd_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
    }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t, typename output_t>
void selective_scan_fwd_launch(SSMParamsBase &params, cudaStream_t stream) {
    const bool is_even_len = (params.seqlen % (kNThreads * kNItems) == 0);
    const bool delta_softplus = params.delta_softplus;

    if (is_even_len) {
        if (delta_softplus) {
            selective_scan_fwd_launch_impl<kNThreads, kNItems, true, true, input_t, weight_t, output_t>(params, stream);
        } else {
            selective_scan_fwd_launch_impl<kNThreads, kNItems, true, false, input_t, weight_t, output_t>(params, stream);
        }
    } else {
        if (delta_softplus) {
            selective_scan_fwd_launch_impl<kNThreads, kNItems, false, true, input_t, weight_t, output_t>(params, stream);
        } else {
            selective_scan_fwd_launch_impl<kNThreads, kNItems, false, false, input_t, weight_t, output_t>(params, stream);
        }
    }
}

template<int knrows, typename input_t, typename weight_t, typename output_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream) {
    if (params.seqlen <= 128) {
        selective_scan_fwd_launch<32, 4, input_t, weight_t, output_t>(params, stream);
    } else if (params.seqlen <= 256) {
        selective_scan_fwd_launch<32, 8, input_t, weight_t, output_t>(params, stream);
    } else if (params.seqlen <= 512) {
        selective_scan_fwd_launch<32, 16, input_t, weight_t, output_t>(params, stream);
    } else if (params.seqlen <= 1024) {
        selective_scan_fwd_launch<64, 16, input_t, weight_t, output_t>(params, stream);
    } else {
        selective_scan_fwd_launch<128, 16, input_t, weight_t, output_t>(params, stream);
    }
}