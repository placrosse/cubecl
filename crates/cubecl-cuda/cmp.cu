
[START_KERNEL_COMPILATION]
name: cubecl_linalg::matmul::kernels::cmma_old::base::cmma_launch::CmmaLaunch<
    f32,
    half::binary16::f16,
    cubecl_cuda::runtime::CudaRuntime,
>
cube_dim: (32, 4, 1)
shared_memory: 12288 bytes
info: KernelId {
    type_id: TypeId {
        t: (
            11245507605303491359,
            11119494559033075139,
        ),
    },
    info: Some (
         (
            CubeDim {
                x: 32,
                y: 4,
                z: 1,
            },
            ComptimeCmmaInfo {
                block_size_m: 64,
                block_size_k: 32,
                block_size_n: 64,
                tile_size_m: 16,
                tile_size_k: 16,
                tile_size_n: 16,
                check_m_bounds: false,
                check_k_bounds: false,
                check_n_bounds: false,
                unroll: false,
                plane_dim: 32,
                num_compute_planes: 4,
                num_buffers: 2,
                num_accumulators: 4,
                write_out_strategy: ReuseSmem,
                rasterization_strategy: Swizzle,
                compute_loop_order_strategy: AllAccumulatorsFirst (
                    true,
                ),
                lhs_smem_loader_strategy: Continuous (
                    RowMajor,
                ),
                rhs_smem_loader_strategy: Continuous (
                    RowMajor,
                ),
                main_loop_strategy: Standard,
                num_compute_planes_strategy: NumTilesM,
            },
            TensorCompilationArg {
                inplace: None,
                vectorisation: Some (
                    8,
                ),
            },
            TensorCompilationArg {
                inplace: None,
                vectorisation: Some (
                    8,
                ),
            },
            TensorCompilationArg {
                inplace: None,
                vectorisation: Some (
                    8,
                ),
            },
        ),
    ),
    mode: Some (
        Unchecked,
    ),
}
source:
```cpp
#include <mma.h>
#include <cuda_fp16.h>
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint;
typedef unsigned long long int uint64;
typedef long long int int64;

struct __align__(32) float_8 {
    float i_0;
    float i_1;
    float i_2;
    float i_3;
    float i_4;
    float i_5;
    float i_6;
    float i_7;
};


extern "C" __global__ void kernel(
float_8 input_0[],float_8 input_1[],float_8 output_0[],uint info[]
) {
    __shared__ __half shared_memory_0[2048];
    __shared__ __half shared_memory_1[2048];
    __shared__ float shared_memory_2[1024];
    uint l_0_0;
    uint l_0_1;
    uint l_0_2;
    uint l_0_3;
    uint l_0_4;
    uint l_0_5;
    uint l_0_6;
    uint l_0_7;
    uint l_0_8;
    uint l_0_9;
    uint l_0_10;
    uint l_0_11;
    uint l_0_12;
    uint l_0_13;
    uint l_0_14;
    uint l_0_15;
    uint l_0_16;
    uint l_0_17;
    uint l_0_18;
    uint l_0_19;
    uint l_0_20;
    uint l_0_21;
    uint l_0_22;
    float_8 l_0_23;
    float l_0_24;
    __half l_0_25;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_0_0;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_1_0;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_2_0;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_3_0;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_4_0;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_5_0;
    l_0_0 = info[uint(6)];
    l_0_1 = l_0_0 - uint(2);
    l_0_0 = l_0_0 - uint(1);
    l_0_2 = info[info[uint(9)] + l_0_1];
    l_0_1 = info[info[uint(9)] + l_0_0];
    l_0_3 = info[info[uint(10)] + l_0_0];
    l_0_0 = gridDim.y * uint(2);
    l_0_4 = blockIdx.x * gridDim.y;
    l_0_4 = l_0_4 + blockIdx.y;
    l_0_5 = l_0_4 % l_0_0;
    l_0_6 = l_0_4 / l_0_0;
    l_0_7 = l_0_5 / gridDim.y;
    l_0_6 = l_0_6 * uint(2);
    l_0_7 = l_0_6 + l_0_7;
    l_0_6 = l_0_5 % gridDim.y;
    l_0_5 = l_0_4 / l_0_0;
    l_0_5 = l_0_5 % uint(2);
    l_0_4 = uint(2) * l_0_6;
    l_0_4 = gridDim.y - l_0_4;
    l_0_4 = l_0_4 - uint(1);
    l_0_5 = l_0_5 * l_0_4;
    l_0_6 = l_0_6 + l_0_5;
    l_0_5 = l_0_6 * uint(64);
    l_0_4 = l_0_7 * uint(64);
    l_0_7 = info[uint(8)];
    l_0_6 = l_0_7 - uint(2);
    l_0_0 = info[info[uint(9)] + l_0_6];
    l_0_6 = l_0_7 - uint(1);
    l_0_8 = info[info[uint(10)] + l_0_6];
    l_0_8 = l_0_0 * l_0_8;
    l_0_8 = l_0_8 * blockIdx.z;
    l_0_6 = uint(0);
    l_0_0 = uint(0);
    l_0_7 = l_0_7 - uint(2);

    for (uint l_1_0 = uint(0); l_1_0 < l_0_7; ++l_1_0) {
        l_0_9 = info[info[uint(14)] + l_1_0];
        l_0_9 = l_0_8 / l_0_9;
        l_0_10 = info[info[uint(9)] + l_1_0];
        l_0_10 = l_0_9 % l_0_10;
        l_0_11 = info[info[uint(12)] + l_1_0];
        l_0_11 = l_0_10 * l_0_11;
        l_0_6 = l_0_6 + l_0_11;
        l_0_11 = info[info[uint(10)] + l_1_0];
        l_0_11 = l_0_9 % l_0_11;
        l_0_10 = info[info[uint(13)] + l_1_0];
        l_0_11 = l_0_11 * l_0_10;
        l_0_0 = l_0_0 + l_0_11;
    }
    l_0_11 = l_0_5;
    l_0_10 = l_0_4;
    l_0_9 = threadIdx.y;
    l_0_7 = threadIdx.x;
    l_0_5 = threadIdx.y;
    l_0_4 = threadIdx.x;
    nvcuda::wmma::fill_fragment(frag_0_0, float(0.0));
    nvcuda::wmma::fill_fragment(frag_1_0, float(0.0));
    nvcuda::wmma::fill_fragment(frag_2_0, float(0.0));
    nvcuda::wmma::fill_fragment(frag_3_0, float(0.0));
    l_0_12 = l_0_1 + uint(32);
    l_0_12 = l_0_12 - uint(1);
    l_0_12 = l_0_12 / uint(32);

    // global loop
    for (uint l_1_0 = uint(0); l_1_0 < l_0_12; ++l_1_0) {
        l_0_13 = l_1_0 * uint(32);
        l_0_14 = uint(32);
        l_0_15 = uint(64);
        l_0_15 = l_0_14 * l_0_15;
        l_0_14 = uint(4) * uint(8);
        l_0_14 = l_0_14 * uint(32);
        l_0_15 = l_0_15 / l_0_14;
        l_0_16 = l_0_5 * uint(32);
        l_0_16 = l_0_16 + l_0_4;
        l_0_16 = l_0_16 * uint(8);

        // load lhs
        for (uint l_2_0 = uint(0); l_2_0 < l_0_15; ++l_2_0) {
            l_0_17 = l_2_0 * l_0_14;
            l_0_17 = l_0_16 + l_0_17;
            l_0_18 = uint(16);
            l_0_19 = uint(16);
            l_0_19 = l_0_18 * l_0_19;
            l_0_18 = uint(32);
            l_0_20 = uint(16);
            l_0_20 = l_0_18 / l_0_20;
            l_0_18 = uint(64);
            l_0_21 = uint(16);
            l_0_21 = l_0_18 / l_0_21;
            l_0_18 = l_0_17 / l_0_19;
            l_0_22 = l_0_18 / l_0_20;
            l_0_20 = l_0_18 % l_0_20;
            l_0_21 = uint(16);
            l_0_19 = l_0_17 % l_0_19;
            l_0_18 = l_0_19 / l_0_21;
            l_0_21 = l_0_19 % l_0_21;
            l_0_19 = uint(16);
            l_0_19 = l_0_22 * l_0_19;
            l_0_19 = l_0_19 + l_0_18;
            l_0_18 = uint(16);
            l_0_18 = l_0_20 * l_0_18;
            l_0_21 = l_0_18 + l_0_21;
            l_0_22 = l_0_19 + l_0_11;
            l_0_20 = l_0_21 + l_0_13;
            l_0_22 = l_0_22 * l_0_1;
            l_0_22 = l_0_6 + l_0_22;
            l_0_22 = l_0_22 + l_0_20;
            l_0_22 = l_0_22 / uint(8);
            l_0_23 = input_0[l_0_22];
            l_0_22 = l_0_17 + uint(0);
            l_0_24 = l_0_23.i_0;
            l_0_25 = __half(l_0_24);
            shared_memory_0[l_0_22] = l_0_25;
            l_0_22 = l_0_17 + uint(1);
            l_0_24 = l_0_23.i_1;
            l_0_25 = __half(l_0_24);
            shared_memory_0[l_0_22] = l_0_25;
            l_0_22 = l_0_17 + uint(2);
            l_0_24 = l_0_23.i_2;
            l_0_25 = __half(l_0_24);
            shared_memory_0[l_0_22] = l_0_25;
            l_0_22 = l_0_17 + uint(3);
            l_0_24 = l_0_23.i_3;
            l_0_25 = __half(l_0_24);
            shared_memory_0[l_0_22] = l_0_25;
            l_0_22 = l_0_17 + uint(4);
            l_0_24 = l_0_23.i_4;
            l_0_25 = __half(l_0_24);
            shared_memory_0[l_0_22] = l_0_25;
            l_0_22 = l_0_17 + uint(5);
            l_0_24 = l_0_23.i_5;
            l_0_25 = __half(l_0_24);
            shared_memory_0[l_0_22] = l_0_25;
            l_0_22 = l_0_17 + uint(6);
            l_0_24 = l_0_23.i_6;
            l_0_25 = __half(l_0_24);
            shared_memory_0[l_0_22] = l_0_25;
            l_0_22 = l_0_17 + uint(7);
            l_0_24 = l_0_23.i_7;
            l_0_25 = __half(l_0_24);
            shared_memory_0[l_0_22] = l_0_25;
        }
        l_0_22 = uint(64);
        l_0_21 = uint(32);
        l_0_22 = l_0_22 * l_0_21;
        l_0_21 = uint(4) * uint(8);
        l_0_21 = l_0_21 * uint(32);
        l_0_22 = l_0_22 / l_0_21;
        l_0_20 = l_0_5 * uint(32);
        l_0_20 = l_0_20 + l_0_4;
        l_0_20 = l_0_20 * uint(8);

        // load rhs
        for (uint l_2_0 = uint(0); l_2_0 < l_0_22; ++l_2_0) {
            l_0_19 = l_2_0 * l_0_21;
            l_0_19 = l_0_20 + l_0_19;
            l_0_18 = uint(16);
            l_0_17 = uint(16);
            l_0_18 = l_0_18 * l_0_17;
            l_0_17 = uint(64);
            l_0_16 = uint(16);
            l_0_17 = l_0_17 / l_0_16;
            l_0_16 = uint(32);
            l_0_15 = uint(16);
            l_0_16 = l_0_16 / l_0_15;
            l_0_15 = l_0_19 / l_0_18;
            l_0_14 = l_0_15 / l_0_17;
            l_0_17 = l_0_15 % l_0_17;
            l_0_16 = uint(16);
            l_0_18 = l_0_19 % l_0_18;
            l_0_15 = l_0_18 / l_0_16;
            l_0_18 = l_0_18 % l_0_16;
            l_0_16 = uint(16);
            l_0_16 = l_0_14 * l_0_16;
            l_0_16 = l_0_16 + l_0_15;
            l_0_15 = uint(16);
            l_0_15 = l_0_17 * l_0_15;
            l_0_18 = l_0_15 + l_0_18;
            l_0_17 = l_0_16 + l_0_13;
            l_0_15 = l_0_18 + l_0_10;
            l_0_17 = l_0_17 * l_0_3;
            l_0_17 = l_0_0 + l_0_17;
            l_0_17 = l_0_17 + l_0_15;
            l_0_17 = l_0_17 / uint(8);
            l_0_23 = input_1[l_0_17];
            l_0_17 = l_0_19 + uint(0);
            l_0_24 = l_0_23.i_0;
            l_0_25 = __half(l_0_24);
            shared_memory_1[l_0_17] = l_0_25;
            l_0_17 = l_0_19 + uint(1);
            l_0_24 = l_0_23.i_1;
            l_0_25 = __half(l_0_24);
            shared_memory_1[l_0_17] = l_0_25;
            l_0_17 = l_0_19 + uint(2);
            l_0_24 = l_0_23.i_2;
            l_0_25 = __half(l_0_24);
            shared_memory_1[l_0_17] = l_0_25;
            l_0_17 = l_0_19 + uint(3);
            l_0_24 = l_0_23.i_3;
            l_0_25 = __half(l_0_24);
            shared_memory_1[l_0_17] = l_0_25;
            l_0_17 = l_0_19 + uint(4);
            l_0_24 = l_0_23.i_4;
            l_0_25 = __half(l_0_24);
            shared_memory_1[l_0_17] = l_0_25;
            l_0_17 = l_0_19 + uint(5);
            l_0_24 = l_0_23.i_5;
            l_0_25 = __half(l_0_24);
            shared_memory_1[l_0_17] = l_0_25;
            l_0_17 = l_0_19 + uint(6);
            l_0_24 = l_0_23.i_6;
            l_0_25 = __half(l_0_24);
            shared_memory_1[l_0_17] = l_0_25;
            l_0_17 = l_0_19 + uint(7);
            l_0_24 = l_0_23.i_7;
            l_0_25 = __half(l_0_24);
            shared_memory_1[l_0_17] = l_0_25;
        }
        __syncthreads();
        l_0_22 = l_0_9 / uint(1);
        l_0_21 = l_0_9 % uint(1);
        l_0_21 = l_0_21 * uint(4);

        for (uint l_2_0 = uint(0); l_2_0 < uint(2); ++l_2_0) {
            l_0_20 = uint(32);
            l_0_19 = uint(16);
            l_0_20 = l_0_20 / l_0_19;
            l_0_19 = uint(64);
            l_0_18 = uint(16);
            l_0_19 = l_0_19 / l_0_18;
            l_0_20 = l_0_22 * l_0_20;
            l_0_20 = l_0_20 + l_2_0;
            l_0_19 = uint(16);
            l_0_18 = uint(16);
            l_0_19 = l_0_19 * l_0_18;
            l_0_20 = l_0_20 * l_0_19;
            l_0_19 = l_0_20 + l_0_19;
            const uint slice_2_0_length = l_0_19 - l_0_20;
            __half *slice_2_0 = shared_memory_0 + l_0_20;
            l_0_20 = uint(16);
            nvcuda::wmma::load_matrix_sync(frag_4_0, slice_2_0, l_0_20);
            l_0_20 = l_0_21 + uint(0);
            l_0_19 = uint(64);
            l_0_18 = uint(16);
            l_0_19 = l_0_19 / l_0_18;
            l_0_18 = uint(32);
            l_0_17 = uint(16);
            l_0_18 = l_0_18 / l_0_17;
            l_0_19 = l_2_0 * l_0_19;
            l_0_19 = l_0_19 + l_0_20;
            l_0_20 = uint(16);
            l_0_18 = uint(16);
            l_0_20 = l_0_20 * l_0_18;
            l_0_19 = l_0_19 * l_0_20;
            l_0_20 = l_0_19 + l_0_20;
            const uint slice_2_1_length = l_0_20 - l_0_19;
            __half *slice_2_1 = shared_memory_1 + l_0_19;
            l_0_20 = uint(16);
            nvcuda::wmma::load_matrix_sync(frag_5_0, slice_2_1, l_0_20);
            nvcuda::wmma::mma_sync(frag_0_0, frag_4_0, frag_5_0, frag_0_0);
            l_0_20 = l_0_21 + uint(1);
            l_0_19 = uint(64);
            l_0_18 = uint(16);
            l_0_19 = l_0_19 / l_0_18;
            l_0_18 = uint(32);
            l_0_17 = uint(16);
            l_0_18 = l_0_18 / l_0_17;
            l_0_19 = l_2_0 * l_0_19;
            l_0_19 = l_0_19 + l_0_20;
            l_0_20 = uint(16);
            l_0_18 = uint(16);
            l_0_20 = l_0_20 * l_0_18;
            l_0_19 = l_0_19 * l_0_20;
            l_0_20 = l_0_19 + l_0_20;
            const uint slice_2_2_length = l_0_20 - l_0_19;
            __half *slice_2_2 = shared_memory_1 + l_0_19;
            l_0_20 = uint(16);
            nvcuda::wmma::load_matrix_sync(frag_5_0, slice_2_2, l_0_20);
            nvcuda::wmma::mma_sync(frag_1_0, frag_4_0, frag_5_0, frag_1_0);
            l_0_20 = l_0_21 + uint(2);
            l_0_19 = uint(64);
            l_0_18 = uint(16);
            l_0_19 = l_0_19 / l_0_18;
            l_0_18 = uint(32);
            l_0_17 = uint(16);
            l_0_18 = l_0_18 / l_0_17;
            l_0_19 = l_2_0 * l_0_19;
            l_0_19 = l_0_19 + l_0_20;
            l_0_20 = uint(16);
            l_0_18 = uint(16);
            l_0_20 = l_0_20 * l_0_18;
            l_0_19 = l_0_19 * l_0_20;
            l_0_20 = l_0_19 + l_0_20;
            const uint slice_2_3_length = l_0_20 - l_0_19;
            __half *slice_2_3 = shared_memory_1 + l_0_19;
            l_0_20 = uint(16);
            nvcuda::wmma::load_matrix_sync(frag_5_0, slice_2_3, l_0_20);
            nvcuda::wmma::mma_sync(frag_2_0, frag_4_0, frag_5_0, frag_2_0);
            l_0_20 = l_0_21 + uint(3);
            l_0_19 = uint(64);
            l_0_18 = uint(16);
            l_0_19 = l_0_19 / l_0_18;
            l_0_18 = uint(32);
            l_0_17 = uint(16);
            l_0_18 = l_0_18 / l_0_17;
            l_0_19 = l_2_0 * l_0_19;
            l_0_19 = l_0_19 + l_0_20;
            l_0_20 = uint(16);
            l_0_18 = uint(16);
            l_0_20 = l_0_20 * l_0_18;
            l_0_19 = l_0_19 * l_0_20;
            l_0_20 = l_0_19 + l_0_20;
            const uint slice_2_4_length = l_0_20 - l_0_19;
            __half *slice_2_4 = shared_memory_1 + l_0_19;
            l_0_20 = uint(16);
            nvcuda::wmma::load_matrix_sync(frag_5_0, slice_2_4, l_0_20);
            nvcuda::wmma::mma_sync(frag_3_0, frag_4_0, frag_5_0, frag_3_0);
        }
        __syncthreads();
    }
    l_0_22 = l_0_9 * uint(256);
    l_0_21 = l_0_22 + uint(256);
    const uint slice_0_0_length = l_0_21 - l_0_22;
    float *slice_0_0 = shared_memory_2 + l_0_22;
    nvcuda::wmma::store_matrix_sync(slice_0_0, frag_0_0, uint(16), nvcuda::wmma::mem_row_major);
    l_0_22 = l_0_9 / uint(1);
    l_0_21 = l_0_9 % uint(1);
    l_0_21 = l_0_21 * uint(4);
    l_0_20 = l_0_9 * uint(256);
    l_0_19 = l_0_7 * uint(8);
    l_0_20 = l_0_20 + l_0_19;
    l_0_19 = l_0_7 / uint(2);
    l_0_18 = l_0_7 % uint(2);
    l_0_18 = l_0_18 * uint(8);
    l_0_22 = l_0_22 * uint(16);
    l_0_22 = l_0_11 + l_0_22;
    l_0_21 = l_0_21 * uint(16);
    l_0_21 = l_0_10 + l_0_21;
    l_0_21 = l_0_21 + l_0_18;
    l_0_18 = uint(0) * uint(16);
    l_0_21 = l_0_21 + l_0_18;
    l_0_18 = uint(0) * uint(256);
    l_0_18 = l_0_20 + l_0_18;
    l_0_17 = l_0_22 + l_0_19;
    l_0_16 = uint(0) * uint(16);
    l_0_17 = l_0_17 + l_0_16;
    l_0_17 = l_0_17 * l_0_3;
    l_0_17 = l_0_8 + l_0_17;
    l_0_17 = l_0_17 + l_0_21;
    l_0_16 = l_0_18 + uint(0);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_0 = l_0_24;
    l_0_16 = l_0_18 + uint(1);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_1 = l_0_24;
    l_0_16 = l_0_18 + uint(2);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_2 = l_0_24;
    l_0_16 = l_0_18 + uint(3);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_3 = l_0_24;
    l_0_16 = l_0_18 + uint(4);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_4 = l_0_24;
    l_0_16 = l_0_18 + uint(5);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_5 = l_0_24;
    l_0_16 = l_0_18 + uint(6);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_6 = l_0_24;
    l_0_16 = l_0_18 + uint(7);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_7 = l_0_24;
    l_0_17 = l_0_17 / uint(8);
    output_0[l_0_17] = l_0_23;
    nvcuda::wmma::store_matrix_sync(slice_0_0, frag_1_0, uint(16), nvcuda::wmma::mem_row_major);
    l_0_22 = l_0_9 / uint(1);
    l_0_21 = l_0_9 % uint(1);
    l_0_21 = l_0_21 * uint(4);
    l_0_20 = l_0_9 * uint(256);
    l_0_19 = l_0_7 * uint(8);
    l_0_20 = l_0_20 + l_0_19;
    l_0_19 = l_0_7 / uint(2);
    l_0_18 = l_0_7 % uint(2);
    l_0_18 = l_0_18 * uint(8);
    l_0_22 = l_0_22 * uint(16);
    l_0_22 = l_0_11 + l_0_22;
    l_0_21 = l_0_21 * uint(16);
    l_0_21 = l_0_10 + l_0_21;
    l_0_21 = l_0_21 + l_0_18;
    l_0_18 = uint(1) * uint(16);
    l_0_21 = l_0_21 + l_0_18;
    l_0_18 = uint(0) * uint(256);
    l_0_18 = l_0_20 + l_0_18;
    l_0_17 = l_0_22 + l_0_19;
    l_0_16 = uint(0) * uint(16);
    l_0_17 = l_0_17 + l_0_16;
    l_0_17 = l_0_17 * l_0_3;
    l_0_17 = l_0_8 + l_0_17;
    l_0_17 = l_0_17 + l_0_21;
    l_0_16 = l_0_18 + uint(0);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_0 = l_0_24;
    l_0_16 = l_0_18 + uint(1);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_1 = l_0_24;
    l_0_16 = l_0_18 + uint(2);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_2 = l_0_24;
    l_0_16 = l_0_18 + uint(3);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_3 = l_0_24;
    l_0_16 = l_0_18 + uint(4);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_4 = l_0_24;
    l_0_16 = l_0_18 + uint(5);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_5 = l_0_24;
    l_0_16 = l_0_18 + uint(6);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_6 = l_0_24;
    l_0_16 = l_0_18 + uint(7);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_7 = l_0_24;
    l_0_17 = l_0_17 / uint(8);
    output_0[l_0_17] = l_0_23;
    nvcuda::wmma::store_matrix_sync(slice_0_0, frag_2_0, uint(16), nvcuda::wmma::mem_row_major);
    l_0_22 = l_0_9 / uint(1);
    l_0_21 = l_0_9 % uint(1);
    l_0_21 = l_0_21 * uint(4);
    l_0_20 = l_0_9 * uint(256);
    l_0_19 = l_0_7 * uint(8);
    l_0_20 = l_0_20 + l_0_19;
    l_0_19 = l_0_7 / uint(2);
    l_0_18 = l_0_7 % uint(2);
    l_0_18 = l_0_18 * uint(8);
    l_0_22 = l_0_22 * uint(16);
    l_0_22 = l_0_11 + l_0_22;
    l_0_21 = l_0_21 * uint(16);
    l_0_21 = l_0_10 + l_0_21;
    l_0_21 = l_0_21 + l_0_18;
    l_0_18 = uint(2) * uint(16);
    l_0_21 = l_0_21 + l_0_18;
    l_0_18 = uint(0) * uint(256);
    l_0_18 = l_0_20 + l_0_18;
    l_0_17 = l_0_22 + l_0_19;
    l_0_16 = uint(0) * uint(16);
    l_0_17 = l_0_17 + l_0_16;
    l_0_17 = l_0_17 * l_0_3;
    l_0_17 = l_0_8 + l_0_17;
    l_0_17 = l_0_17 + l_0_21;
    l_0_16 = l_0_18 + uint(0);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_0 = l_0_24;
    l_0_16 = l_0_18 + uint(1);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_1 = l_0_24;
    l_0_16 = l_0_18 + uint(2);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_2 = l_0_24;
    l_0_16 = l_0_18 + uint(3);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_3 = l_0_24;
    l_0_16 = l_0_18 + uint(4);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_4 = l_0_24;
    l_0_16 = l_0_18 + uint(5);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_5 = l_0_24;
    l_0_16 = l_0_18 + uint(6);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_6 = l_0_24;
    l_0_16 = l_0_18 + uint(7);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_7 = l_0_24;
    l_0_17 = l_0_17 / uint(8);
    output_0[l_0_17] = l_0_23;
    nvcuda::wmma::store_matrix_sync(slice_0_0, frag_3_0, uint(16), nvcuda::wmma::mem_row_major);
    l_0_22 = l_0_9 / uint(1);
    l_0_21 = l_0_9 % uint(1);
    l_0_21 = l_0_21 * uint(4);
    l_0_20 = l_0_9 * uint(256);
    l_0_19 = l_0_7 * uint(8);
    l_0_20 = l_0_20 + l_0_19;
    l_0_19 = l_0_7 / uint(2);
    l_0_18 = l_0_7 % uint(2);
    l_0_18 = l_0_18 * uint(8);
    l_0_22 = l_0_22 * uint(16);
    l_0_22 = l_0_11 + l_0_22;
    l_0_21 = l_0_21 * uint(16);
    l_0_21 = l_0_10 + l_0_21;
    l_0_21 = l_0_21 + l_0_18;
    l_0_18 = uint(3) * uint(16);
    l_0_21 = l_0_21 + l_0_18;
    l_0_18 = uint(0) * uint(256);
    l_0_18 = l_0_20 + l_0_18;
    l_0_17 = l_0_22 + l_0_19;
    l_0_16 = uint(0) * uint(16);
    l_0_17 = l_0_17 + l_0_16;
    l_0_17 = l_0_17 * l_0_3;
    l_0_17 = l_0_8 + l_0_17;
    l_0_17 = l_0_17 + l_0_21;
    l_0_16 = l_0_18 + uint(0);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_0 = l_0_24;
    l_0_16 = l_0_18 + uint(1);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_1 = l_0_24;
    l_0_16 = l_0_18 + uint(2);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_2 = l_0_24;
    l_0_16 = l_0_18 + uint(3);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_3 = l_0_24;
    l_0_16 = l_0_18 + uint(4);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_4 = l_0_24;
    l_0_16 = l_0_18 + uint(5);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_5 = l_0_24;
    l_0_16 = l_0_18 + uint(6);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_6 = l_0_24;
    l_0_16 = l_0_18 + uint(7);
    l_0_24 = shared_memory_2[l_0_16];
    l_0_23.i_7 = l_0_24;
    l_0_17 = l_0_17 / uint(8);
    output_0[l_0_17] = l_0_23;

}
```
[END_KERNEL_COMPILATION]

