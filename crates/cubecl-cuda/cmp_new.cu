
[START_KERNEL_COMPILATION]
name: cubecl_linalg::matmul::components::batch::base::launch::Launch<
    f32,
    cubecl_linalg::matmul::components::batch::one_to_one::Matmul<
        f32,
        half::binary16::f16,
        cubecl_linalg::matmul::components::global::homogeneous::base::Matmul<
            f32,
            half::binary16::f16,
            cubecl_linalg::matmul::components::stage::multi_buffer::base::Matmul<
                half::binary16::f16,
                f32,
                f32,
                cubecl_linalg::matmul::components::tile::accelerated::Accelerated16x16x16<
                    half::binary16::f16,
                    f32,
                >,
                cubecl_linalg::matmul::components::stage::base::S4x4x2,
            >,
        >,
        cubecl_linalg::matmul::components::batch::cube_dispatch::NaturalDispatch,
    >,
    cubecl_cuda::runtime::CudaRuntime,
>
cube_dim: (32, 4, 1)
shared_memory: 12288 bytes
info: KernelId {
    type_id: TypeId {
        t: (
            1277700598496077665,
            13781536268853219741,
        ),
    },
    info: Some (
         (
            CubeDim {
                x: 32,
                y: 4,
                z: 1,
            },
            Config {
                gmm_config: Config {
                    smm_config: Config {
                        tmm_config: Config {
                            plane_dim: 32,
                            lhs_layout: RowMajor,
                            rhs_layout: RowMajor,
                            lhs_line_size: 8,
                            rhs_line_size: 8,
                            out_line_size: 8,
                        },
                        lhs_stage_dim: StageDim {
                            tile_size_x: 16,
                            tile_size_y: 16,
                            num_tiles_x: 4,
                            num_tiles_y: 2,
                            num_tiles_per_buffer: 4,
                        },
                        rhs_stage_dim: StageDim {
                            tile_size_x: 16,
                            tile_size_y: 16,
                            num_tiles_x: 2,
                            num_tiles_y: 4,
                            num_tiles_per_buffer: 4,
                        },
                        out_stage_dim: StageDim {
                            tile_size_x: 16,
                            tile_size_y: 16,
                            num_tiles_x: 4,
                            num_tiles_y: 4,
                            num_tiles_per_buffer: 0,
                        },
                        num_planes: 4,
                        lhs_tiling_order: ColMajor,
                        rhs_tiling_order: RowMajor,
                    },
                    check_m_bounds: false,
                    check_n_bounds: false,
                    lhs_layout: RowMajor,
                    rhs_layout: RowMajor,
                    lhs_line_size: 8,
                    rhs_line_size: 8,
                    out_line_size: 8,
                },
                cube_count: (
                    2,
                    2,
                    3,
                ),
                _c: PhantomData<cubecl_linalg: : matmul: : components: : batch: : cube_dispatch: : NaturalDispatch>,
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

struct __align__(16) __half2_4 {
    __half2 i_0;
    __half2 i_1;
    __half2 i_2;
    __half2 i_3;
};

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

struct __align__(16) __half_8 {
    __half i_0;
    __half i_1;
    __half i_2;
    __half i_3;
    __half i_4;
    __half i_5;
    __half i_6;
    __half i_7;
};


extern "C" __global__ void kernel(
float_8 input_0[],float_8 input_1[],float_8 output_0[],uint info[]
) {
__shared__ __half_8 shared_memory_0[256];
__shared__ __half_8 shared_memory_1[256];
__shared__ float_8 shared_memory_2[128];
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
uint l_0_23;
uint l_0_24;
uint l_0_25;
uint l_0_26;
uint l_0_27;
uint l_0_28;
uint l_0_29;
uint l_0_30;
uint l_0_31;
uint l_0_32;
uint l_0_33;
uint l_0_34;
uint l_0_35;
uint l_0_36;
uint l_0_37;
uint l_0_38;
bool l_0_39;
bool l_0_40;
float_8 l_0_41;
float_8 l_0_42;
float_8 l_0_43;
__half_8 l_0_44;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_0_0;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_1_0;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_2_0;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_3_0;
l_0_0 = blockIdx.x * uint(64);
l_0_1 = blockIdx.y * uint(64);
l_0_2 = info[uint(6)];
l_0_2 = l_0_2 - uint(1);
l_0_3 = info[info[uint(9)] + l_0_2];
nvcuda::wmma::fill_fragment(frag_0_0, float(0.0));
l_0_2 = uint(16);
nvcuda::wmma::fill_fragment(frag_1_0, float(0.0));
l_0_4 = uint(16);
nvcuda::wmma::fill_fragment(frag_2_0, float(0.0));
l_0_5 = uint(16);
nvcuda::wmma::fill_fragment(frag_3_0, float(0.0));
l_0_6 = uint(16);
l_0_7 = info[uint(6)];
l_0_8 = l_0_7 - uint(2);
l_0_9 = info[info[uint(12)] + l_0_8];
l_0_8 = l_0_7 - uint(1);
l_0_10 = info[info[uint(12)] + l_0_8];
l_0_8 = l_0_7 - uint(2);
l_0_11 = info[info[uint(9)] + l_0_8];
l_0_8 = l_0_7 - uint(1);
l_0_12 = info[info[uint(9)] + l_0_8];
l_0_8 = l_0_7 - uint(3);
l_0_7 = info[info[uint(12)] + l_0_8];
l_0_8 = l_0_0;
l_0_13 = uint(0);
l_0_7 = blockIdx.z * l_0_7;
l_0_14 = info[uint(7)];
l_0_15 = l_0_14 - uint(2);
l_0_16 = info[info[uint(13)] + l_0_15];
l_0_15 = l_0_14 - uint(1);
l_0_17 = info[info[uint(13)] + l_0_15];
l_0_15 = l_0_14 - uint(2);
l_0_18 = info[info[uint(10)] + l_0_15];
l_0_15 = l_0_14 - uint(1);
l_0_19 = info[info[uint(10)] + l_0_15];
l_0_15 = l_0_14 - uint(3);
l_0_14 = info[info[uint(13)] + l_0_15];
l_0_15 = uint(0);
l_0_20 = l_0_1;
l_0_14 = blockIdx.z * l_0_14;
l_0_21 = info[uint(8)];
l_0_22 = l_0_21 - uint(2);
l_0_23 = info[info[uint(14)] + l_0_22];
l_0_22 = l_0_21 - uint(1);
l_0_24 = info[info[uint(14)] + l_0_22];
l_0_22 = l_0_21 - uint(2);
l_0_25 = info[info[uint(11)] + l_0_22];
l_0_22 = l_0_21 - uint(1);
l_0_26 = info[info[uint(11)] + l_0_22];
l_0_22 = l_0_21 - uint(3);
l_0_21 = info[info[uint(14)] + l_0_22];
l_0_22 = blockIdx.z * l_0_21;
l_0_21 = l_0_3 - uint(0);
l_0_21 = l_0_21 + uint(32);
l_0_21 = l_0_21 - uint(1);
l_0_21 = l_0_21 / uint(32);

for (uint l_1_0 = uint(0); l_1_0 < l_0_21; ++l_1_0) {
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_0_1;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_1_1;
l_0_27 = threadIdx.y * uint(32);
l_0_27 = l_0_27 + threadIdx.x;
l_0_27 = l_0_27 * uint(8);

for (uint l_2_0 = uint(0); l_2_0 < uint(2); ++l_2_0) {
l_0_28 = l_2_0 * uint(1024);
l_0_28 = l_0_27 + l_0_28;
l_0_29 = l_0_28 / uint(256);
l_0_30 = l_0_28 % uint(256);
l_0_31 = l_0_29 % uint(4);
l_0_32 = l_0_29 / uint(4);
l_0_33 = l_0_31 * uint(16);
l_0_33 = l_0_33 + l_0_8;
l_0_34 = l_0_32 * uint(16);
l_0_34 = l_0_34 + l_0_13;
l_0_35 = l_0_30 / uint(16);
l_0_36 = l_0_30 % uint(16);
l_0_33 = l_0_33 + l_0_35;
l_0_34 = l_0_34 + l_0_36;
l_0_37 = l_0_33 * l_0_9;
l_0_38 = l_0_34 * l_0_10;
l_0_38 = l_0_37 + l_0_38;
l_0_38 = l_0_38 + l_0_7;
l_0_38 = l_0_38 / uint(8);
l_0_39 = l_0_33 < l_0_11;
l_0_40 = l_0_34 < l_0_12;
l_0_40 = l_0_39 && l_0_40;
l_0_41 = input_0[l_0_38];
l_0_43 = float_8{
float(0.0),float(0.0),float(0.0),float(0.0),float(0.0),float(0.0),float(0.0),float(0.0),};
l_0_43 = float_8 {
(l_0_40) ? l_0_41.i_0 : l_0_43.i_0,
(l_0_40) ? l_0_41.i_1 : l_0_43.i_1,
(l_0_40) ? l_0_41.i_2 : l_0_43.i_2,
(l_0_40) ? l_0_41.i_3 : l_0_43.i_3,
(l_0_40) ? l_0_41.i_4 : l_0_43.i_4,
(l_0_40) ? l_0_41.i_5 : l_0_43.i_5,
(l_0_40) ? l_0_41.i_6 : l_0_43.i_6,
(l_0_40) ? l_0_41.i_7 : l_0_43.i_7,
};
l_0_38 = l_0_28 / uint(8);
l_0_44 = __half_8{
__half(l_0_43.i_0),__half(l_0_43.i_1),__half(l_0_43.i_2),__half(l_0_43.i_3),__half(l_0_43.i_4),__half(l_0_43.i_5),__half(l_0_43.i_6),__half(l_0_43.i_7),};
shared_memory_0[l_0_38] = l_0_44;
}
l_0_38 = threadIdx.y * uint(32);
l_0_38 = l_0_38 + threadIdx.x;
l_0_38 = l_0_38 * uint(8);

for (uint l_2_0 = uint(0); l_2_0 < uint(2); ++l_2_0) {
l_0_37 = l_2_0 * uint(1024);
l_0_37 = l_0_38 + l_0_37;
l_0_36 = l_0_37 / uint(256);
l_0_35 = l_0_37 % uint(256);
l_0_34 = l_0_36 / uint(4);
l_0_33 = l_0_36 % uint(4);
l_0_32 = l_0_34 * uint(16);
l_0_32 = l_0_32 + l_0_15;
l_0_31 = l_0_33 * uint(16);
l_0_31 = l_0_31 + l_0_20;
l_0_30 = l_0_35 / uint(16);
l_0_29 = l_0_35 % uint(16);
l_0_32 = l_0_32 + l_0_30;
l_0_31 = l_0_31 + l_0_29;
l_0_28 = l_0_32 * l_0_16;
l_0_27 = l_0_31 * l_0_17;
l_0_28 = l_0_28 + l_0_27;
l_0_28 = l_0_28 + l_0_14;
l_0_28 = l_0_28 / uint(8);
l_0_40 = l_0_32 < l_0_18;
l_0_39 = l_0_31 < l_0_19;
l_0_40 = l_0_40 && l_0_39;
l_0_43 = input_1[l_0_28];
l_0_41 = float_8{
float(0.0),float(0.0),float(0.0),float(0.0),float(0.0),float(0.0),float(0.0),float(0.0),};
l_0_43 = float_8 {
(l_0_40) ? l_0_43.i_0 : l_0_41.i_0,
(l_0_40) ? l_0_43.i_1 : l_0_41.i_1,
(l_0_40) ? l_0_43.i_2 : l_0_41.i_2,
(l_0_40) ? l_0_43.i_3 : l_0_41.i_3,
(l_0_40) ? l_0_43.i_4 : l_0_41.i_4,
(l_0_40) ? l_0_43.i_5 : l_0_41.i_5,
(l_0_40) ? l_0_43.i_6 : l_0_41.i_6,
(l_0_40) ? l_0_43.i_7 : l_0_41.i_7,
};
l_0_37 = l_0_37 / uint(8);
l_0_44 = __half_8{
__half(l_0_43.i_0),__half(l_0_43.i_1),__half(l_0_43.i_2),__half(l_0_43.i_3),__half(l_0_43.i_4),__half(l_0_43.i_5),__half(l_0_43.i_6),__half(l_0_43.i_7),};
shared_memory_1[l_0_37] = l_0_44;
}
__syncthreads();
l_0_38 = uint(16);
l_0_37 = uint(16);
l_0_36 = uint(0) * uint(4);
l_0_36 = l_0_36 + threadIdx.y;
l_0_36 = l_0_36 * uint(32);
l_0_35 = l_0_36 + uint(32);
const uint slice_1_0_length = l_0_35 - l_0_36;
__half_8 *slice_1_0 = shared_memory_0 + l_0_36;
nvcuda::wmma::load_matrix_sync(frag_0_1, reinterpret_cast<__half *>(slice_1_0), l_0_38);
l_0_36 = uint(0) * uint(4);
l_0_36 = l_0_36 + uint(0);
l_0_36 = l_0_36 * uint(32);
l_0_35 = l_0_36 + uint(32);
const uint slice_1_1_length = l_0_35 - l_0_36;
__half_8 *slice_1_1 = shared_memory_1 + l_0_36;
nvcuda::wmma::load_matrix_sync(frag_1_1, reinterpret_cast<__half *>(slice_1_1), l_0_37);
nvcuda::wmma::mma_sync(frag_0_0, frag_0_1, frag_1_1, frag_0_0);
l_0_36 = uint(0) * uint(4);
l_0_36 = l_0_36 + uint(1);
l_0_36 = l_0_36 * uint(32);
l_0_35 = l_0_36 + uint(32);
const uint slice_1_2_length = l_0_35 - l_0_36;
__half_8 *slice_1_2 = shared_memory_1 + l_0_36;
nvcuda::wmma::load_matrix_sync(frag_1_1, reinterpret_cast<__half *>(slice_1_2), l_0_37);
nvcuda::wmma::mma_sync(frag_1_0, frag_0_1, frag_1_1, frag_1_0);
l_0_36 = uint(0) * uint(4);
l_0_36 = l_0_36 + uint(2);
l_0_36 = l_0_36 * uint(32);
l_0_35 = l_0_36 + uint(32);
const uint slice_1_3_length = l_0_35 - l_0_36;
__half_8 *slice_1_3 = shared_memory_1 + l_0_36;
nvcuda::wmma::load_matrix_sync(frag_1_1, reinterpret_cast<__half *>(slice_1_3), l_0_37);
nvcuda::wmma::mma_sync(frag_2_0, frag_0_1, frag_1_1, frag_2_0);
l_0_36 = uint(0) * uint(4);
l_0_36 = l_0_36 + uint(3);
l_0_36 = l_0_36 * uint(32);
l_0_35 = l_0_36 + uint(32);
const uint slice_1_4_length = l_0_35 - l_0_36;
__half_8 *slice_1_4 = shared_memory_1 + l_0_36;
nvcuda::wmma::load_matrix_sync(frag_1_1, reinterpret_cast<__half *>(slice_1_4), l_0_37);
nvcuda::wmma::mma_sync(frag_3_0, frag_0_1, frag_1_1, frag_3_0);
l_0_36 = uint(1) * uint(4);
l_0_36 = l_0_36 + threadIdx.y;
l_0_36 = l_0_36 * uint(32);
l_0_35 = l_0_36 + uint(32);
const uint slice_1_5_length = l_0_35 - l_0_36;
__half_8 *slice_1_5 = shared_memory_0 + l_0_36;
nvcuda::wmma::load_matrix_sync(frag_0_1, reinterpret_cast<__half *>(slice_1_5), l_0_38);
l_0_36 = uint(1) * uint(4);
l_0_36 = l_0_36 + uint(0);
l_0_36 = l_0_36 * uint(32);
l_0_35 = l_0_36 + uint(32);
const uint slice_1_6_length = l_0_35 - l_0_36;
__half_8 *slice_1_6 = shared_memory_1 + l_0_36;
nvcuda::wmma::load_matrix_sync(frag_1_1, reinterpret_cast<__half *>(slice_1_6), l_0_37);
nvcuda::wmma::mma_sync(frag_0_0, frag_0_1, frag_1_1, frag_0_0);
l_0_36 = uint(1) * uint(4);
l_0_36 = l_0_36 + uint(1);
l_0_36 = l_0_36 * uint(32);
l_0_35 = l_0_36 + uint(32);
const uint slice_1_7_length = l_0_35 - l_0_36;
__half_8 *slice_1_7 = shared_memory_1 + l_0_36;
nvcuda::wmma::load_matrix_sync(frag_1_1, reinterpret_cast<__half *>(slice_1_7), l_0_37);
nvcuda::wmma::mma_sync(frag_1_0, frag_0_1, frag_1_1, frag_1_0);
l_0_36 = uint(1) * uint(4);
l_0_36 = l_0_36 + uint(2);
l_0_36 = l_0_36 * uint(32);
l_0_35 = l_0_36 + uint(32);
const uint slice_1_8_length = l_0_35 - l_0_36;
__half_8 *slice_1_8 = shared_memory_1 + l_0_36;
nvcuda::wmma::load_matrix_sync(frag_1_1, reinterpret_cast<__half *>(slice_1_8), l_0_37);
nvcuda::wmma::mma_sync(frag_2_0, frag_0_1, frag_1_1, frag_2_0);
l_0_36 = uint(1) * uint(4);
l_0_36 = l_0_36 + uint(3);
l_0_36 = l_0_36 * uint(32);
l_0_35 = l_0_36 + uint(32);
const uint slice_1_9_length = l_0_35 - l_0_36;
__half_8 *slice_1_9 = shared_memory_1 + l_0_36;
nvcuda::wmma::load_matrix_sync(frag_1_1, reinterpret_cast<__half *>(slice_1_9), l_0_37);
nvcuda::wmma::mma_sync(frag_3_0, frag_0_1, frag_1_1, frag_3_0);
__syncthreads();
l_0_13 = l_0_13 + uint(32);
l_0_15 = l_0_15 + uint(32);
}
l_0_38 = uint(32) * threadIdx.y;
l_0_37 = l_0_38 + uint(32);
const uint slice_0_0_length = l_0_37 - l_0_38;
float_8 *slice_0_0 = shared_memory_2 + l_0_38;
nvcuda::wmma::store_matrix_sync(reinterpret_cast<float *>(slice_0_0), frag_0_0, l_0_2, nvcuda::wmma::mem_row_major);

for (uint l_1_0 = uint(0); l_1_0 < uint(1); ++l_1_0) {
l_0_37 = threadIdx.x * uint(8);
l_0_36 = l_1_0 * uint(256);
l_0_37 = l_0_37 + l_0_36;
l_0_36 = l_0_37 / uint(8);
l_0_43 = slice_0_0[l_0_36];
l_0_36 = threadIdx.y * uint(16);
l_0_35 = l_0_37 / uint(16);
l_0_36 = l_0_36 + l_0_35;
l_0_36 = l_0_36 + l_0_0;
l_0_35 = uint(0) * uint(16);
l_0_37 = l_0_37 % uint(16);
l_0_37 = l_0_35 + l_0_37;
l_0_37 = l_0_37 + l_0_1;
l_0_35 = l_0_36 * l_0_23;
l_0_34 = l_0_37 * l_0_24;
l_0_35 = l_0_35 + l_0_34;
l_0_35 = l_0_35 + l_0_22;
l_0_35 = l_0_35 / uint(8);
output_0[l_0_35] = l_0_43;
}
l_0_37 = l_0_38 + uint(32);
const uint slice_0_1_length = l_0_37 - l_0_38;
float_8 *slice_0_1 = shared_memory_2 + l_0_38;
nvcuda::wmma::store_matrix_sync(reinterpret_cast<float *>(slice_0_1), frag_1_0, l_0_4, nvcuda::wmma::mem_row_major);

for (uint l_1_0 = uint(0); l_1_0 < uint(1); ++l_1_0) {
l_0_37 = threadIdx.x * uint(8);
l_0_36 = l_1_0 * uint(256);
l_0_37 = l_0_37 + l_0_36;
l_0_36 = l_0_37 / uint(8);
l_0_43 = slice_0_1[l_0_36];
l_0_36 = threadIdx.y * uint(16);
l_0_35 = l_0_37 / uint(16);
l_0_36 = l_0_36 + l_0_35;
l_0_36 = l_0_36 + l_0_0;
l_0_35 = uint(1) * uint(16);
l_0_37 = l_0_37 % uint(16);
l_0_37 = l_0_35 + l_0_37;
l_0_37 = l_0_37 + l_0_1;
l_0_35 = l_0_36 * l_0_23;
l_0_34 = l_0_37 * l_0_24;
l_0_35 = l_0_35 + l_0_34;
l_0_35 = l_0_35 + l_0_22;
l_0_35 = l_0_35 / uint(8);
output_0[l_0_35] = l_0_43;
}
l_0_37 = l_0_38 + uint(32);
const uint slice_0_2_length = l_0_37 - l_0_38;
float_8 *slice_0_2 = shared_memory_2 + l_0_38;
nvcuda::wmma::store_matrix_sync(reinterpret_cast<float *>(slice_0_2), frag_2_0, l_0_5, nvcuda::wmma::mem_row_major);

for (uint l_1_0 = uint(0); l_1_0 < uint(1); ++l_1_0) {
l_0_37 = threadIdx.x * uint(8);
l_0_36 = l_1_0 * uint(256);
l_0_37 = l_0_37 + l_0_36;
l_0_36 = l_0_37 / uint(8);
l_0_43 = slice_0_2[l_0_36];
l_0_36 = threadIdx.y * uint(16);
l_0_35 = l_0_37 / uint(16);
l_0_36 = l_0_36 + l_0_35;
l_0_36 = l_0_36 + l_0_0;
l_0_35 = uint(2) * uint(16);
l_0_37 = l_0_37 % uint(16);
l_0_37 = l_0_35 + l_0_37;
l_0_37 = l_0_37 + l_0_1;
l_0_35 = l_0_36 * l_0_23;
l_0_34 = l_0_37 * l_0_24;
l_0_35 = l_0_35 + l_0_34;
l_0_35 = l_0_35 + l_0_22;
l_0_35 = l_0_35 / uint(8);
output_0[l_0_35] = l_0_43;
}
l_0_37 = l_0_38 + uint(32);
const uint slice_0_3_length = l_0_37 - l_0_38;
float_8 *slice_0_3 = shared_memory_2 + l_0_38;
nvcuda::wmma::store_matrix_sync(reinterpret_cast<float *>(slice_0_3), frag_3_0, l_0_6, nvcuda::wmma::mem_row_major);

for (uint l_1_0 = uint(0); l_1_0 < uint(1); ++l_1_0) {
l_0_37 = threadIdx.x * uint(8);
l_0_36 = l_1_0 * uint(256);
l_0_37 = l_0_37 + l_0_36;
l_0_36 = l_0_37 / uint(8);
l_0_43 = slice_0_3[l_0_36];
l_0_36 = threadIdx.y * uint(16);
l_0_35 = l_0_37 / uint(16);
l_0_36 = l_0_36 + l_0_35;
l_0_36 = l_0_36 + l_0_0;
l_0_35 = uint(3) * uint(16);
l_0_37 = l_0_37 % uint(16);
l_0_37 = l_0_35 + l_0_37;
l_0_37 = l_0_37 + l_0_1;
l_0_35 = l_0_36 * l_0_23;
l_0_34 = l_0_37 * l_0_24;
l_0_35 = l_0_35 + l_0_34;
l_0_35 = l_0_35 + l_0_22;
l_0_35 = l_0_35 / uint(8);
output_0[l_0_35] = l_0_43;
}

}
```
[END_KERNEL_COMPILATION]

