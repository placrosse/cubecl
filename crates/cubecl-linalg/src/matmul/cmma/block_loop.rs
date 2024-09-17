use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::{
    base::{make_cmma_matrices, make_shared_memories},
    write_output::smem_store::AddStore,
};

use super::{
    base::RuntimeCmmaInfo,
    compute_loop::compute_loop,
    config::ComptimeCmmaInfo,
    load_shared_memory::load_to_shared_memories,
    write_output::{
        base::OutputWriter,
        large_smem::LargeSmemWriter,
        reuse_smem::ReuseSmemWriter,
        smem_store::{OverrideStore, SmemStore},
    },
};

#[cube]
pub(crate) trait BlockLoop {
    fn block_loop<F: Float, FC: Float>(
        lhs: &Tensor<F>,
        rhs: &Tensor<F>,
        out: &mut Tensor<F>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    );
}

pub(crate) struct SingleBufferLoop {}
pub(crate) struct DoubleBufferLoop {}

#[cube]
impl BlockLoop for SingleBufferLoop {
    fn block_loop<F: Float, FC: Float>(
        lhs: &Tensor<F>,
        rhs: &Tensor<F>,
        out: &mut Tensor<F>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) {
        let block_size_k = comptime_info.block_size_k;

        let shared_memories = make_shared_memories::<FC>(comptime_info);
        let mut cmma_matrices = make_cmma_matrices::<F, FC>(comptime_info);

        // Equals ceil(dims.k / block_size_k)
        let dims = runtime_info.dims;
        let num_loops = (dims.k + block_size_k - 1) / block_size_k;

        for block in 0..num_loops {
            let k_offset = block * block_size_k;

            load_to_shared_memories::<F, FC>(
                lhs,
                rhs,
                k_offset,
                shared_memories,
                runtime_info,
                comptime_info,
            );

            sync_units();

            compute_loop::<F, FC>(
                shared_memories,
                &mut cmma_matrices,
                runtime_info.ids,
                comptime_info,
            );

            sync_units();
        }

        write_out::<F, OverrideStore>(out, cmma_matrices.accumulators, runtime_info, comptime_info);
    }
}

#[cube]
impl BlockLoop for DoubleBufferLoop {
    fn block_loop<F: Float, FC: Float>(
        lhs: &Tensor<F>,
        rhs: &Tensor<F>,
        out: &mut Tensor<F>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) {
        let block_size_k = comptime_info.block_size_k;

        let shared_memories_0 = make_shared_memories::<FC>(comptime_info);
        let shared_memories_1 = make_shared_memories::<FC>(comptime_info);
        let mut cmma_matrices_0 = make_cmma_matrices::<F, FC>(comptime_info);
        let mut cmma_matrices_1 = make_cmma_matrices::<F, FC>(comptime_info);

        // Equals ceil(dims.k / block_size_k)
        let dims = runtime_info.dims;
        let num_blocks = (dims.k + block_size_k - 1) / block_size_k;
        let num_loops = (num_blocks + 1) / 2;

        let ids = runtime_info.ids;
        if ids.team == 0 {
            for iteration in 0..num_loops {
                let k_offset_0 = (iteration * 2) * block_size_k;

                load_to_shared_memories::<F, FC>(
                    lhs,
                    rhs,
                    k_offset_0,
                    shared_memories_0,
                    runtime_info,
                    comptime_info,
                );

                sync_units();

                compute_loop::<F, FC>(
                    shared_memories_0,
                    &mut cmma_matrices_0,
                    runtime_info.ids,
                    comptime_info,
                );

                sync_units();
            }

            write_out::<F, OverrideStore>(
                out,
                cmma_matrices_0.accumulators,
                runtime_info,
                comptime_info,
            );

            sync_units();
        } else {
            for iteration in 0..num_loops {
                let k_offset = (iteration * 2 + 1) * block_size_k;

                sync_units();

                load_to_shared_memories::<F, FC>(
                    lhs,
                    rhs,
                    k_offset,
                    shared_memories_1,
                    runtime_info,
                    comptime_info,
                );

                sync_units();

                compute_loop::<F, FC>(
                    shared_memories_1,
                    &mut cmma_matrices_1,
                    runtime_info.ids,
                    comptime_info,
                );
            }

            sync_units();

            write_out::<F, AddStore>(
                out,
                cmma_matrices_1.accumulators,
                runtime_info,
                comptime_info,
            );
        }
    }
}

#[cube]
fn write_out<F: Float, S: SmemStore>(
    out: &mut Tensor<F>,
    accumulators: Sequence<cmma::Matrix<F>>,
    runtime_info: RuntimeCmmaInfo,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    if comptime_info.write_out_reuse_smem {
        // Can't use with double buffering
        // TODO pretty ugly that we need to pass S and not use it
        ReuseSmemWriter::write_to_output::<F, OverrideStore>(
            out,
            accumulators,
            runtime_info,
            comptime_info,
        );
    } else {
        LargeSmemWriter::write_to_output::<F, S>(out, accumulators, runtime_info, comptime_info);
    }
}
