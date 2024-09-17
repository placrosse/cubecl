use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::{base::RuntimeCmmaInfo, write_output::smem_store::OverrideStore};

use super::{
    super::config::ComptimeCmmaInfo,
    base::{shared_memory_to_output, OutputWriter},
    smem_store::SmemStore,
};

pub(crate) struct ReuseSmemWriter;

#[cube]
impl OutputWriter for ReuseSmemWriter {
    fn make_smem<F: Float>(#[comptime] comptime_info: ComptimeCmmaInfo) -> SharedMemory<F> {
        let tile_size = comptime_info.tile_size;
        let smem_size = comptime_info.num_coops * tile_size * tile_size;

        SharedMemory::<F>::new(smem_size)
    }

    fn write_to_output<F: Float>(
        out: &mut Tensor<F>,
        accumulators: Sequence<cmma::Matrix<F>>,
        acc_sm: &mut SharedMemory<F>,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) {
        let num_accumulators = comptime_info.num_accumulators;
        let tile_size = comptime_info.tile_size;
        let ids = runtime_info.ids;

        let sm_stride = tile_size * tile_size;

        let slice_offset = ids.coop * sm_stride;
        let slice = acc_sm.slice_mut_unsafe(slice_offset, slice_offset + sm_stride);

        #[unroll]
        for n in 0..num_accumulators {
            OverrideStore::store(slice, accumulators.index(n), runtime_info.ids);

            shared_memory_to_output(out, ids.coop, acc_sm, n, runtime_info, comptime_info);
        }
    }
}
