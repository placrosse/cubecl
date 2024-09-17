use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma::base::Ids;

#[cube]
pub(crate) trait SmemStore: Send + Sync + 'static {
    fn store<F: Float>(slice: &mut SliceMut<'_, F>, accumulator: &cmma::Matrix<F>, _ids: Ids);
}

pub(crate) struct OverrideStore {}
pub(crate) struct AddStore {}

#[cube]
impl SmemStore for OverrideStore {
    fn store<F: Float>(slice: &mut SliceMut<'_, F>, accumulator: &cmma::Matrix<F>, _ids: Ids) {
        cmma::store::<F>(slice, accumulator, 16, cmma::MatrixLayout::RowMajor);
    }
}

#[cube]
impl SmemStore for AddStore {
    fn store<F: Float>(smem_slice: &mut SliceMut<'_, F>, accumulator: &cmma::Matrix<F>, ids: Ids) {
        let lane_offset = 8 * ids.lane; 
        let mut array = Array::<F>::new(8);

        #[unroll]
        for i in 0..8 {
            array[i] = smem_slice[lane_offset + i] + F::new(5.);
        }

        cmma::store::<F>(smem_slice, accumulator, 16, cmma::MatrixLayout::RowMajor);

        #[unroll]
        for i in 0..8 {
            smem_slice[lane_offset + i] += array[i];
        }
    }
}
