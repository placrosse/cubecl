use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub(crate) trait SmemStore {
    fn store<F: Float>(slice: &mut SliceMut<'_, F>, accumulator: &cmma::Matrix<F>);
}

pub(crate) struct OverrideStore {}
pub(crate) struct AddStore {}

#[cube]
impl SmemStore for OverrideStore {
    fn store<F: Float>(slice: &mut SliceMut<'_, F>, accumulator: &cmma::Matrix<F>) {
        cmma::store::<F>(slice, accumulator, 16, cmma::MatrixLayout::RowMajor);
    }
}

#[cube]
impl SmemStore for AddStore {
    fn store<F: Float>(slice: &mut SliceMut<'_, F>, accumulator: &cmma::Matrix<F>) {
        cmma::store::<F>(slice, accumulator, 16, cmma::MatrixLayout::RowMajor);
    }
}
