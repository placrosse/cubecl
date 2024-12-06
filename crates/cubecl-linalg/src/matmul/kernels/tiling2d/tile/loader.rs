use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, CubeType};
use std::marker::PhantomData;

use crate::matmul::kernels::tiling2d::{
    config::CubeTiling2dConfig,
    load_shared_memory::{LoadInfo, Loader},
};

use super::{
    block_io::base::BlockLoader,
    memory_access::{MatchingVectorization, UnmatchingVectorization},
};

// Transposed tensor's vectorization must be 1
// Plain tensor's vectorization must equal tile size
pub(crate) struct TileLoader<F: Float> {
    _f: PhantomData<F>,
}

#[derive(CubeType)]
pub(crate) struct LoadIndices {
    pub offset: u32,
    pub gm_stride: u32,
    pub sm_stride: u32,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct CheckBounds {
    pub dim_vertical: u32,
    pub dim_horizontal: u32,
    pub skip_row: u32,
    pub skip_col: u32,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct ReadTileInfo {
    pub read_row: u32,
    pub read_col: u32,
    pub gm_position_base: u32,
    pub sm_position_base: u32,
    pub gm_stride: u32,
    pub sm_stride: u32,
}

#[cube]
impl<F: Float> Loader<F> for TileLoader<F> {
    fn load_lhs_plain<B: BlockLoader<F>>(
        lhs: &Tensor<Line<F>>,
        load_info: LoadInfo<F>,
        #[comptime] config: CubeTiling2dConfig,
    ) {
        let dims = load_info.dims;
        let coordinates = load_info.coordinates;
        let gm_stride = dims.m;

        let load_indices = LoadIndices {
            offset: coordinates.skip_row + load_info.k * gm_stride + load_info.batch_offset,
            gm_stride,
            sm_stride: config.block_size_n,
        };
        let check_bounds = CheckBounds {
            dim_vertical: dims.k,
            dim_horizontal: dims.m,
            skip_row: load_info.k,
            skip_col: coordinates.skip_row,
        };

        load_plain::<F, B>(lhs, load_info, load_indices, check_bounds, config);
    }

    fn load_lhs_transposed<B: BlockLoader<F>>(
        lhs: &Tensor<Line<F>>,
        load_info: LoadInfo<F>,
        #[comptime] config: CubeTiling2dConfig,
    ) {
        let dims = load_info.dims;
        let coordinates = load_info.coordinates;
        let gm_stride = dims.k;

        let load_indices = LoadIndices {
            offset: coordinates.skip_row * gm_stride + load_info.k + load_info.batch_offset,
            gm_stride,
            sm_stride: config.block_size_m,
        };
        let check_bounds = CheckBounds {
            dim_vertical: dims.m,
            dim_horizontal: dims.k,
            skip_row: coordinates.skip_row,
            skip_col: load_info.k,
        };

        load_transposed::<F, B>(lhs, load_info, load_indices, check_bounds, config);
    }

    fn load_rhs_plain<B: BlockLoader<F>>(
        rhs: &Tensor<Line<F>>,
        load_info: LoadInfo<F>,
        #[comptime] config: CubeTiling2dConfig,
    ) {
        let coordinates = load_info.coordinates;
        let dims = load_info.dims;
        let gm_stride = dims.n;

        let load_indices = LoadIndices {
            offset: coordinates.skip_col + load_info.k * gm_stride + load_info.batch_offset,
            gm_stride,
            sm_stride: config.block_size_n,
        };
        let check_bounds = CheckBounds {
            dim_vertical: dims.k,
            dim_horizontal: dims.n,
            skip_row: load_info.k,
            skip_col: coordinates.skip_col,
        };

        load_plain::<F, B>(rhs, load_info, load_indices, check_bounds, config);
    }

    fn load_rhs_transposed<B: BlockLoader<F>>(
        rhs: &Tensor<Line<F>>,
        load_info: LoadInfo<F>,
        #[comptime] config: CubeTiling2dConfig,
    ) {
        let dims = load_info.dims;
        let coordinates = load_info.coordinates;
        let gm_stride = dims.k;

        let load_indices = LoadIndices {
            offset: coordinates.skip_col * gm_stride + load_info.k + load_info.batch_offset,
            gm_stride,
            sm_stride: config.block_size_n,
        };
        let check_bounds = CheckBounds {
            dim_vertical: dims.n,
            dim_horizontal: dims.k,
            skip_row: coordinates.skip_col,
            skip_col: load_info.k,
        };

        load_transposed::<F, B>(rhs, load_info, load_indices, check_bounds, config);
    }
}

#[cube]
pub(crate) fn load_plain<F: Float, L: BlockLoader<F>>(
    tensor: &Tensor<Line<F>>,
    load_info: LoadInfo<F>,
    load_indices: LoadIndices,
    check_bounds: CheckBounds,
    #[comptime] config: CubeTiling2dConfig,
) {
    let coordinates = load_info.coordinates;

    let vectorization = line_size_of(tensor);
    let tile_size = config.tile_size;
    let sm_dim_vertical = config.block_size_k;

    let read_row = coordinates.unit_row;
    let read_col = coordinates.unit_col;
    let write_row = coordinates.unit_row;
    let write_col = coordinates.unit_col;

    let gm_position_base = read_row * load_indices.gm_stride + read_col + load_indices.offset;
    let sm_position_base = write_row * load_indices.sm_stride + write_col;

    let read_tile_info = ReadTileInfo {
        read_row,
        read_col,
        gm_position_base,
        sm_position_base,
        gm_stride: load_indices.gm_stride,
        sm_stride: load_indices.sm_stride,
    };
    let mut sm = load_info.shared_memory;

    if write_row < sm_dim_vertical {
        if vectorization == tile_size {
            L::load_tile_plain::<MatchingVectorization>(
                tensor,
                &mut sm,
                read_tile_info,
                config,
                check_bounds,
            );
        } else {
            L::load_tile_plain::<UnmatchingVectorization>(
                tensor,
                &mut sm,
                read_tile_info,
                config,
                check_bounds,
            );
        }
    }
}

#[cube]
pub(crate) fn load_transposed<F: Float, L: BlockLoader<F>>(
    tensor: &Tensor<Line<F>>,
    load_info: LoadInfo<F>,
    load_indices: LoadIndices,
    check_bounds: CheckBounds,
    #[comptime] config: CubeTiling2dConfig,
) {
    let coordinates = load_info.coordinates;

    let sm_dim_vertical = config.block_size_k;

    let read_row = coordinates.unit_row;
    let read_col = coordinates.unit_col;
    let write_row = coordinates.unit_col;
    let write_col = coordinates.unit_row;

    let gm_position_base = read_row * load_indices.gm_stride + read_col + load_indices.offset;
    let sm_position_base = write_row * load_indices.sm_stride + write_col;

    let read_tile_info = ReadTileInfo {
        read_row,
        read_col,
        gm_position_base,
        sm_position_base,
        gm_stride: load_indices.gm_stride,
        sm_stride: load_indices.sm_stride,
    };
    let mut sm = load_info.shared_memory;

    if write_row < sm_dim_vertical {
        L::load_tile_transposed(tensor, &mut sm, read_tile_info, config, check_bounds);
    }
}
