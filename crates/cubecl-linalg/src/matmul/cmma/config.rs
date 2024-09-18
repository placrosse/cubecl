use cubecl_core::prelude::*;

// It is assumed that CMMA uses 32 units to compute 16x16x16 tiles
// TODO put it in config and split tile size into three different parameters
// TODO add number of smem banks
pub(crate) const CMMA_COOP_DIM: usize = 32;
pub(crate) const CMMA_TILE_SIZE: usize = 16;

#[derive(PartialEq, Eq, Clone, Copy)]
/// Defines how data travels from accumulators to global output
pub enum WriteOutStrategy {
    /// Accumulators for one warp are put concurrently in a shared memory large enough to contain them all
    LargeSmem,
    /// Accumulators for one warp are put sequentially in a shared memory with only one reusable spot
    ReuseSmem,
}

impl From<WriteOutStrategy> for bool {
    fn from(value: WriteOutStrategy) -> Self {
        match value {
            WriteOutStrategy::LargeSmem => false,
            WriteOutStrategy::ReuseSmem => true,
        }
    }
}

/// How cubes are dispatched in the hypercube
/// Should impact L2 cache reuse
#[derive(Clone, Copy)]
pub enum CubeDispatchStrategy {
    /// Cubes are dispatched row major
    RowMajor,
    /// Cubes are dispatched col major
    ColMajor,
    /// Cubes follow swizzle pattern, see https://bruce-lee-ly.medium.com/nvidia-tensor-core-cuda-hgemm-advanced-optimization-5a17eb77dd85
    Swizzle,
}

impl From<CubeDispatchStrategy> for u32 {
    fn from(value: CubeDispatchStrategy) -> Self {
        match value {
            CubeDispatchStrategy::RowMajor => 0,
            CubeDispatchStrategy::ColMajor => 1,
            CubeDispatchStrategy::Swizzle => 2,
        }
    }
}

pub struct CmmaConfig {
    /// b_m / tile_size
    pub num_compute_warps: usize,
    /// b_k / tile_size
    pub num_buffers: usize,
    /// b_n / tile_size
    pub num_accumulators: usize,
    /// Whether to unroll loop over k within the shared memory
    pub unroll: bool,
    /// Whether to write all accumulators in different spots of a large shared memory or reuse the space
    pub write_out_strategy: WriteOutStrategy,
    /// Order in which to dispatch cubes
    pub cube_dispatch: CubeDispatchStrategy,
}

impl Default for CmmaConfig {
    fn default() -> Self {
        Self {
            num_compute_warps: 8,
            num_buffers: 1,
            num_accumulators: 8,
            unroll: false,
            write_out_strategy: WriteOutStrategy::ReuseSmem,
            cube_dispatch: CubeDispatchStrategy::Swizzle,
        }
    }
}

impl CmmaConfig {
    pub(crate) fn comptime_info(&self, m: usize, k: usize, n: usize) -> ComptimeCmmaInfo {
        let b_m = self.num_compute_warps * CMMA_TILE_SIZE;
        let b_k = self.num_buffers * CMMA_TILE_SIZE;
        let b_n = self.num_accumulators * CMMA_TILE_SIZE;

        ComptimeCmmaInfo {
            block_size_m: b_m as u32,
            block_size_k: b_k as u32,
            block_size_n: b_n as u32,
            tile_size: CMMA_TILE_SIZE as u32,
            unroll: self.unroll,
            check_m_bounds: m % b_m != 0,
            check_k_bounds: k % b_k != 0,
            check_n_bounds: n % b_n != 0,
            coop_dim: CMMA_COOP_DIM as u32,
            num_coops: self.num_compute_warps as u32,
            num_accumulators: self.num_accumulators as u32,
            write_out_reuse_smem: self.write_out_strategy.into(),
            cube_dispatch: self.cube_dispatch.into(),
        }
    }

    pub(crate) fn cube_count<R: Runtime>(
        &self,
        output_shape: &[usize],
    ) -> CubeCount<<R as Runtime>::Server> {
        let rank = output_shape.len();
        let num_rows = *output_shape.get(rank - 2).unwrap();
        let num_cols = *output_shape.get(rank - 1).unwrap();

        let cubes_x =
            f32::ceil(num_rows as f32 / (self.num_compute_warps * CMMA_TILE_SIZE) as f32) as u32;
        let cubes_y =
            f32::ceil(num_cols as f32 / (self.num_accumulators * CMMA_TILE_SIZE) as f32) as u32;

        let mut num_iter = 1;
        for shape in output_shape.iter().take(rank - 2) {
            num_iter *= shape;
        }

        CubeCount::Static(cubes_x, cubes_y, num_iter as u32)
    }

    pub(crate) fn cube_dim(&self) -> CubeDim {
        CubeDim {
            x: CMMA_COOP_DIM as u32,
            y: self.num_compute_warps as u32,
            z: 1,
        }
    }

    pub(crate) fn available_vectorizations(&self) -> Vec<u8> {
        let vectorizations = vec![8, 4, 2];
        for v in vectorizations.iter() {
            assert!(CMMA_TILE_SIZE * CMMA_TILE_SIZE % (*v as usize * CMMA_COOP_DIM) == 0);
        }
        vectorizations
    }
}

impl Init for ComptimeCmmaInfo {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct ComptimeCmmaInfo {
    /// Block size along dimension of lhs
    pub block_size_m: u32,
    /// Block size along common dimension
    pub block_size_k: u32,
    /// Block size along dimension of rhs
    pub block_size_n: u32,
    /// Tile size (dimension of one side). Should correspond to cmma supported tile size
    pub tile_size: u32,
    /// Bounds must be checked on lhs dimension
    pub check_m_bounds: bool,
    /// Bounds must be checked on common dimension
    pub check_k_bounds: bool,
    /// Bounds must be checked on rhs dimension
    pub check_n_bounds: bool,
    /// Unroll
    pub unroll: bool,
    /// The number of units that can collaborate
    pub coop_dim: u32,
    /// The number of collaboration groups
    pub num_coops: u32,
    /// Number of cmma per subcube performed in one pass
    pub num_accumulators: u32,
    /// Write out strategy: false = large, true = reuse
    pub write_out_reuse_smem: bool,
    /// 0 = RowMajor, 1 = ColMajor, 2 = Swizzle
    pub cube_dispatch: u32,
}
