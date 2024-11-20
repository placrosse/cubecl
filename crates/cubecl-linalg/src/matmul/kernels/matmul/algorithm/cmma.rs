use std::marker::PhantomData;

use cubecl_core::prelude::*;

use crate::matmul::components::stage::{self, S4x4x2, StageSize};
use crate::matmul::components::tile::accelerated::Accelerated16x16x16;
use crate::matmul::components::tile::Matmul;
use crate::matmul::components::MatmulProblem;
use crate::matmul::components::{batch, global};

use super::base;

pub struct Cmma<EG: Numeric, SS: StageSize = S4x4x2> {
    pub _eg: PhantomData<EG>,
    pub _ss: PhantomData<SS>
}

impl<EG: Numeric, SS: StageSize> base::Algorithm<EG> for Cmma<EG, SS> {
    const PLANE_DIM: u32 = 32;
    type EG = EG;
    type ES = half::f16;
    type EA = f32;

    type TileMatmul = Accelerated16x16x16<Self::ES, Self::EA>;

    type StageMatmul =
        stage::multi_buffer::Matmul<Self::ES, Self::EG, Self::EA, Self::TileMatmul, SS>;

    type GlobalMatmul = global::homogeneous::Matmul<Self::EG, Self::ES, Self::StageMatmul>;

    type BatchMatmul =
        batch::one_to_one::Matmul<Self::EG, Self::ES, Self::GlobalMatmul, batch::SwizzleTransposedDispatch<2>>;

    fn cube_dim() -> CubeDim {
        CubeDim::new(Self::PLANE_DIM, SS::NUM_M, 1)
    }

    fn cube_count(problem: &MatmulProblem) -> CubeCount {
        let m_stage = SS::NUM_M * Self::TileMatmul::M;
        let n_stage = SS::NUM_N * Self::TileMatmul::N;
        let cubes_needed_m = (problem.m as u32 + m_stage - 1) / m_stage;
        let cubes_needed_n = (problem.n as u32 + n_stage - 1) / n_stage;

        CubeCount::Static(cubes_needed_m, cubes_needed_n, problem.num_batches() as u32)
    }
}
