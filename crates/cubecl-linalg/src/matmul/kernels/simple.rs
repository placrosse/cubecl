//! Naive matmul kernel implementation
//!
//! Each local unit will compute a single element of the output matrix.
use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::tensor::{into_contiguous, matrix_layout, MatrixLayout, TensorHandle};

#[cube(launch_unchecked)]
fn matmul_kernel<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    // number of dimensions not involved in the matmul
    #[comptime] num_batches: Option<u32>,
) {
    let rank = out.rank();
    let end = num_batches.unwrap_or_else(|| rank - 2);
    let unroll = num_batches.is_some();

    let n_rows = lhs.shape(rank - 2);
    let n_cols = rhs.shape(rank - 1);
    let mut k = rhs.shape(rank - 2);

    let batch_pos = ABSOLUTE_POS_Z;
    let row = CUBE_DIM_X * CUBE_POS_X + UNIT_POS_X;
    let col = CUBE_DIM_Y * CUBE_POS_Y + UNIT_POS_Y;

    if row >= n_rows || col >= n_cols {
        return;
    }

    let vectorization_factor = line_size_of(lhs);

    let mut offset_lhs = 0;
    let mut offset_rhs = 0;
    let offset_out = n_rows * n_cols * batch_pos;

    #[unroll(unroll)]
    for i in 0..end {
        let ogwl = offset_out / out.stride(i);

        offset_lhs += ogwl % lhs.shape(i) * lhs.stride(i);
        offset_rhs += ogwl % rhs.shape(i) * rhs.stride(i);
    }

    offset_lhs /= vectorization_factor;
    offset_rhs /= vectorization_factor;

    let mut sum = F::lined(0., vectorization_factor);

    k /= vectorization_factor;

    for i in 0..k {
        let lhs_index = row * k + i + offset_lhs;
        let rhs_index = col * k + i + offset_rhs;

        sum += lhs[lhs_index] * rhs[rhs_index];
    }

    let mut out_index = row * n_cols + col;
    out_index += offset_out;

    let unroll_sum = vectorization_factor != 1;
    if unroll_sum {
        let mut accum = F::new(0.);
        // we unroll the loop to sum `vectorization_factor` elements at once, which lets us
        // use SIMD instructions to speed up the computation
        #[unroll]
        for v in 0..vectorization_factor {
            accum += sum[v];
        }

        out[out_index] = accum;
    } else {
        out[out_index] = sum;
    }
}

/// Matrix multiplication using memory coalescing algorithm with custom cube dimensions
pub fn launch_ref<R: Runtime, E: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
) {
    let lhs =
        TensorHandle::<R, E>::new(lhs.shape.to_vec(), lhs.strides.to_vec(), lhs.handle.clone());
    let rhs =
        TensorHandle::<R, E>::new(rhs.shape.to_vec(), rhs.strides.to_vec(), rhs.handle.clone());

    launch(client, lhs, rhs, out);
}

fn launch<R: Runtime, E: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: TensorHandle<R, E>,
    rhs: TensorHandle<R, E>,
    out: &TensorHandleRef<'_, R>,
) {
    let (cube_dim_x, cube_dim_y) = (32, 8);
    let ndims = lhs.shape.len();
    let dim1 = ndims - 1;
    let dim2 = ndims - 2;

    let lhs_layout = matrix_layout(&lhs.strides);
    let rhs_layout = matrix_layout(&rhs.strides);

    let lhs = if !matches!(lhs_layout, MatrixLayout::Contiguous) {
        into_contiguous::<R, E>(client, &lhs.as_ref())
    } else {
        lhs
    };

    // we swap the dimensions to achieve memory-coalescing:
    // consecutive elements of a column in the original rhs tensor will now be stored
    // consecutively in memory, which allows to fetch them with fewer memory instructions
    let correct_rhs_layout = |mut rhs: TensorHandle<R, E>| {
        let rhs_original_shape = rhs.shape.clone();
        rhs.strides.swap(dim1, dim2);
        rhs.shape.swap(dim1, dim2);

        let rhs = into_contiguous::<R, E>(client, &rhs.as_ref());

        (rhs_original_shape, rhs)
    };

    let (rhs_original_shape, rhs) = match rhs_layout {
        MatrixLayout::Contiguous => correct_rhs_layout(rhs),
        MatrixLayout::MildlyPermuted {
            transposed,
            batch_swap,
        } => {
            if transposed && !batch_swap {
                let rhs_original_shape = rhs.shape.clone();
                (rhs_original_shape, rhs)
            } else {
                correct_rhs_layout(rhs)
            }
        }
        MatrixLayout::HighlyPermuted => correct_rhs_layout(rhs),
    };

    let cube_count = simple_cube_count(
        &lhs.shape,
        &rhs_original_shape,
        out.shape,
        cube_dim_x,
        cube_dim_y,
    );

    let vectorization_factor = match lhs.shape[ndims - 1] % 4 == 0 {
        true => 4,
        false => 1,
    };

    unsafe {
        matmul_kernel::launch_unchecked::<E, R>(
            client,
            cube_count,
            CubeDim::new(cube_dim_x as u32, cube_dim_y as u32, 1),
            lhs.as_arg(vectorization_factor),
            TensorArg::from_raw_parts::<E>(
                &rhs.handle,
                &rhs.strides,
                &rhs_original_shape, // We need the original shape.
                vectorization_factor,
            ),
            out.as_tensor_arg(1),
            Some(ndims as u32 - 2),
        );
    };
}

fn simple_cube_count(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    output_shape: &[usize],
    cube_dim_x: usize,
    cube_dim_y: usize,
) -> CubeCount {
    let ndims = lhs_shape.len();
    let num_rows = lhs_shape[ndims - 2];
    let num_cols = rhs_shape[ndims - 1];

    let cubes_x = f32::ceil(num_rows as f32 / cube_dim_x as f32) as u32;
    let cubes_y = f32::ceil(num_cols as f32 / cube_dim_y as f32) as u32;
    let mut num_iter = 1;

    #[allow(clippy::needless_range_loop)]
    for i in 0..ndims - 2 {
        num_iter *= output_shape[i];
    }

    CubeCount::Static(cubes_x, cubes_y, num_iter as u32)
}
