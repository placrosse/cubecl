use std::fmt::Display;

use crate::{self as cubecl, as_type};

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

#[track_caller]
pub(crate) fn assert_equals_approx<
    R: Runtime,
    F: Float + num_traits::Float + CubeElement + Display,
>(
    client: &ComputeClient<R::Server, R::Channel>,
    output: Handle,
    expected: &[F],
    epsilon: f32,
) {
    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);

    // normalize to type epsilon
    let epsilon = (epsilon / f32::EPSILON * F::EPSILON.to_f32().unwrap()).max(epsilon);

    for (i, (a, e)) in actual[0..expected.len()]
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        // account for lower precision at higher values
        let allowed_error = F::new((epsilon * e.to_f32().unwrap()).max(epsilon));
        assert!(
            (*a - *e).abs() < allowed_error || (a.is_nan() && e.is_nan()),
            "Values differ more than epsilon: actual={}, expected={}, difference={}, epsilon={}
index: {}
actual: {:?}
expected: {:?}",
            a,
            e,
            (*a - *e).abs(),
            epsilon,
            i,
            actual,
            expected
        );
    }
}

macro_rules! test_binary_impl {
    (
        $test_name:ident,
        $float_type:ident,
        $binary_func:expr,
        [$({
            input_line_size: $input_line_size:expr,
            out_line_size: $out_line_size:expr,
            lhs: $lhs:expr,
            rhs: $rhs:expr,
            expected: $expected:expr
        }),*]) => {
        pub fn $test_name<R: Runtime, $float_type: Float + num_traits::Float + CubeElement + Display>(client: ComputeClient<R::Server, R::Channel>) {
            #[cube(launch_unchecked)]
            fn test_function<$float_type: Float>(lhs: &Array<$float_type>, rhs: &Array<$float_type>, output: &mut Array<$float_type>) {
                if ABSOLUTE_POS < rhs.len() {
                    output[ABSOLUTE_POS] = $binary_func(lhs[ABSOLUTE_POS], rhs[ABSOLUTE_POS]);
                }
            }

            $(
            {
                let lhs = $lhs;
                let rhs = $rhs;
                let output_handle = client.empty($expected.len() * core::mem::size_of::<$float_type>());
                let lhs_handle = client.create($float_type::as_bytes(lhs));
                let rhs_handle = client.create($float_type::as_bytes(rhs));

                unsafe {
                    test_function::launch_unchecked::<$float_type, R>(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new((lhs.len() / $input_line_size as usize) as u32, 1, 1),
                        ArrayArg::from_raw_parts::<$float_type>(&lhs_handle, lhs.len(), $input_line_size),
                        ArrayArg::from_raw_parts::<$float_type>(&rhs_handle, rhs.len(), $input_line_size),
                        ArrayArg::from_raw_parts::<$float_type>(&output_handle, $expected.len(), $out_line_size),
                    )
                };

                assert_equals_approx::<R, F>(&client, output_handle, $expected, 0.001);
            }
            )*
        }
    };
}

test_binary_impl!(
    test_dot,
    F,
    F::dot,
    [
        {
            input_line_size: 1,
            out_line_size: 1,
            lhs: as_type![F: 1., -3.1, -2.4, 15.1],
            rhs: as_type![F: -1., 23.1, -1.4, 5.1],
            expected: as_type![F: -1.0, -71.61, 3.36, 77.01]
        },
        {
            input_line_size: 2,
            out_line_size: 1,
            lhs: as_type![F: 1., -3.1, -2.4, 15.1],
            rhs: as_type![F: -1., 23.1, -1.4, 5.1],
            expected: as_type![F: -72.61, 80.37]
        },
        {
            input_line_size: 4,
            out_line_size: 1,
            lhs: as_type![F: 1., -3.1, -2.4, 15.1],
            rhs: as_type![F: -1., 23.1, -1.4, 5.1],
            expected: as_type![F: 7.76]
        },
        {
            input_line_size: 4,
            out_line_size: 1,
            lhs: as_type![F: 1., -3.1, -2.4, 15.1, -1., 23.1, -1.4, 5.1],
            rhs: as_type![F: -1., 23.1, -1.4, 5.1, 1., -3.1, -2.4, 15.1],
            expected: as_type![F: 7.76, 7.76]
        }

    ]
);

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_binary {
    () => {
        mod binary {
            use super::*;

            macro_rules! add_test {
                ($test_name:ident) => {
                    #[test]
                    fn $test_name() {
                        let client = TestRuntime::client(&Default::default());
                        cubecl_core::runtime_tests::binary::$test_name::<TestRuntime, FloatType>(
                            client,
                        );
                    }
                };
            }

            add_test!(test_dot);
        }
    };
}
