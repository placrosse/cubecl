#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma_matmul {
    () => {
        #[test]
        pub fn test_matmul_cmma_one_cube() {
            tests::matmul_tests::test_matmul_cmma_one_cube::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_several_cubes() {
            tests::matmul_tests::test_matmul_cmma_several_cubes::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_with_check_bounds() {
            tests::matmul_tests::test_matmul_cmma_with_check_bounds::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn test_matmul_cmma_with_batches() {
            tests::matmul_tests::test_matmul_cmma_with_batches::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_unvectorizable_shapes() {
            tests::matmul_tests::test_matmul_cmma_unvectorizable_shapes::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn test_matmul_cmma_vec2_shapes() {
            tests::matmul_tests::test_matmul_cmma_vec2_shapes::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_16_16_16() {
            tests::matmul_tests::test_matmul_cmma_16_16_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_16_16_32() {
            tests::matmul_tests::test_matmul_cmma_16_16_32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_16_16_64() {
            tests::matmul_tests::test_matmul_cmma_16_16_64::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_16_16_128() {
            tests::matmul_tests::test_matmul_cmma_16_16_128::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_16_32_16() {
            tests::matmul_tests::test_matmul_cmma_16_32_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_16_32_32() {
            tests::matmul_tests::test_matmul_cmma_16_32_32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_16_32_64() {
            tests::matmul_tests::test_matmul_cmma_16_32_64::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_16_32_128() {
            tests::matmul_tests::test_matmul_cmma_16_32_128::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_16_64_16() {
            tests::matmul_tests::test_matmul_cmma_16_64_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_16_64_32() {
            tests::matmul_tests::test_matmul_cmma_16_64_32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_16_64_64() {
            tests::matmul_tests::test_matmul_cmma_16_64_64::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_16_64_128() {
            tests::matmul_tests::test_matmul_cmma_16_64_128::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_16_128_16() {
            tests::matmul_tests::test_matmul_cmma_16_128_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_16_128_32() {
            tests::matmul_tests::test_matmul_cmma_16_128_32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_16_128_64() {
            tests::matmul_tests::test_matmul_cmma_16_128_64::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_16_128_128() {
            tests::matmul_tests::test_matmul_cmma_16_128_128::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_32_16_16() {
            tests::matmul_tests::test_matmul_cmma_32_16_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_32_16_32() {
            tests::matmul_tests::test_matmul_cmma_32_16_32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_32_16_64() {
            tests::matmul_tests::test_matmul_cmma_32_16_64::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_32_16_128() {
            tests::matmul_tests::test_matmul_cmma_32_16_128::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_32_32_16() {
            tests::matmul_tests::test_matmul_cmma_32_32_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_32_32_32() {
            tests::matmul_tests::test_matmul_cmma_32_32_32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_32_32_64() {
            tests::matmul_tests::test_matmul_cmma_32_32_64::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_32_32_128() {
            tests::matmul_tests::test_matmul_cmma_32_32_128::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_32_64_16() {
            tests::matmul_tests::test_matmul_cmma_32_64_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_32_64_32() {
            tests::matmul_tests::test_matmul_cmma_32_64_32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_32_64_64() {
            tests::matmul_tests::test_matmul_cmma_32_64_64::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_32_64_128() {
            tests::matmul_tests::test_matmul_cmma_32_64_128::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_32_128_16() {
            tests::matmul_tests::test_matmul_cmma_32_128_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_32_128_32() {
            tests::matmul_tests::test_matmul_cmma_32_128_32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_32_128_64() {
            tests::matmul_tests::test_matmul_cmma_32_128_64::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_32_128_128() {
            tests::matmul_tests::test_matmul_cmma_32_128_128::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_64_16_16() {
            tests::matmul_tests::test_matmul_cmma_64_16_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_64_16_32() {
            tests::matmul_tests::test_matmul_cmma_64_16_32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_64_16_64() {
            tests::matmul_tests::test_matmul_cmma_64_16_64::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_64_16_128() {
            tests::matmul_tests::test_matmul_cmma_64_16_128::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_64_32_16() {
            tests::matmul_tests::test_matmul_cmma_64_32_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_64_32_32() {
            tests::matmul_tests::test_matmul_cmma_64_32_32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_64_32_64() {
            tests::matmul_tests::test_matmul_cmma_64_32_64::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_64_32_128() {
            tests::matmul_tests::test_matmul_cmma_64_32_128::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_64_64_16() {
            tests::matmul_tests::test_matmul_cmma_64_64_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_64_64_32() {
            tests::matmul_tests::test_matmul_cmma_64_64_32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_64_64_64() {
            tests::matmul_tests::test_matmul_cmma_64_64_64::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_64_64_128() {
            tests::matmul_tests::test_matmul_cmma_64_64_128::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_64_128_16() {
            tests::matmul_tests::test_matmul_cmma_64_128_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_64_128_32() {
            tests::matmul_tests::test_matmul_cmma_64_128_32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_64_128_64() {
            tests::matmul_tests::test_matmul_cmma_64_128_64::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_64_128_128() {
            tests::matmul_tests::test_matmul_cmma_64_128_128::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_128_16_16() {
            tests::matmul_tests::test_matmul_cmma_128_16_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_128_16_32() {
            tests::matmul_tests::test_matmul_cmma_128_16_32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_128_16_64() {
            tests::matmul_tests::test_matmul_cmma_128_16_64::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_128_16_128() {
            tests::matmul_tests::test_matmul_cmma_128_16_128::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_128_32_16() {
            tests::matmul_tests::test_matmul_cmma_128_32_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_128_32_32() {
            tests::matmul_tests::test_matmul_cmma_128_32_32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_128_32_64() {
            tests::matmul_tests::test_matmul_cmma_128_32_64::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_128_32_128() {
            tests::matmul_tests::test_matmul_cmma_128_32_128::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_128_64_16() {
            tests::matmul_tests::test_matmul_cmma_128_64_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_128_64_32() {
            tests::matmul_tests::test_matmul_cmma_128_64_32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_128_64_64() {
            tests::matmul_tests::test_matmul_cmma_128_64_64::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_128_64_128() {
            tests::matmul_tests::test_matmul_cmma_128_64_128::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_128_128_16() {
            tests::matmul_tests::test_matmul_cmma_128_128_16::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_128_128_32() {
            tests::matmul_tests::test_matmul_cmma_128_128_32::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_128_128_64() {
            tests::matmul_tests::test_matmul_cmma_128_128_64::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn test_matmul_cmma_128_128_128() {
            tests::matmul_tests::test_matmul_cmma_128_128_128::<TestRuntime>(&Default::default())
        }
    };
}
