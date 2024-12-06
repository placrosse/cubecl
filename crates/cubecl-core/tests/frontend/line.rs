use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub fn line_binary<T: Numeric>(lhs: T) {
    let _ = lhs + T::from_vec([4, 5]);
}

#[cube]
pub fn line_cmp<T: Numeric>(rhs: T) {
    let _ = T::from_vec([4, 5]) > rhs;
}

mod tests {
    use std::num::NonZero;

    use super::*;
    use cubecl_core::ir::Item;

    type ElemType = f32;

    #[test]
    fn cube_line_binary_op_with_same_scheme_does_not_fail() {
        let mut context = CubeContext::default();

        let lhs = context.create_local_binding(Item::lined(ElemType::as_elem(), NonZero::new(2)));

        line_binary::expand::<ElemType>(&mut context, lhs.into());
    }

    #[test]
    #[should_panic]
    fn cube_line_binary_op_with_different_scheme_fails() {
        let mut context = CubeContext::default();

        let lhs = context.create_local_binding(Item::lined(ElemType::as_elem(), NonZero::new(4)));

        line_binary::expand::<ElemType>(&mut context, lhs.into());
    }

    #[test]
    fn cube_line_cmp_op_with_same_scheme_does_not_fail() {
        let mut context = CubeContext::default();

        let lhs = context.create_local_binding(Item::lined(ElemType::as_elem(), NonZero::new(2)));

        line_cmp::expand::<ElemType>(&mut context, lhs.into());
    }

    #[test]
    #[should_panic]
    fn cube_line_cmp_op_with_different_scheme_fails() {
        let mut context = CubeContext::default();

        let lhs = context.create_local_binding(Item::lined(ElemType::as_elem(), NonZero::new(4)));

        line_cmp::expand::<ElemType>(&mut context, lhs.into());
    }

    #[test]
    fn cube_line_can_be_broadcasted() {
        let mut context = CubeContext::default();

        let lhs = context.create_local_binding(Item::lined(ElemType::as_elem(), None));

        line_cmp::expand::<ElemType>(&mut context, lhs.into());
    }
}
