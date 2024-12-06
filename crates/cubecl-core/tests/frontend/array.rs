use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube]
pub fn array_read_write<T: Numeric>(#[comptime] array_size: u32) {
    let mut array = Array::<T>::new(array_size);
    array[0] = T::from_int(3);
    let _a = array[0];
}

#[cube]
pub fn array_to_lined_variable<T: Numeric>() -> T {
    let mut array = Array::<T>::new(2);
    array[0] = T::from_int(0);
    array[1] = T::from_int(1);
    array.to_line(2)
}

#[cube]
pub fn array_of_one_to_lined_variable<T: Numeric>() -> T {
    let mut array = Array::<T>::new(1);
    array[0] = T::from_int(3);
    array.to_line(1)
}

#[cube]
pub fn array_add_assign_simple(array: &mut Array<u32>) {
    array[1] += 1;
}

#[cube]
pub fn array_add_assign_expr(array: &mut Array<u32>) {
    array[1 + 5] += 1;
}

mod tests {
    use pretty_assertions::assert_eq;
    use std::num::NonZero;

    use super::*;
    use cubecl_core::{
        cpa,
        ir::{self, Item, Variable, VariableKind},
    };

    type ElemType = f32;

    #[test]
    fn cube_support_array() {
        let mut context = CubeContext::default();

        array_read_write::expand::<ElemType>(&mut context, 512);
        assert_eq!(
            context.into_scope().operations,
            inline_macro_ref_read_write()
        )
    }

    #[test]
    fn array_add_assign() {
        let mut context = CubeContext::default();
        let array = context.input(0, Item::new(u32::as_elem()));

        array_add_assign_simple::expand(&mut context, array.into());
        let scope = context.into_scope();

        assert_eq!(scope.operations, inline_macro_array_add_assign_simple());
    }

    #[test]
    fn cube_array_to_lined() {
        let mut context = CubeContext::default();

        array_to_lined_variable::expand::<ElemType>(&mut context);
        assert_eq!(context.into_scope().operations, inline_macro_ref_to_lined());
    }

    #[test]
    fn cube_array_of_one_to_lined() {
        let mut context = CubeContext::default();

        array_of_one_to_lined_variable::expand::<ElemType>(&mut context);
        assert_eq!(
            context.into_scope().operations,
            inline_macro_ref_one_to_lined()
        );
    }

    fn inline_macro_ref_read_write() -> Vec<ir::Instruction> {
        let context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());

        let mut scope = context.into_scope();
        let var = scope.create_local(item);
        let pos: Variable = 0u32.into();

        // Create
        let array = scope.create_local_array(item, 512);

        // Write
        cpa!(scope, array[pos] = 3.0_f32);

        // Read
        cpa!(scope, var = array[pos]);

        scope.operations
    }

    #[test]
    fn array_add_assign_expr() {
        let mut context = CubeContext::default();
        let array = context.input(0, Item::new(u32::as_elem()));

        array_add_assign_expr::expand(&mut context, array.into());
        let scope = context.into_scope();

        assert_eq!(scope.operations, inline_macro_array_add_assign_expr());
    }

    fn inline_macro_array_add_assign_simple() -> Vec<ir::Instruction> {
        let context = CubeContext::default();

        let mut scope = context.into_scope();
        let local = scope.create_local(Item::new(u32::as_elem()));

        let array = Variable::new(VariableKind::GlobalInputArray(0), Item::new(u32::as_elem()));
        let index: Variable = 1u32.into();
        let value: Variable = 1u32.into();

        cpa!(scope, local = array[index]);
        cpa!(scope, local += value);
        cpa!(scope, array[index] = local);

        scope.operations
    }

    fn inline_macro_ref_to_lined() -> Vec<ir::Instruction> {
        let context = CubeContext::default();
        let scalar_item = Item::new(ElemType::as_elem());
        let lined_item = Item::lined(ElemType::as_elem(), NonZero::new(2));

        let mut scope = context.into_scope();
        let pos0: Variable = 0u32.into();
        let pos1: Variable = 1u32.into();
        let array = scope.create_local_array(scalar_item, 2);
        cpa!(scope, array[pos0] = 0.0_f32);
        cpa!(scope, array[pos1] = 1.0_f32);

        let lined_var = scope.create_local(lined_item);
        let tmp = scope.create_local(scalar_item);
        cpa!(scope, tmp = array[pos0]);
        cpa!(scope, lined_var[pos0] = tmp);
        cpa!(scope, tmp = array[pos1]);
        cpa!(scope, lined_var[pos1] = tmp);

        scope.operations
    }

    fn inline_macro_ref_one_to_lined() -> Vec<ir::Instruction> {
        let context = CubeContext::default();
        let scalar_item = Item::new(ElemType::as_elem());
        let unlined_item = Item::lined(ElemType::as_elem(), NonZero::new(1));

        let mut scope = context.into_scope();
        let pos0: Variable = 0u32.into();
        let array = scope.create_local_array(scalar_item, 1);
        cpa!(scope, array[pos0] = 3.0_f32);

        let unlined_var = scope.create_local(unlined_item);
        let tmp = scope.create_local(scalar_item);
        cpa!(scope, tmp = array[pos0]);
        cpa!(scope, unlined_var = tmp);

        scope.operations
    }

    fn inline_macro_array_add_assign_expr() -> Vec<ir::Instruction> {
        let context = CubeContext::default();

        let mut scope = context.into_scope();
        let local = scope.create_local(Item::new(u32::as_elem()));

        let array = Variable::new(VariableKind::GlobalInputArray(0), Item::new(u32::as_elem()));
        let index: Variable = 6u32.into();
        let value: Variable = 1u32.into();

        cpa!(scope, local = array[index]);
        cpa!(scope, local += value);
        cpa!(scope, array[index] = local);

        scope.operations
    }
}
