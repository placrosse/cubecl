use crate::{ir::Select, prelude::*};
use crate::{
    prelude::{CubePrimitive, Line},
    unexpanded,
};

/// Executes both branches, *then* selects a value based on the condition. This *should* be
/// branchless, but might depend on the compiler.
///
/// # Safety
///
/// Since both branches are *evaluated* regardless of the condition, both branches must be *valid*
/// regardless of the condition. Illegal memory accesses should not be done in either branch.
pub fn select<C: CubePrimitive>(condition: bool, then: C, or_else: C) -> C {
    if condition {
        then
    } else {
        or_else
    }
}

/// Same as [select] but with lines instead.
#[allow(unused_variables)]
pub fn select_many<C: CubePrimitive>(
    condition: Line<bool>,
    then: Line<C>,
    or_else: Line<C>,
) -> Line<C> {
    unexpanded!()
}

pub mod select {
    use std::num::NonZero;

    use crate::ir::{Instruction, Operator};

    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        condition: ExpandElementTyped<bool>,
        then: ExpandElementTyped<C>,
        or_else: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        let cond = condition.expand.consume();
        let then = then.expand.consume();
        let or_else = or_else.expand.consume();

        let vf = cond.line_size();
        let vf = Ord::max(vf, then.line_size());
        let vf = Ord::max(vf, or_else.line_size());

        let output = context.create_local_binding(then.item.to_line(NonZero::new(vf)));
        let out = *output;

        let select = Operator::Select(Select {
            cond,
            then,
            or_else,
        });
        context.register(Instruction::new(select, out));

        output.into()
    }
}

pub mod select_many {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        condition: ExpandElementTyped<Line<bool>>,
        then: ExpandElementTyped<Line<C>>,
        or_else: ExpandElementTyped<Line<C>>,
    ) -> ExpandElementTyped<Line<C>> {
        select::expand(context, condition.expand.into(), then, or_else)
    }
}
