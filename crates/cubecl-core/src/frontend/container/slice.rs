use std::marker::PhantomData;

use crate::{
    frontend::{indexation::Index, Tensor},
    ir::{self, Operator},
    prelude::{AsPtr, CubeContext, IntoRuntime},
    unexpanded,
};
use crate::{
    frontend::{
        Array, CubePrimitive, CubeType, ExpandElement, ExpandElementTyped, Init, SharedMemory,
        SizedContainer,
    },
    ir::Instruction,
};

use super::Line;

/// A read-only contiguous list of elements
///
/// # Safety
///
/// Since data can't be deallocated during kernel execution, this is safe.
#[derive(Clone)]
pub struct Slice<E> {
    _e: PhantomData<E>,
}

impl<E: CubePrimitive> AsPtr for Slice<E> {}
impl<E: CubePrimitive> AsPtr for SliceMut<E> {}

/// A read-write contiguous list of elements.
///
/// # Safety
///
/// Since data can be accessed by any unit during kernel execution, this can never be safe.
pub struct SliceMut<E> {
    _e: PhantomData<E>,
}

mod metadata {
    use super::*;

    impl<E> Slice<E> {
        /// Get the length of the slice.
        #[allow(clippy::len_without_is_empty)]
        pub fn len(&self) -> u32 {
            unexpanded!()
        }

        /// Returns the same slice, but with lines of length 1.
        pub fn to_aligned(&self) -> Slice<Line<E>>
        where
            E: CubePrimitive,
        {
            unexpanded!()
        }
    }

    impl<E> SliceMut<E> {
        /// Get the length of the slice.
        #[allow(clippy::len_without_is_empty)]
        pub fn len(&self) -> u32 {
            unexpanded!()
        }

        /// Returns the same slice, but with lines of length 1.
        pub fn into_aligned(self) -> SliceMut<Line<E>>
        where
            E: CubePrimitive,
        {
            unexpanded!()
        }
    }

    impl<C: CubeType> ExpandElementTyped<Slice<C>> {
        // Expand method of [len](Slice::len).
        pub fn __expand_len_method(self, context: &mut CubeContext) -> ExpandElementTyped<u32> {
            let elem: ExpandElementTyped<Array<u32>> = self.expand.into();
            elem.__expand_len_method(context)
        }

        // Expand method of [len](Slice::to_aligned).
        pub fn __expand_to_aligned_method(
            self,
            _context: &mut CubeContext,
        ) -> ExpandElementTyped<Slice<Line<C>>>
        where
            C: CubePrimitive,
        {
            self.expand.into()
        }

        // Expand method of [clone](Clone::clone).
        pub fn __expand_clone_method(
            self,
            _context: &mut CubeContext,
        ) -> ExpandElementTyped<Slice<Line<C>>>
        where
            C: CubePrimitive,
        {
            self.expand.into()
        }
    }

    impl<C: CubeType> ExpandElementTyped<SliceMut<C>> {
        // Expand method of [len](SliceMut::len).
        pub fn __expand_len_method(self, context: &mut CubeContext) -> ExpandElementTyped<u32> {
            let elem: ExpandElementTyped<Array<u32>> = self.expand.into();
            elem.__expand_len_method(context)
        }

        // Expand method of [len](SliceMut::into_aligned).
        pub fn __expand_into_aligned_method(
            self,
            _context: &mut CubeContext,
        ) -> ExpandElementTyped<SliceMut<Line<C>>>
        where
            C: CubePrimitive,
        {
            self.expand.into()
        }
    }
}

/// Module that contains the implementation details of the index functions.
mod indexation {
    use ir::Instruction;

    use crate::{
        ir::{BinaryOperator, Operator},
        prelude::{CubeIndex, CubeIndexMut},
    };

    use super::*;

    impl<E: CubePrimitive> Slice<E> {
        /// Perform an unchecked index into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        pub unsafe fn index_unchecked<I: Index>(&self, _i: I) -> &E
        where
            Self: CubeIndex<I>,
        {
            unexpanded!()
        }
    }

    impl<E: CubePrimitive> SliceMut<E> {
        /// Perform an unchecked index into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        pub unsafe fn index_unchecked<I: Index>(&self, _i: I) -> &E
        where
            Self: CubeIndex<I>,
        {
            unexpanded!()
        }

        /// Perform an unchecked index assignment into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        pub unsafe fn index_assign_unchecked<I: Index>(&mut self, _i: I, _value: E)
        where
            Self: CubeIndexMut<I>,
        {
            unexpanded!()
        }
    }

    impl<E: CubePrimitive> ExpandElementTyped<Slice<E>> {
        pub fn __expand_index_unchecked_method(
            self,
            context: &mut CubeContext,
            i: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<E> {
            let out = context.create_local_binding(self.expand.item);
            context.register(Instruction::new(
                Operator::UncheckedIndex(BinaryOperator {
                    lhs: *self.expand,
                    rhs: i.expand.consume(),
                }),
                *out,
            ));
            out.into()
        }
    }

    impl<E: CubePrimitive> ExpandElementTyped<SliceMut<E>> {
        pub fn __expand_index_unchecked_method(
            self,
            context: &mut CubeContext,
            i: ExpandElementTyped<u32>,
        ) -> ExpandElementTyped<E> {
            let out = context.create_local_binding(self.expand.item);
            context.register(Instruction::new(
                Operator::UncheckedIndex(BinaryOperator {
                    lhs: *self.expand,
                    rhs: i.expand.consume(),
                }),
                *out,
            ));
            out.into()
        }

        pub fn __expand_index_assign_unchecked_method(
            self,
            context: &mut CubeContext,
            i: ExpandElementTyped<u32>,
            value: ExpandElementTyped<E>,
        ) {
            context.register(Instruction::new(
                Operator::UncheckedIndexAssign(BinaryOperator {
                    lhs: i.expand.consume(),
                    rhs: value.expand.consume(),
                }),
                *self.expand,
            ));
        }
    }
}

impl<E: CubeType> CubeType for Slice<E> {
    type ExpandType = ExpandElementTyped<Slice<E>>;
}

impl<C: CubeType> Init for ExpandElementTyped<Slice<C>> {
    fn init(self, _context: &mut crate::prelude::CubeContext) -> Self {
        // The type can't be deeply cloned/copied.
        self
    }
}

impl<E: CubeType> CubeType for SliceMut<E> {
    type ExpandType = ExpandElementTyped<SliceMut<E>>;
}

impl<E: CubeType> CubeType for &mut SliceMut<E> {
    type ExpandType = ExpandElementTyped<SliceMut<E>>;
}

impl<C: CubeType> Init for ExpandElementTyped<SliceMut<C>> {
    fn init(self, _context: &mut crate::prelude::CubeContext) -> Self {
        // The type can't be deeply cloned/copied.
        self
    }
}

impl<C: CubeType<ExpandType = ExpandElementTyped<C>>> SizedContainer for Slice<C> {
    type Item = C;
}

impl<T: CubeType> Iterator for Slice<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        unexpanded!()
    }
}

pub trait SliceOperator<E: CubeType>: CubeType<ExpandType = Self::Expand> {
    type Expand: SliceOperatorExpand<E>;

    /// Return a read-only view of all elements comprise between the start and end index.
    #[allow(unused_variables)]
    fn slice<Start: Index, End: Index>(&self, start: Start, end: End) -> Slice<E> {
        unexpanded!()
    }
    /// Expand function of [SliceOperator::slice].
    fn __expand_slice(
        context: &mut CubeContext,
        expand: Self::Expand,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Slice<E>> {
        expand.__expand_slice_method(context, start, end)
    }

    /// Return a read-write view of all elements comprise between the start and end index.
    #[allow(unused_variables)]
    fn slice_mut<Start: Index, End: Index>(&mut self, start: Start, end: End) -> SliceMut<E> {
        unexpanded!()
    }

    /// Expand function of [SliceOperator::slice_mut].
    fn __expand_slice_mut(
        context: &mut CubeContext,
        expand: Self::Expand,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<SliceMut<E>> {
        expand.__expand_slice_mut_method(context, start, end)
    }

    /// Reinterprete the current type as a read-only slice.
    #[allow(unused_variables)]
    fn to_slice(&self) -> Slice<E> {
        unexpanded!()
    }

    /// Expand function of [SliceOperator::to_slice].
    fn __expand_to_slice(
        context: &mut CubeContext,
        expand: Self::Expand,
    ) -> ExpandElementTyped<Slice<E>> {
        expand.__expand_to_slice_method(context)
    }

    /// Reinterprete the current type as a read-write slice.
    #[allow(unused_variables)]
    fn to_slice_mut(&mut self) -> SliceMut<E> {
        unexpanded!()
    }

    /// Expand function of [SliceOperator::to_slice_mut].
    fn __expand_to_slice_mut(
        context: &mut CubeContext,
        expand: Self::Expand,
    ) -> ExpandElementTyped<SliceMut<E>> {
        expand.__expand_to_slice_mut_method(context)
    }
}

pub trait SliceOperatorExpand<E: CubeType>: Into<ExpandElement> + Clone {
    fn slice_base<Start: Index, End: Index>(
        &self,
        context: &mut CubeContext,
        start: Start,
        end: End,
    ) -> ExpandElement;

    fn __expand_slice_method(
        &self,
        context: &mut CubeContext,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Slice<E>> {
        ExpandElementTyped::new(self.slice_base(context, start, end))
    }

    fn __expand_slice_mut_method(
        &self,
        context: &mut CubeContext,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<SliceMut<E>> {
        ExpandElementTyped::new(self.slice_base(context, start, end))
    }

    fn __expand_to_slice_method(&self, _context: &mut CubeContext) -> ExpandElementTyped<Slice<E>> {
        let expand = self.clone().into();
        ExpandElementTyped::new(expand)
    }

    fn __expand_to_slice_mut_method(
        &self,
        _context: &mut CubeContext,
    ) -> ExpandElementTyped<SliceMut<E>> {
        let expand = self.clone().into();
        ExpandElementTyped::new(expand)
    }
}

macro_rules! slice_op {
    ($type:ident) => {
        impl<E: CubePrimitive> SliceOperator<E> for $type<E> {
            type Expand = ExpandElementTyped<$type<E>>;
        }

        impl<E: CubePrimitive> SliceOperatorExpand<E> for ExpandElementTyped<$type<E>> {
            fn slice_base<Start: Index, End: Index>(
                &self,
                context: &mut CubeContext,
                start: Start,
                end: End,
            ) -> ExpandElement {
                slice_expand(context, self.clone(), start, end)
            }
        }
    };
    (slice $type:ident) => {
        impl<E: CubePrimitive> SliceOperator<E> for $type<E> {
            type Expand = ExpandElementTyped<$type<E>>;
        }

        impl<E: CubePrimitive> SliceOperatorExpand<E> for ExpandElementTyped<$type<E>> {
            fn slice_base<Start: Index, End: Index>(
                &self,
                context: &mut CubeContext,
                start: Start,
                end: End,
            ) -> ExpandElement {
                slice_expand(context, self.clone(), start, end)
            }
        }
    };
}

slice_op!(Array);
slice_op!(Tensor);
slice_op!(SharedMemory);
slice_op!(slice Slice);
slice_op!(slice SliceMut);

pub fn slice_expand<I: Into<ExpandElement>, S1: Index, S2: Index>(
    context: &mut CubeContext,
    input: I,
    start: S1,
    end: S2, // Todo use it to get the length.
) -> ExpandElement {
    let input = input.into();
    let out = context.create_slice(input.item);

    context.register(Instruction::new(
        Operator::Slice(ir::SliceOperator {
            input: *input,
            start: start.value(),
            end: end.value(),
        }),
        *out,
    ));

    out
}

impl<E: CubePrimitive> IntoRuntime for Slice<E> {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> Self::ExpandType {
        unimplemented!("Array can't exist at compile time")
    }
}

impl<E: CubePrimitive> IntoRuntime for SliceMut<E> {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> Self::ExpandType {
        unimplemented!("Array can't exist at compile time")
    }
}
