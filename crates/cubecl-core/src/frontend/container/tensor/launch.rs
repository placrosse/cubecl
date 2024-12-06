use std::{marker::PhantomData, num::NonZero};

use crate::{
    compute::{KernelBuilder, KernelLauncher},
    ir::{Item, LineSize},
    prelude::{ArgSettings, CubePrimitive, ExpandElementTyped, LaunchArg, LaunchArgExpand},
    Runtime,
};

use super::Tensor;

/// Argument to be used for [tensors](Tensor) passed as arguments to kernels.
#[derive(Debug)]
pub enum TensorArg<'a, R: Runtime> {
    /// The tensor is passed with a tensor handle.
    Handle {
        /// The tensor handle.
        handle: TensorHandleRef<'a, R>,
        /// The line size.
        line_size: u8,
    },
    /// The tensor is aliasing another input tensor.
    Alias {
        /// The position of the input tensor.
        input_pos: usize,
    },
}

/// Tensor representation with a reference to the [server handle](cubecl_runtime::server::Handle),
/// the strides and the shape.
pub struct TensorHandleRef<'a, R: Runtime> {
    pub handle: &'a cubecl_runtime::server::Handle,
    pub strides: &'a [usize],
    pub shape: &'a [usize],
    pub elem_size: usize,
    pub runtime: PhantomData<R>,
}

impl<R: Runtime> core::fmt::Debug for TensorHandleRef<'_, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "TensorHandleRef {{ strides: {:?}, shape: {:?} }}",
            self.strides, self.shape
        )
    }
}

/// Compilation argument for a [tensor](Tensor).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TensorCompilationArg {
    pub inplace: Option<u16>,
    pub line_size: LineSize,
}

impl<C: CubePrimitive> LaunchArgExpand for Tensor<C> {
    type CompilationArg = TensorCompilationArg;

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<Tensor<C>> {
        builder
            .input_tensor(Item::lined(C::as_elem(), arg.line_size))
            .into()
    }
    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<Tensor<C>> {
        match arg.inplace {
            Some(id) => builder.inplace_output(id).into(),
            None => builder
                .output_tensor(Item::lined(C::as_elem(), arg.line_size))
                .into(),
        }
    }
}

impl<C: CubePrimitive> LaunchArg for Tensor<C> {
    type RuntimeArg<'a, R: Runtime> = TensorArg<'a, R>;

    fn compilation_arg<R: Runtime>(runtime_arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
        match runtime_arg {
            TensorArg::Handle {
                line_size,
                ..
            } => TensorCompilationArg {
                inplace: None,
                line_size: LineSize::Some(NonZero::new(*line_size).unwrap()),
            },
            TensorArg::Alias { input_pos } => TensorCompilationArg {
                inplace: Some(*input_pos as u16),
                line_size: LineSize::None,
            },
        }
    }
}

impl<'a, R: Runtime> TensorArg<'a, R> {
    /// Create a new tensor argument specified with its line size.
    ///
    /// # Safety
    ///
    /// If you provide wrong strides or shapes, it might create undefined behavior caused by
    /// out-of-bound reads and writes.
    pub unsafe fn from_raw_parts<E: CubePrimitive>(
        handle: &'a cubecl_runtime::server::Handle,
        strides: &'a [usize],
        shape: &'a [usize],
        line_size: u8,
    ) -> Self {
        unsafe {
            Self::Handle {
                handle: TensorHandleRef::from_raw_parts(
                    handle,
                    strides,
                    shape,
                    E::as_elem().size(),
                ),
                line_size,
            }
        }
    }

    /// Create a new tensor argument specified with its line size with a manual element
    /// size in bytes.
    ///
    /// # Safety
    ///
    /// If you provide wrong strides or shapes, it might create undefined behavior caused by
    /// out-of-bound reads and writes.
    pub unsafe fn from_raw_parts_and_size(
        handle: &'a cubecl_runtime::server::Handle,
        strides: &'a [usize],
        shape: &'a [usize],
        line_size: u8,
        elem_size: usize,
    ) -> Self {
        unsafe {
            Self::Handle {
                handle: TensorHandleRef::from_raw_parts(handle, strides, shape, elem_size),
                line_size,
            }
        }
    }

    /// Create an alias argument.
    pub fn alias(position: usize) -> Self {
        Self::Alias {
            input_pos: position,
        }
    }
}

impl<R: Runtime> ArgSettings<R> for TensorArg<'_, R> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        launcher.register_tensor(self)
    }
}

impl<'a, R: Runtime> TensorHandleRef<'a, R> {
    /// Convert the handle into a [tensor argument](TensorArg).
    pub fn as_tensor_arg(&'a self, line_size: u8) -> TensorArg<'a, R> {
        unsafe {
            TensorArg::from_raw_parts_and_size(
                self.handle,
                self.strides,
                self.shape,
                line_size,
                self.elem_size,
            )
        }
    }
    /// Create a handle from raw parts.
    ///
    /// # Safety
    ///
    /// If you provide wrong strides or shapes, it might create undefined behavior caused by
    /// out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts(
        handle: &'a cubecl_runtime::server::Handle,
        strides: &'a [usize],
        shape: &'a [usize],
        elem_size: usize,
    ) -> Self {
        Self {
            handle,
            strides,
            shape,
            elem_size,
            runtime: PhantomData,
        }
    }
}
