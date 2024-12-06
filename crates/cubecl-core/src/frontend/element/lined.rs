use crate::frontend::{Array, CubeType, ExpandElement, Tensor};
use crate::unexpanded;

pub trait Lined {
    fn line_size(&self) -> u32;
    fn to_line(self, line_size: u32) -> Self;
}

impl<T: CubeType> Lined for Tensor<T> {
    fn line_size(&self) -> u32 {
        unexpanded!()
    }

    fn to_line(self, _line_size: u32) -> Self {
        unexpanded!()
    }
}

impl<T: CubeType> Lined for &Tensor<T> {
    fn line_size(&self) -> u32 {
        unexpanded!()
    }

    fn to_line(self, _line_size: u32) -> Self {
        unexpanded!()
    }
}

impl<T: CubeType> Lined for Array<T> {
    fn line_size(&self) -> u32 {
        unexpanded!()
    }

    fn to_line(self, _line_size: u32) -> Self {
        unexpanded!()
    }
}

impl<T: CubeType> Lined for &Array<T> {
    fn line_size(&self) -> u32 {
        unexpanded!()
    }

    fn to_line(self, _line_size: u32) -> Self {
        unexpanded!()
    }
}

impl<T: CubeType> Lined for &mut Tensor<T> {
    fn line_size(&self) -> u32 {
        unexpanded!()
    }

    fn to_line(self, _line_size: u32) -> Self {
        unexpanded!()
    }
}

impl Lined for ExpandElement {
    fn line_size(&self) -> u32 {
        let var = match self {
            ExpandElement::Managed(var) => var,
            ExpandElement::Plain(var) => var,
        };

        var.item.line_size.map(|it| it.get()).unwrap_or(1) as u32
    }

    fn to_line(self, _line_size: u32) -> Self {
        todo!()
    }
}

impl Lined for &ExpandElement {
    fn line_size(&self) -> u32 {
        let var = match self {
            ExpandElement::Managed(var) => var,
            ExpandElement::Plain(var) => var,
        };

        var.item.line_size.map(|it| it.get()).unwrap_or(1) as u32
    }

    fn to_line(self, _line_size: u32) -> Self {
        todo!()
    }
}

/// Cubecl intrinsic. Gets the line size of an element at compile time.
pub fn line_size_of<C: CubeType>(_element: &C) -> u32 {
    1
}

pub mod line_size_of {
    use crate::prelude::*;

    pub fn expand<C: CubeType>(_context: &mut CubeContext, element: ExpandElementTyped<C>) -> u32 {
        let elem: ExpandElement = element.into();
        elem.item.line_size.map(|it| it.get() as u32).unwrap_or(1)
    }
}
