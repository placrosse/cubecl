use cubecl_core::ir::{self as cube, ConstantScalarValue, FloatKind, IntKind};
use std::fmt::Display;

#[derive(Debug, Clone, PartialEq)]
pub enum Variable {
    GlobalInputArray(u16, Item),
    GlobalOutputArray(u16, Item),
    GlobalScalar(u16, Elem, cube::Elem),
    ConstantScalar(ConstantScalarValue, Elem),
    Local {
        id: u16,
        item: Item,
        depth: u8,
    },
    LocalBinding {
        id: u16,
        item: Item,
    },
    Named {
        name: String,
        item: Item,
        is_array: bool,
    },
    Slice {
        id: u16,
        item: Item,
        depth: u8,
    },
    LocalScalar {
        id: u16,
        elem: Elem,
        depth: u8,
    },
    SharedMemory(u16, Item, u32),
    ConstantArray(u16, Item, u32),
    LocalArray(u16, Item, u8, u32),
    Id,
    LocalInvocationIndex,
    LocalInvocationIdX,
    LocalInvocationIdY,
    LocalInvocationIdZ,
    WorkgroupId,
    WorkgroupIdX,
    WorkgroupIdY,
    WorkgroupIdZ,
    GlobalInvocationIdX,
    GlobalInvocationIdY,
    GlobalInvocationIdZ,
    WorkgroupSize,
    WorkgroupSizeX,
    WorkgroupSizeY,
    WorkgroupSizeZ,
    NumWorkgroups,
    NumWorkgroupsX,
    NumWorkgroupsY,
    NumWorkgroupsZ,
    SubgroupSize,
    SubgroupInvocationId,
    Ptr {
        id: u16,
        depth: u8,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Elem {
    F32,
    I32,
    AtomicI32,
    U32,
    AtomicU32,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Item {
    Vec4(Elem),
    Vec3(Elem),
    Vec2(Elem),
    Scalar(Elem),
}

#[derive(Debug, Clone)]
pub struct IndexedVariable {
    var: Variable,
    index: usize,
}

impl Variable {
    pub fn is_always_scalar(&self) -> bool {
        match self {
            Variable::GlobalScalar(_, _, _) => true,
            Variable::ConstantScalar(_, _) => true,
            Variable::LocalScalar { .. } => true,
            Variable::Id => true,
            Variable::LocalInvocationIndex => true,
            Variable::LocalInvocationIdX => true,
            Variable::LocalInvocationIdY => true,
            Variable::LocalInvocationIdZ => true,
            Variable::GlobalInputArray(_, _) => false,
            Variable::GlobalOutputArray(_, _) => false,
            Variable::SharedMemory(_, _, _) => false,
            Variable::ConstantArray(_, _, _) => false,
            Variable::LocalArray(_, _, _, _) => false,
            Variable::Local { .. } => false,
            Variable::LocalBinding { .. } => false,
            Variable::Named { .. } => false,
            Variable::Slice { .. } => false,
            Variable::WorkgroupIdX => true,
            Variable::WorkgroupIdY => true,
            Variable::WorkgroupIdZ => true,
            Variable::GlobalInvocationIdX => true,
            Variable::GlobalInvocationIdY => true,
            Variable::GlobalInvocationIdZ => true,
            Variable::WorkgroupSizeX => true,
            Variable::WorkgroupSizeY => true,
            Variable::WorkgroupSizeZ => true,
            Variable::NumWorkgroupsX => true,
            Variable::NumWorkgroupsY => true,
            Variable::NumWorkgroupsZ => true,
            Variable::WorkgroupId => true,
            Variable::WorkgroupSize => true,
            Variable::NumWorkgroups => true,
            Variable::SubgroupSize => true,
            Variable::SubgroupInvocationId => true,
            Variable::Ptr { .. } => false,
        }
    }
    pub fn index(&self, index: usize) -> IndexedVariable {
        IndexedVariable {
            var: self.clone(),
            index,
        }
    }
    pub fn is_atomic(&self) -> bool {
        match self {
            Variable::GlobalInputArray(_, item) => item.elem().is_atomic(),
            Variable::GlobalOutputArray(_, item) => item.elem().is_atomic(),
            Variable::GlobalScalar(_, elem, _) => elem.is_atomic(),
            Variable::Local { item, .. } => item.elem().is_atomic(),
            Variable::Named { item, .. } => item.elem().is_atomic(),
            Variable::Slice { item, .. } => item.elem().is_atomic(),
            Variable::LocalScalar { elem, .. } => elem.is_atomic(),
            Variable::SharedMemory(_, item, _) => item.elem().is_atomic(),
            Variable::LocalArray(_, item, _, _) => item.elem().is_atomic(),
            _ => false,
        }
    }

    pub fn item(&self) -> Item {
        match self {
            Self::GlobalInputArray(_, e) => *e,
            Self::GlobalOutputArray(_, e) => *e,
            Self::SharedMemory(_, e, _) => *e,
            Self::ConstantArray(_, e, _) => *e,
            Self::LocalArray(_, e, _, _) => *e,
            Self::Local { item, .. } => *item,
            Self::LocalBinding { item, .. } => *item,
            Self::Slice { item, .. } => *item,
            Self::Named { item, .. } => *item,
            Self::ConstantScalar(_, e) => Item::Scalar(*e),
            Self::GlobalScalar(_, e, _) => Item::Scalar(*e),
            Self::Id => Item::Scalar(Elem::U32),
            Self::LocalInvocationIndex => Item::Scalar(Elem::U32),
            Self::LocalInvocationIdX => Item::Scalar(Elem::U32),
            Self::LocalInvocationIdY => Item::Scalar(Elem::U32),
            Self::LocalInvocationIdZ => Item::Scalar(Elem::U32),
            Self::LocalScalar { elem, .. } => Item::Scalar(*elem),
            Self::WorkgroupId => Item::Scalar(Elem::U32),
            Self::WorkgroupIdX => Item::Scalar(Elem::U32),
            Self::WorkgroupIdY => Item::Scalar(Elem::U32),
            Self::WorkgroupIdZ => Item::Scalar(Elem::U32),
            Self::GlobalInvocationIdX => Item::Scalar(Elem::U32),
            Self::GlobalInvocationIdY => Item::Scalar(Elem::U32),
            Self::GlobalInvocationIdZ => Item::Scalar(Elem::U32),
            Self::WorkgroupSize => Item::Scalar(Elem::U32),
            Self::WorkgroupSizeX => Item::Scalar(Elem::U32),
            Self::WorkgroupSizeY => Item::Scalar(Elem::U32),
            Self::WorkgroupSizeZ => Item::Scalar(Elem::U32),
            Self::NumWorkgroups => Item::Scalar(Elem::U32),
            Self::NumWorkgroupsX => Item::Scalar(Elem::U32),
            Self::NumWorkgroupsY => Item::Scalar(Elem::U32),
            Self::NumWorkgroupsZ => Item::Scalar(Elem::U32),
            Self::SubgroupSize => Item::Scalar(Elem::U32),
            Self::SubgroupInvocationId => Item::Scalar(Elem::U32),
            Variable::GlobalInputArray(_, item) => todo!(),
            Variable::GlobalOutputArray(_, item) => todo!(),
            Variable::GlobalScalar(_, elem, elem1) => todo!(),
            Variable::ConstantScalar(constant_scalar_value, elem) => todo!(),
            Variable::Local { id, item, depth } => todo!(),
            Variable::LocalBinding { id, item } => todo!(),
            Variable::Named {
                name,
                item,
                is_array,
            } => todo!(),
            Variable::Slice { id, item, depth } => todo!(),
            Variable::LocalScalar { id, elem, depth } => todo!(),
            Variable::SharedMemory(_, item, _) => todo!(),
            Variable::ConstantArray(_, item, _) => todo!(),
            Variable::LocalArray(_, item, _, _) => todo!(),
            Variable::Id => todo!(),
            Variable::LocalInvocationIndex => todo!(),
            Variable::LocalInvocationIdX => todo!(),
            Variable::LocalInvocationIdY => todo!(),
            Variable::LocalInvocationIdZ => todo!(),
            Variable::WorkgroupId => todo!(),
            Variable::WorkgroupIdX => todo!(),
            Variable::WorkgroupIdY => todo!(),
            Variable::WorkgroupIdZ => todo!(),
            Variable::GlobalInvocationIdX => todo!(),
            Variable::GlobalInvocationIdY => todo!(),
            Variable::GlobalInvocationIdZ => todo!(),
            Variable::WorkgroupSize => todo!(),
            Variable::WorkgroupSizeX => todo!(),
            Variable::WorkgroupSizeY => todo!(),
            Variable::WorkgroupSizeZ => todo!(),
            Variable::NumWorkgroups => todo!(),
            Variable::NumWorkgroupsX => todo!(),
            Variable::NumWorkgroupsY => todo!(),
            Variable::NumWorkgroupsZ => todo!(),
            Variable::SubgroupSize => todo!(),
            Variable::SubgroupInvocationId => todo!(),
            Variable::Ptr { id, depth } => todo!(),
        }
    }
    pub fn elem(&self) -> Elem {
        *self.item().elem()
    }

    pub fn fmt_cast_to(&self, item: Item) -> String {
        if self.item() != item {
            format!("{item}({self})")
        } else {
            format!("{self}")
        }
    }
}

impl Item {
    pub fn elem(&self) -> &Elem {
        match self {
            Item::Vec4(e) => e,
            Item::Vec3(e) => e,
            Item::Vec2(e) => e,
            Item::Scalar(e) => e,
        }
    }

    pub fn vectorization_factor(&self) -> usize {
        match self {
            Item::Vec4(_) => 4,
            Item::Vec3(_) => 3,
            Item::Vec2(_) => 2,
            Item::Scalar(_) => 1,
        }
    }

    pub fn fmt_cast_to(&self, item: Item, text: String) -> String {
        if *self != item {
            format!("{item}({text})")
        } else {
            text
        }
    }
}

impl Elem {
    pub fn size(&self) -> usize {
        match self {
            Self::F32 => core::mem::size_of::<f32>(),
            Self::I32 => core::mem::size_of::<i32>(),
            Self::AtomicI32 => core::mem::size_of::<i32>(),
            Self::U32 => core::mem::size_of::<u32>(),
            Self::AtomicU32 => core::mem::size_of::<u32>(),
            Self::Bool => core::mem::size_of::<bool>(),
        }
    }

    pub fn is_atomic(&self) -> bool {
        matches!(self, Self::AtomicI32 | Self::AtomicU32)
    }
}

impl Display for Elem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => f.write_str("f32"),
            Self::I32 => f.write_str("i32"),
            Self::AtomicI32 => f.write_str("atomic<i32>"),
            Self::U32 => f.write_str("u32"),
            Self::AtomicU32 => f.write_str("atomic<u32>"),
            Self::Bool => f.write_str("bool"),
        }
    }
}

impl Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Item::Vec4(elem) => write!(f, "vec4<{elem}>"),
            Item::Vec3(elem) => write!(f, "vec3<{elem}>"),
            Item::Vec2(elem) => write!(f, "vec2<{elem}>"),
            Item::Scalar(elem) => write!(f, "{elem}"),
        }
    }
}

fn format_number(num: f64) -> String {
    let formatted = format!("{:.34}", num);
    let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
    trimmed.to_string() + "f"
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::GlobalInputArray(number, _) => {
                write!(f, "input_{number}_global")
            }
            Variable::LocalScalar {
                id: index,
                depth: scope_depth,
                ..
            } => write!(f, "s_{scope_depth}_{index}"),
            Variable::Local {
                id: index,
                depth: scope_depth,
                ..
            } => write!(f, "l_{scope_depth}_{index}"),
            Variable::LocalBinding { id: index, .. } => write!(f, "_{index}"),
            Variable::Named { name, .. } => f.write_str(name),
            Variable::Slice {
                id: index,
                item: _,
                depth: scope_depth,
            } => write!(f, "slice_{scope_depth}_{index}"),
            Variable::GlobalOutputArray(number, _) => {
                write!(f, "output_{number}_global")
            }
            Variable::GlobalScalar(number, _, elem) => {
                write!(f, "scalars_{elem}[{number}]")
            }
            // We do the conversion in Rust and then render the number to avoid overflow or other
            // precision related problems.
            Variable::ConstantScalar(number, _elem) => match number {
                ConstantScalarValue::Int(val, kind) => match kind {
                    IntKind::I32 => write!(f, "{}i", *val as i32),
                    _ => unimplemented!("{:?} not supported in WGSL", kind),
                },
                ConstantScalarValue::Float(val, kind) => match kind {
                    FloatKind::F16 | FloatKind::BF16 | FloatKind::TF32 => {
                        todo!("Unsupported")
                    }
                    FloatKind::F32 | FloatKind::Flex32 | FloatKind::F64 => {
                        f.write_str(&format_number(*val))
                    }
                },
                ConstantScalarValue::UInt(val, _) => write!(f, "{}u", *val as u32),
                ConstantScalarValue::Bool(val) => write!(f, "{}", val),
            },
            Variable::SharedMemory(number, _, _) => {
                write!(f, "shared_memory_{number}")
            }
            Variable::ConstantArray(number, _, _) => write!(f, "arrays_{number}"),
            Variable::LocalArray(number, _, scope_depth, _) => {
                write!(f, "a_{scope_depth}_{number}")
            }
            Variable::Id => f.write_str("id"),
            Variable::LocalInvocationIndex => f.write_str("local_idx"),
            Variable::LocalInvocationIdX => f.write_str("local_invocation_id.x"),
            Variable::LocalInvocationIdY => f.write_str("local_invocation_id.y"),
            Variable::LocalInvocationIdZ => f.write_str("local_invocation_id.z"),
            Variable::WorkgroupId => f.write_str("workgroup_id_no_axis"),
            Variable::WorkgroupIdX => f.write_str("workgroup_id.x"),
            Variable::WorkgroupIdY => f.write_str("workgroup_id.y"),
            Variable::WorkgroupIdZ => f.write_str("workgroup_id.z"),
            Variable::GlobalInvocationIdX => f.write_str("global_id.x"),
            Variable::GlobalInvocationIdY => f.write_str("global_id.y"),
            Variable::GlobalInvocationIdZ => f.write_str("global_id.z"),
            Variable::WorkgroupSizeX => f.write_str("WORKGROUP_SIZE_X"),
            Variable::WorkgroupSizeY => f.write_str("WORKGROUP_SIZE_Y"),
            Variable::WorkgroupSizeZ => f.write_str("WORKGROUP_SIZE_Z"),
            Variable::NumWorkgroupsX => f.write_str("num_workgroups.x"),
            Variable::NumWorkgroupsY => f.write_str("num_workgroups.y"),
            Variable::NumWorkgroupsZ => f.write_str("num_workgroups.z"),
            Variable::WorkgroupSize => f.write_str("workgroup_size_no_axis"),
            Variable::NumWorkgroups => f.write_str("num_workgroups_no_axis"),
            Variable::SubgroupSize => f.write_str("subgroup_size"),
            Variable::SubgroupInvocationId => f.write_str("subgroup_invocation_id"),
            Variable::Ptr { id, depth } => todo!(),
        }
    }
}

impl Display for IndexedVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let var = &self.var;
        let item = var.item();
        let index = self.index;

        match &self.var {
            Variable::GlobalScalar(_, _, _) => write!(f, "{var}"),
            var if matches!(item, Item::Scalar(_)) => write!(f, "{var}"),
            var => write!(f, "{var}[{index}]"),
        }
    }
}

impl Variable {
    pub fn fmt_left(&self) -> String {
        match self {
            Variable::LocalBinding { id, .. } => {
                format!("let _{id}")
            }
            var => format!("{}", var),
        }
    }
}

impl IndexedVariable {
    pub fn fmt_left(&self) -> String {
        let item = self.var.item();
        match &self.var {
            Variable::GlobalScalar(_, _, _) => self.var.fmt_left(),
            var if matches!(item, Item::Scalar(_)) => var.fmt_left(),
            _ => format!("{self}"),
        }
    }

    pub fn fmt_cast(&self, item: Item) -> String {
        if self.var.item() != item {
            format!("{item}({self})")
        } else {
            format!("{self}")
        }
    }
}
