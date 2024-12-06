use std::num::NonZero;

use super::Compiler;
use crate::{
    ir::{
        Binding, CubeDim, Elem, Item, KernelDefinition, LineSize, Location, ReadingStrategy, Scope,
        Variable, VariableKind, Visibility,
    },
    prelude::CubePrimitive,
    Runtime,
};

/// The kernel integrator allows you to create a [kernel definition](KernelDefinition) based on
/// [kernel expansion](KernelExpansion) and [kernel settings](KernelSettings).
#[derive(Clone)]
pub struct KernelIntegrator {
    expansion: KernelExpansion,
    input_bindings: Vec<Binding>,
    output_bindings: Vec<Binding>,
    named_bindings: Vec<(String, Binding)>,
}

/// The information necessary to compile a [kernel definition](KernelDefinition).
#[derive(Clone)]
pub struct KernelExpansion {
    pub inputs: Vec<InputInfo>,
    pub outputs: Vec<OutputInfo>,
    pub scope: Scope,
    pub kernel_name: String,
}

/// Simply indicate the line size that can be replaced by the input.
#[derive(new, Default, Clone, Debug, Hash, PartialEq, Eq)]
pub struct InplaceMapping {
    /// line size n.
    pub pos_input: usize,
    /// Output position.
    pub pos_output: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum LinePartial {
    Input { pos: usize, line_size: LineSize },
    Output { pos: usize, line_size: LineSize },
}

#[derive(Default, Clone, Debug, Hash, PartialEq, Eq)]
pub struct KernelSettings {
    pub mappings: Vec<InplaceMapping>,
    line_partial: Vec<LinePartial>,
    pub cube_dim: CubeDim,
    pub reading_strategy: Vec<(u16, ReadingStrategy)>,
    pub kernel_name: String,
}

impl core::fmt::Display for KernelSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The line size  implementation is to generate the shortest representation
        // that won't clash with any other compilation settings. This is crucial since we rely on
        // this representation to know when to compile a new version of a kernel.
        //
        // Each main section starts with a letter that can't be used by other main sections:
        //
        // * Mapping:          m
        //   * Input:  i
        //   * Output: o
        //
        // * Reading Strategy: r
        //   * Output layout: o
        //   * Plain:         p
        //
        // * Line Size Global:    vg{line_size}
        // * Line Size Partial Input:    v{line_size}i{pos}
        // * Line Size Partial Output:    vo
        // * Cube Dim X: x
        // * Cube Dim Y: y
        // * Cube Dim Z: z
        f.write_str("m")?;
        for mapping in self.mappings.iter() {
            f.write_fmt(format_args!(
                "i{}o{}",
                mapping.pos_input, mapping.pos_output
            ))?;
        }

        f.write_str("r")?;

        for (input, strategy) in self.reading_strategy.iter() {
            match strategy {
                ReadingStrategy::OutputLayout => f.write_fmt(format_args!("i{}o", input)),
                ReadingStrategy::Plain => f.write_fmt(format_args!("i{}p", input)),
            }?;
        }

        for line_size in self.line_partial.iter() {
            match line_size {
                LinePartial::Input { pos, line_size } => f.write_fmt(format_args!(
                    "v{}i{pos}",
                    line_size.map(NonZero::get).unwrap_or(1)
                ))?,
                LinePartial::Output { pos, line_size } => f.write_fmt(format_args!(
                    "v{}o{pos}",
                    line_size.map(NonZero::get).unwrap_or(1)
                ))?,
            };
        }

        f.write_fmt(format_args!(
            "x{}y{}z{}",
            self.cube_dim.x, self.cube_dim.y, self.cube_dim.x
        ))
    }
}

impl KernelSettings {
    /// Compile the shader with line size enabled for an input.
    #[allow(dead_code)]
    pub fn lined_input(mut self, position: usize, line_size: LineSize) -> Self {
        // Not setting the line size when it's the default value reduces the kernel id
        // size.
        if line_size.is_none() {
            return self;
        }

        self.line_partial.push(LinePartial::Input {
            pos: position,
            line_size,
        });
        self
    }

    /// Compile the shader with lines enabled for an output.
    #[allow(dead_code)]
    pub fn lined_output(mut self, position: usize, line_size: LineSize) -> Self {
        // Not setting the line size when it's the default value reduces the kernel id size.
        if line_size.is_none() {
            return self;
        }

        self.line_partial.push(LinePartial::Output {
            pos: position,
            line_size,
        });
        self
    }

    /// Fetch the line size for the provided input position.
    pub fn line_size_input(&self, position: usize) -> LineSize {
        for partial in self.line_partial.iter() {
            if let LinePartial::Input { pos, line_size } = partial {
                if *pos == position {
                    return *line_size;
                }
            }
        }

        None
    }

    /// Fetch the line size for the provided output position.
    pub fn line_size_output(&self, position: usize) -> LineSize {
        for partial in self.line_partial.iter() {
            if let LinePartial::Output { pos, line_size } = partial {
                if *pos == position {
                    return *line_size;
                }
            }
        }

        None
    }

    /// Compile the shader with inplace enabled by the given [mapping](InplaceMapping).
    ///
    /// Notes:
    ///
    /// You should favor using `dynamic_settings` when using fusion, since the mapping is going to
    /// be created from the runtime information.
    pub fn inplace(mut self, mappings: Vec<InplaceMapping>) -> Self {
        self.mappings = mappings;
        self
    }

    /// Set cube dimension.
    #[allow(dead_code)]
    pub fn cube_dim(mut self, cube_dim: CubeDim) -> Self {
        self.cube_dim = cube_dim;
        self
    }

    /// Set kernel name.
    #[allow(dead_code)]
    pub fn kernel_name<S: AsRef<str>>(mut self, name: S) -> Self {
        self.kernel_name = name.as_ref().to_string();
        self
    }
}

#[allow(dead_code)]
fn is_contiguous(strides: &[usize]) -> bool {
    let mut current = 0;

    for stride in strides.iter().rev() {
        if current > *stride {
            return false;
        }
        current = *stride;
    }

    true
}

/// Information related to an input.
#[derive(Clone, Debug)]
pub enum InputInfo {
    Array {
        item: Item,
        visibility: Visibility,
        /// Whether this input has extended metadata (rank, shape, strides)
        has_extended_meta: bool,
    },
    Scalar {
        elem: Elem,
        size: usize,
    },
}

impl InputInfo {
    /// The item type of the input.
    #[allow(dead_code)]
    pub fn item(&self) -> Item {
        match self {
            InputInfo::Array { item, .. } => *item,
            InputInfo::Scalar { elem, size: _ } => Item::new(*elem),
        }
    }
}

impl OutputInfo {
    /// The item type of the input.
    #[allow(dead_code)]
    pub fn item(&self) -> Item {
        match self {
            OutputInfo::ArrayWrite { item, .. } => *item,
            OutputInfo::InputArrayWrite { item, .. } => *item,
            OutputInfo::Array { item, .. } => *item,
        }
    }
}

/// Information related to an output.
#[derive(Clone, Debug)]
pub enum OutputInfo {
    /// Write the local variable to a new array.
    ///
    /// This will create a new binding in the [kernel definition](KernelDefinition).
    ArrayWrite {
        item: Item,
        local: u16,
        position: Variable,
        /// Whether this output has extended metadata (rank, shape, strides)
        has_extended_meta: bool,
    },
    /// Write the local variable to an existing input binding.
    InputArrayWrite {
        item: Item,
        input: u16,
        local: u16,
        position: Variable,
    },
    /// Simply register the output, but don't automatically add a write to it.
    ///
    /// Useful when a procedure writes to the output using operations.
    Array {
        item: Item,
        /// Whether this output has extended metadata (rank, shape, strides)
        has_extended_meta: bool,
    },
}

impl OutputInfo {
    #[allow(dead_code)]
    pub fn elem_size<R: Runtime>(&self) -> usize {
        let elem = match self {
            OutputInfo::ArrayWrite { item, .. } => bool_elem(item.elem()),
            OutputInfo::InputArrayWrite { item, .. } => bool_elem(item.elem()),
            OutputInfo::Array { item, .. } => bool_elem(item.elem()),
        };
        <R::Compiler as Compiler>::elem_size(elem)
    }
}

impl KernelIntegrator {
    /// Starts a new compilation.
    pub fn new(info: KernelExpansion) -> Self {
        Self {
            expansion: info,
            input_bindings: Default::default(),
            output_bindings: Default::default(),
            named_bindings: Default::default(),
        }
    }

    /// Performs the compilation with the provided [settings](KernelSettings).
    pub fn integrate(mut self, mut settings: KernelSettings) -> KernelDefinition {
        self.register_inputs(&settings);
        self.register_outputs(&mut settings);

        let inputs = self.input_bindings;
        let outputs = self.output_bindings;
        let mut named = Vec::with_capacity(2);

        named.push((
            "info".to_string(),
            Binding {
                item: Item::new(u32::as_elem()),
                visibility: Visibility::Read,
                location: Location::Storage,
                has_extended_meta: false,
                size: None, // We avoid putting the length here since it will force a new kernel
                            // for each tensor rank.
            },
        ));

        for (name, binding) in self.named_bindings.into_iter() {
            named.push((name, binding));
        }

        KernelDefinition {
            inputs,
            outputs,
            named,
            cube_dim: settings.cube_dim,
            body: self.expansion.scope,
            kernel_name: self.expansion.kernel_name,
        }
    }

    fn register_inputs(&mut self, settings: &KernelSettings) {
        for (id, strategy) in settings.reading_strategy.iter() {
            self.expansion.scope.update_read(*id, *strategy);
        }

        for input in self.expansion.inputs.drain(..) {
            match input {
                InputInfo::Array {
                    item,
                    visibility,
                    has_extended_meta,
                } => {
                    self.input_bindings.push(Binding {
                        item: bool_item(item),
                        visibility,
                        location: Location::Storage,
                        has_extended_meta,
                        size: None,
                    });
                }
                InputInfo::Scalar { elem, size } => {
                    let elem = bool_elem(elem);

                    self.named_bindings.push((
                        format!("scalars_{}", elem),
                        Binding {
                            item: Item::new(elem),
                            visibility: Visibility::Read,
                            location: Location::Storage,
                            has_extended_meta: false,
                            size: Some(size),
                        },
                    ));
                }
            }
        }
    }

    fn register_outputs(&mut self, settings: &mut KernelSettings) {
        let mut index = 0;

        if !settings.mappings.is_empty() {
            let mut mappings = Vec::new();
            core::mem::swap(&mut settings.mappings, &mut mappings);

            for mapping in mappings {
                self.register_inplace_mapping(mapping);
            }
        }

        for array in self.expansion.outputs.drain(..) {
            match array {
                OutputInfo::ArrayWrite {
                    item,
                    local,
                    position,
                    has_extended_meta,
                } => {
                    let item_adapted = bool_item(item);

                    self.output_bindings.push(Binding {
                        item: item_adapted,
                        visibility: Visibility::ReadWrite,
                        location: Location::Storage,
                        has_extended_meta,
                        size: None,
                    });
                    self.expansion.scope.write_global(
                        Variable::new(
                            VariableKind::Local {
                                id: local,

                                depth: self.expansion.scope.depth,
                            },
                            item,
                        ),
                        Variable::new(VariableKind::GlobalOutputArray(index), item_adapted),
                        position,
                    );
                    index += 1;
                }
                OutputInfo::InputArrayWrite {
                    item,
                    input,
                    local,
                    position,
                } => {
                    self.expansion.scope.write_global(
                        Variable::new(
                            VariableKind::Local {
                                id: local,
                                depth: self.expansion.scope.depth,
                            },
                            item,
                        ),
                        Variable::new(VariableKind::GlobalInputArray(input), bool_item(item)),
                        position,
                    );
                }
                OutputInfo::Array {
                    item,
                    has_extended_meta,
                } => {
                    let elem_adapted = bool_item(item);

                    self.output_bindings.push(Binding {
                        item: elem_adapted,
                        visibility: Visibility::ReadWrite,
                        location: Location::Storage,
                        has_extended_meta,
                        size: None,
                    });

                    index += 1;
                }
            }
        }
    }

    fn register_inplace_mapping(&mut self, mapping: InplaceMapping) {
        let output = match self.expansion.outputs.get_mut(mapping.pos_output) {
            Some(output) => output,
            None => {
                if let Some(binding) = self.input_bindings.get_mut(mapping.pos_input) {
                    // Update input visibility.
                    binding.visibility = Visibility::ReadWrite;
                }

                // The mapping is handled differently, normally by cube itself.
                return;
            }
        };

        let (item, local, position) = match output {
            OutputInfo::ArrayWrite { item, local, position, .. } => (item, local, position),
            OutputInfo::InputArrayWrite {
                item: _,
                input,
                local: _,
                position: _,
            } => {
                assert_eq!(
                    *input, mapping.pos_input as u16,
                    "Can't use different inputs for the same output."
                );
                return;
            }
            OutputInfo::Array { .. } => panic!("Can't register an inplace operation for an array that isn't using a defined writing strategy."),
        };

        let item = match self.input_bindings.get_mut(mapping.pos_input) {
            Some(binding) => {
                // Update input visibility.
                binding.visibility = Visibility::ReadWrite;
                // Inputs modified inplace should be read without any specified layout.
                self.expansion
                    .scope
                    .update_read(mapping.pos_input as u16, ReadingStrategy::Plain);

                // Use the same item as the input.
                //
                // The output can be different (i.e inplace boolean operations on float bindings).
                binding.item
            }
            None => *item,
        };

        // Update the output.
        *output = OutputInfo::InputArrayWrite {
            item,
            input: mapping.pos_input as u16,
            local: *local,
            position: *position,
        };
    }
}

fn bool_item(ty: Item) -> Item {
    Item {
        elem: bool_elem(ty.elem),
        line_size: ty.line_size,
    }
}

pub fn bool_elem(elem: Elem) -> Elem {
    match elem {
        // U32 are used for bool tensors
        Elem::Bool => u32::as_elem(),
        _ => elem,
    }
}
