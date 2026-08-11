//! Validation of problem declarations and reference programs.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::model::{Code, DomainId, IndexId, Problem, TensorId, ValueDef, ValueId, ValueRef};

/// A reason that a [`Problem`] is not well formed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationError {
    ZeroDomainSize {
        domain: DomainId,
    },
    ProgramTooLarge {
        values: usize,
    },
    ArgumentCountMismatch {
        value: ValueId,
        inputs: usize,
        args: usize,
    },
    MissingIndexDomain {
        value: ValueId,
        index: IndexId,
    },
    UnusedIndexDomain {
        value: ValueId,
        index: IndexId,
    },
    UndeclaredDomain {
        value: ValueId,
        index: IndexId,
        domain: DomainId,
    },
    DuplicateOutputIndex {
        value: ValueId,
        index: IndexId,
    },
    UndeclaredTensor {
        value: ValueId,
        tensor: TensorId,
    },
    InvalidValueReference {
        value: ValueId,
        referenced: ValueId,
    },
    InconsistentTensorRank {
        value: ValueId,
        tensor: TensorId,
        expected: usize,
        actual: usize,
    },
    ValueInterfaceMismatch {
        value: ValueId,
        argument: usize,
        referenced: ValueId,
    },
    EmptyEinsum {
        value: ValueId,
    },
    EinsumOutputNotInput {
        value: ValueId,
        index: IndexId,
    },
    CoefficientCountMismatch {
        value: ValueId,
        coefficients: usize,
        args: usize,
    },
    InvalidLinCombInput {
        value: ValueId,
        input: usize,
    },
    InvalidOutput {
        output: usize,
        value: ValueId,
    },
    InvalidSymmetrySign {
        tensor: TensorId,
        generator: usize,
        sign: i8,
    },
    SymmetryRankMismatch {
        tensor: TensorId,
        generator: usize,
        expected: usize,
        actual: usize,
    },
    InvalidSymmetryPermutation {
        tensor: TensorId,
        generator: usize,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDomainSize { domain } => {
                write!(f, "domain {domain:?} has size zero")
            }
            Self::ProgramTooLarge { values } => {
                write!(
                    f,
                    "program has {values} values, which cannot be addressed by ValueId"
                )
            }
            Self::ArgumentCountMismatch {
                value,
                inputs,
                args,
            } => write!(
                f,
                "value {value:?} has {inputs} code inputs but {args} arguments"
            ),
            Self::MissingIndexDomain { value, index } => {
                write!(
                    f,
                    "value {value:?} does not declare a domain for index {index:?}"
                )
            }
            Self::UnusedIndexDomain { value, index } => {
                write!(f, "value {value:?} declares unused index {index:?}")
            }
            Self::UndeclaredDomain {
                value,
                index,
                domain,
            } => write!(
                f,
                "value {value:?} maps index {index:?} to undeclared domain {domain:?}"
            ),
            Self::DuplicateOutputIndex { value, index } => {
                write!(f, "value {value:?} repeats output index {index:?}")
            }
            Self::UndeclaredTensor { value, tensor } => {
                write!(f, "value {value:?} references undeclared tensor {tensor:?}")
            }
            Self::InvalidValueReference { value, referenced } => write!(
                f,
                "value {value:?} references non-prior value {referenced:?}"
            ),
            Self::InconsistentTensorRank {
                value,
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "value {value:?} uses tensor {tensor:?} with rank {actual}, expected {expected}"
            ),
            Self::ValueInterfaceMismatch {
                value,
                argument,
                referenced,
            } => write!(
                f,
                "argument {argument} of value {value:?} does not match the interface of {referenced:?}"
            ),
            Self::EmptyEinsum { value } => {
                write!(f, "value {value:?} is an einsum with no arguments")
            }
            Self::EinsumOutputNotInput { value, index } => write!(
                f,
                "output index {index:?} of einsum {value:?} does not occur in an input"
            ),
            Self::CoefficientCountMismatch {
                value,
                coefficients,
                args,
            } => write!(
                f,
                "linear combination {value:?} has {coefficients} coefficients but {args} arguments"
            ),
            Self::InvalidLinCombInput { value, input } => write!(
                f,
                "input {input} of linear combination {value:?} is not a permutation of its output"
            ),
            Self::InvalidOutput { output, value } => {
                write!(
                    f,
                    "program output {output} references missing value {value:?}"
                )
            }
            Self::InvalidSymmetrySign {
                tensor,
                generator,
                sign,
            } => write!(
                f,
                "symmetry generator {generator} of tensor {tensor:?} has invalid sign {sign}"
            ),
            Self::SymmetryRankMismatch {
                tensor,
                generator,
                expected,
                actual,
            } => write!(
                f,
                "symmetry generator {generator} of tensor {tensor:?} has rank {actual}, expected {expected}"
            ),
            Self::InvalidSymmetryPermutation { tensor, generator } => write!(
                f,
                "symmetry generator {generator} of tensor {tensor:?} is not a permutation"
            ),
        }
    }
}

impl Error for ValidationError {}

/// Validate a problem and its trusted reference program.
pub fn validate(problem: &Problem) -> Result<(), ValidationError> {
    for (&domain, &size) in &problem.sizes {
        if size == 0 {
            return Err(ValidationError::ZeroDomainSize { domain });
        }
    }

    if let Some(last_value) = problem.reference.values.len().checked_sub(1)
        && u32::try_from(last_value).is_err()
    {
        return Err(ValidationError::ProgramTooLarge {
            values: problem.reference.values.len(),
        });
    }

    let mut tensor_ranks = BTreeMap::new();
    let mut value_interfaces = Vec::with_capacity(problem.reference.values.len());

    for (position, definition) in problem.reference.values.iter().enumerate() {
        let value = ValueId(position as u32);
        let (code, args) = match definition {
            ValueDef::Einsum(einsum) => (&einsum.code, &einsum.args),
            ValueDef::LinComb(lincomb) => (&lincomb.code, &lincomb.args),
        };

        validate_code(problem, value, code, args.len())?;

        match definition {
            ValueDef::Einsum(einsum) => validate_einsum(value, &einsum.code, einsum.args.len())?,
            ValueDef::LinComb(lincomb) => validate_lincomb(
                value,
                &lincomb.code,
                lincomb.coeff.len(),
                lincomb.args.len(),
            )?,
        }

        validate_arguments(
            problem,
            value,
            code,
            args,
            &value_interfaces,
            &mut tensor_ranks,
        )?;
        value_interfaces.push(output_interface(code));
    }

    for (output, &value) in problem.reference.outputs.iter().enumerate() {
        if value.0 as usize >= value_interfaces.len() {
            return Err(ValidationError::InvalidOutput { output, value });
        }
    }

    validate_symmetries(problem, &tensor_ranks)
}

fn validate_code(
    problem: &Problem,
    value: ValueId,
    code: &Code,
    argument_count: usize,
) -> Result<(), ValidationError> {
    if code.inputs.len() != argument_count {
        return Err(ValidationError::ArgumentCountMismatch {
            value,
            inputs: code.inputs.len(),
            args: argument_count,
        });
    }

    let mut output_indices = BTreeSet::new();
    for &index in &code.output {
        if !output_indices.insert(index) {
            return Err(ValidationError::DuplicateOutputIndex { value, index });
        }
    }

    let used_indices: BTreeSet<_> = code
        .inputs
        .iter()
        .flatten()
        .chain(&code.output)
        .copied()
        .collect();

    for &index in &used_indices {
        if !code.domains.contains_key(&index) {
            return Err(ValidationError::MissingIndexDomain { value, index });
        }
    }

    for (&index, &domain) in &code.domains {
        if !problem.sizes.contains_key(&domain) {
            return Err(ValidationError::UndeclaredDomain {
                value,
                index,
                domain,
            });
        }
        if !used_indices.contains(&index) {
            return Err(ValidationError::UnusedIndexDomain { value, index });
        }
    }

    Ok(())
}

fn validate_einsum(
    value: ValueId,
    code: &Code,
    argument_count: usize,
) -> Result<(), ValidationError> {
    if argument_count == 0 {
        return Err(ValidationError::EmptyEinsum { value });
    }

    let input_indices: BTreeSet<_> = code.inputs.iter().flatten().copied().collect();
    for &index in &code.output {
        if !input_indices.contains(&index) {
            return Err(ValidationError::EinsumOutputNotInput { value, index });
        }
    }

    Ok(())
}

fn validate_lincomb(
    value: ValueId,
    code: &Code,
    coefficient_count: usize,
    argument_count: usize,
) -> Result<(), ValidationError> {
    if coefficient_count != argument_count {
        return Err(ValidationError::CoefficientCountMismatch {
            value,
            coefficients: coefficient_count,
            args: argument_count,
        });
    }

    let output_indices: BTreeSet<_> = code.output.iter().copied().collect();
    for (input, indices) in code.inputs.iter().enumerate() {
        let input_indices: BTreeSet<_> = indices.iter().copied().collect();
        if indices.len() != code.output.len() || input_indices != output_indices {
            return Err(ValidationError::InvalidLinCombInput { value, input });
        }
    }

    Ok(())
}

fn validate_arguments(
    problem: &Problem,
    value: ValueId,
    code: &Code,
    args: &[ValueRef],
    value_interfaces: &[Vec<DomainId>],
    tensor_ranks: &mut BTreeMap<TensorId, usize>,
) -> Result<(), ValidationError> {
    for (argument, (&arg, indices)) in args.iter().zip(&code.inputs).enumerate() {
        match arg {
            ValueRef::Tensor(tensor) => {
                if !problem.symmetries.contains_key(&tensor) {
                    return Err(ValidationError::UndeclaredTensor { value, tensor });
                }

                let actual = indices.len();
                if let Some(&expected) = tensor_ranks.get(&tensor) {
                    if actual != expected {
                        return Err(ValidationError::InconsistentTensorRank {
                            value,
                            tensor,
                            expected,
                            actual,
                        });
                    }
                } else {
                    tensor_ranks.insert(tensor, actual);
                }
            }
            ValueRef::Value(referenced) => {
                let Some(expected) = value_interfaces.get(referenced.0 as usize) else {
                    return Err(ValidationError::InvalidValueReference { value, referenced });
                };
                let actual = input_interface(code, indices);
                if actual != *expected {
                    return Err(ValidationError::ValueInterfaceMismatch {
                        value,
                        argument,
                        referenced,
                    });
                }
            }
        }
    }

    Ok(())
}

fn input_interface(code: &Code, indices: &[IndexId]) -> Vec<DomainId> {
    indices.iter().map(|index| code.domains[index]).collect()
}

fn output_interface(code: &Code) -> Vec<DomainId> {
    input_interface(code, &code.output)
}

fn validate_symmetries(
    problem: &Problem,
    tensor_ranks: &BTreeMap<TensorId, usize>,
) -> Result<(), ValidationError> {
    for (&tensor, generators) in &problem.symmetries {
        let Some(&rank) = tensor_ranks.get(&tensor) else {
            continue;
        };

        for (generator, symmetry) in generators.iter().enumerate() {
            if symmetry.sign != 1 && symmetry.sign != -1 {
                return Err(ValidationError::InvalidSymmetrySign {
                    tensor,
                    generator,
                    sign: symmetry.sign,
                });
            }
            if symmetry.permutation.len() != rank {
                return Err(ValidationError::SymmetryRankMismatch {
                    tensor,
                    generator,
                    expected: rank,
                    actual: symmetry.permutation.len(),
                });
            }

            let mut seen = vec![false; rank];
            for &axis in &symmetry.permutation {
                let Some(was_seen) = seen.get_mut(axis) else {
                    return Err(ValidationError::InvalidSymmetryPermutation { tensor, generator });
                };
                if *was_seen {
                    return Err(ValidationError::InvalidSymmetryPermutation { tensor, generator });
                }
                *was_seen = true;
            }
        }
    }

    Ok(())
}
