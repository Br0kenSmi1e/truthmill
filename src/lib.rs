//! A trusted verifier and cost evaluator for tensor programs.

mod model;

// Exact verification will consume these internal stages; their behavior is
// exercised directly until that pipeline is connected.
#[cfg_attr(not(test), allow(dead_code))]
mod canon;
// Canonical verification will consume this module; its behavior is exercised
// directly until that pipeline is connected.
#[cfg_attr(not(test), allow(dead_code))]
mod expand;
mod validate;

pub use model::{
    Code, Coefficient, DomainId, Einsum, IndexId, LinComb, Problem, Program, SymmetryGenerator,
    TensorId, ValueDef, ValueId, ValueRef,
};
pub use validate::{ValidationError, validate};
