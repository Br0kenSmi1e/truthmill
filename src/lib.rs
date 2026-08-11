//! A trusted verifier and cost evaluator for tensor programs.

mod model;

mod canon;
mod cost;
mod expand;
mod validate;
mod verify;

pub use cost::{CostError, log_flops};
pub use model::{
    Code, Coefficient, DomainId, Einsum, IndexId, LinComb, Problem, Program, SymmetryGenerator,
    TensorId, ValueDef, ValueId, ValueRef,
};
pub use validate::{ValidationError, validate};
pub use verify::verify;
