//! Floating-point operation cost models for tensor SSA programs.

mod numerical_flops;
mod symbolic_flops;

pub use numerical_flops::{CostError, log_flops};
pub use symbolic_flops::{FlopTerm, SymbolicFlops, symbolic_flops};
