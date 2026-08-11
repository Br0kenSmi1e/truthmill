//! A trusted verifier and cost evaluator for tensor programs.

pub mod ir;

// Canonical verification will consume this module; its behavior is exercised
// directly until that pipeline is connected.
#[cfg_attr(not(test), allow(dead_code))]
mod expand;
mod validate;

pub use validate::{ValidationError, validate};
