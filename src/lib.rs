//! A trusted verifier and cost evaluator for tensor programs.

pub mod ir;

mod validate;

pub use validate::{ValidationError, validate};
