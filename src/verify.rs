//! Exact equivalence verification for tensor programs.

use crate::canon::canonicalize;
use crate::expand::expand;
use crate::model::{Problem, Program};
use crate::validate::{ValidationError, validate};

/// Verify that a candidate is valid under `problem` and exactly equivalent to
/// its trusted reference program.
pub fn verify(problem: &Problem, candidate: &Program) -> Result<bool, ValidationError> {
    validate(problem)?;
    validate(&Problem {
        sizes: problem.sizes.clone(),
        symmetries: problem.symmetries.clone(),
        reference: candidate.clone(),
    })?;

    let reference = expand(&problem.reference)
        .into_iter()
        .map(|polynomial| canonicalize(polynomial, &problem.symmetries))
        .collect::<Vec<_>>();
    let candidate = expand(candidate)
        .into_iter()
        .map(|polynomial| canonicalize(polynomial, &problem.symmetries))
        .collect::<Vec<_>>();
    Ok(reference == candidate)
}
