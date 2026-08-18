//! Numerical floating-point operation counts for tensor SSA programs.

use std::error::Error;
use std::fmt;

use crate::model::{Problem, Program, ValueDef};
use crate::validate::{ValidationError, validate};

/// A failure to evaluate the numerical FLOP cost of a program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CostError {
    Invalid(ValidationError),
    ZeroTotalFlops,
}

impl fmt::Display for CostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(error) => write!(f, "invalid computation: {error}"),
            Self::ZeroTotalFlops => write!(f, "computation has zero modeled FLOPs"),
        }
    }
}

impl Error for CostError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Invalid(error) => Some(error),
            Self::ZeroTotalFlops => None,
        }
    }
}

impl From<ValidationError> for CostError {
    fn from(error: ValidationError) -> Self {
        Self::Invalid(error)
    }
}

/// Calculate the natural logarithm of a candidate's total modeled FLOPs.
///
/// Every SSA definition is charged once. Tensor multiplications and additions
/// are counted, while multiplication by numeric coefficients, copying, and
/// initialization are ignored.
pub fn log_flops(problem: &Problem, candidate: &Program) -> Result<f64, CostError> {
    validate(problem)?;
    validate(&Problem {
        sizes: problem.sizes.clone(),
        symmetries: problem.symmetries.clone(),
        reference: candidate.clone(),
    })?;

    let mut total = None;

    for definition in &candidate.values {
        let (code, operations) = match definition {
            ValueDef::Einsum(einsum) => {
                let has_contraction =
                    usize::from(einsum.code.output.len() < einsum.code.domains.len());
                (
                    &einsum.code,
                    einsum.args.len().saturating_sub(1) + has_contraction,
                )
            }
            ValueDef::LinComb(lincomb) => (&lincomb.code, lincomb.args.len().saturating_sub(1)),
        };

        if operations > 0 {
            let log_size = code
                .domains
                .values()
                .map(|domain| (problem.sizes[domain] as f64).ln())
                .sum::<f64>();
            add_log_cost(&mut total, (operations as f64).ln() + log_size);
        }
    }

    total.ok_or(CostError::ZeroTotalFlops)
}

fn add_log_cost(total: &mut Option<f64>, cost: f64) {
    *total = Some(match *total {
        Some(current) => logaddexp(current, cost),
        None => cost,
    });
}

fn logaddexp(left: f64, right: f64) -> f64 {
    let maximum = left.max(right);
    maximum + (-(left - right).abs()).exp().ln_1p()
}
