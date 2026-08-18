//! Symbolic floating-point operation counts for tensor SSA programs.

use std::collections::BTreeMap;
use std::fmt;

use crate::model::{DomainId, Problem, Program, ValueDef};
use crate::validate::{ValidationError, validate};

/// An exact term in a symbolic FLOP count.
///
/// A term represents `coefficient * product(D_domain ^ exponent)`, where each
/// `D_domain` is the size of one problem domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlopTerm {
    coefficient: u128,
    powers: BTreeMap<DomainId, usize>,
}

impl FlopTerm {
    /// Return the integer multiplier of this term.
    pub fn coefficient(&self) -> u128 {
        self.coefficient
    }

    /// Return the exponent of every domain size present in this term.
    pub fn powers(&self) -> &BTreeMap<DomainId, usize> {
        &self.powers
    }

    /// Return the exponent of `domain`, or zero when it is absent.
    pub fn exponent(&self, domain: DomainId) -> usize {
        self.powers.get(&domain).copied().unwrap_or(0)
    }

    /// Return the degree of this term when all domain sizes scale together.
    pub fn total_degree(&self) -> usize {
        self.powers.values().sum()
    }
}

impl fmt::Display for FlopTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut has_factor = false;

        if self.coefficient != 1 || self.powers.is_empty() {
            write!(f, "{}", self.coefficient)?;
            has_factor = true;
        }

        for (domain, exponent) in &self.powers {
            if has_factor {
                write!(f, " * ")?;
            }
            write!(f, "D{}", domain.0)?;
            if *exponent != 1 {
                write!(f, "^{exponent}")?;
            }
            has_factor = true;
        }

        Ok(())
    }
}

/// An exact sparse polynomial in problem-domain sizes.
///
/// Terms are canonicalized by combining equal powers and are ordered by
/// descending total degree. In the display form, `D0`, `D1`, and so on denote
/// the sizes of `DomainId(0)`, `DomainId(1)`, and so on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SymbolicFlops {
    terms: Vec<FlopTerm>,
}

impl SymbolicFlops {
    /// Return the nonzero terms in this FLOP polynomial.
    pub fn terms(&self) -> &[FlopTerm] {
        &self.terms
    }

    /// Return whether this program has no modeled FLOPs.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Return the scaling degree when every domain size grows uniformly.
    ///
    /// The zero polynomial has no degree.
    pub fn uniform_degree(&self) -> Option<usize> {
        self.terms.first().map(FlopTerm::total_degree)
    }

    /// Return the highest-order terms using the given domain order to break ties.
    ///
    /// Total degree is compared first. Terms tied on total degree are then
    /// compared lexicographically by exponent, with domains listed from most
    /// significant to least significant. An empty or partial `domain_order`
    /// can leave multiple terms tied. The zero polynomial returns an empty
    /// vector.
    pub fn highest_order_terms(&self, domain_order: &[DomainId]) -> Vec<&FlopTerm> {
        let Some(degree) = self.uniform_degree() else {
            return Vec::new();
        };

        let mut terms = self
            .terms
            .iter()
            .take_while(|term| term.total_degree() == degree)
            .collect::<Vec<_>>();
        for domain in domain_order {
            let exponent = terms
                .iter()
                .map(|term| term.exponent(*domain))
                .max()
                .expect("a nonzero polynomial has a highest-order term");
            terms.retain(|term| term.exponent(*domain) == exponent);
        }

        terms
    }

    /// Return the largest exponent of one domain size.
    ///
    /// The zero polynomial has no degree in any domain. A nonzero polynomial
    /// independent of `domain` has degree zero.
    pub fn degree_in(&self, domain: DomainId) -> Option<usize> {
        self.terms.iter().map(|term| term.exponent(domain)).max()
    }
}

impl fmt::Display for SymbolicFlops {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        for (index, term) in self.terms.iter().enumerate() {
            if index != 0 {
                write!(f, " + ")?;
            }
            write!(f, "{term}")?;
        }

        Ok(())
    }
}

/// Calculate a candidate's exact FLOP count as a polynomial in domain sizes.
///
/// Every SSA definition is charged once. Tensor multiplications and additions
/// are counted, while multiplication by numeric coefficients, copying, and
/// initialization are ignored. Equal monomials are combined. A computation
/// containing only unmodeled operations returns the zero polynomial.
pub fn symbolic_flops(
    problem: &Problem,
    candidate: &Program,
) -> Result<SymbolicFlops, ValidationError> {
    validate(problem)?;
    validate(&Problem {
        sizes: problem.sizes.clone(),
        symmetries: problem.symmetries.clone(),
        reference: candidate.clone(),
    })?;

    let mut coefficients = BTreeMap::<BTreeMap<DomainId, usize>, u128>::new();

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
            let mut powers = BTreeMap::new();
            for domain in code.domains.values() {
                *powers.entry(*domain).or_insert(0) += 1;
            }
            *coefficients.entry(powers).or_insert(0) += operations as u128;
        }
    }

    let mut terms = coefficients
        .into_iter()
        .map(|(powers, coefficient)| FlopTerm {
            coefficient,
            powers,
        })
        .collect::<Vec<_>>();
    terms.sort_by(|left, right| {
        right
            .total_degree()
            .cmp(&left.total_degree())
            .then_with(|| right.powers.cmp(&left.powers))
    });

    Ok(SymbolicFlops { terms })
}
