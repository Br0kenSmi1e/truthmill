//! Expansion of tensor SSA into exact sums of tensor products.

use std::collections::{BTreeMap, btree_map::Entry};

use crate::model::{
    Code, Coefficient, DomainId, Einsum, IndexId, LinComb, Program, TensorId, ValueDef, ValueRef,
};

/// An exact polynomial with an explicit ordered list of free indices.
///
/// `output` is expansion bookkeeping. Canonicalization may discard it for an
/// empty polynomial because Truthmill treats all zeros as equal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Polynomial {
    pub output: Vec<Index>,
    pub terms: Vec<Term>,
}

/// One term in an expanded tensor polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Term {
    pub coeff: Coefficient,
    pub factors: Vec<Factor>,
}

/// One occurrence of a problem tensor.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Factor {
    pub tensor: TensorId,
    pub indices: Vec<Index>,
}

/// One index identity in an expanded polynomial.
///
/// An index is free when it occurs in `Polynomial::output`; every other index
/// is local to that polynomial's terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Index {
    pub id: u64,
    pub domain: DomainId,
}

/// Expand every requested output of a validated Program.
pub(crate) fn expand(program: &Program) -> Vec<Polynomial> {
    let mut values = Vec::with_capacity(program.values.len());

    for definition in &program.values {
        let polynomial = match definition {
            ValueDef::Einsum(einsum) => expand_einsum(einsum, &values),
            ValueDef::LinComb(lincomb) => expand_lincomb(lincomb, &values),
        };
        values.push(polynomial);
    }

    program
        .outputs
        .iter()
        .map(|output| values[output.0 as usize].clone())
        .collect()
}

fn expand_einsum(einsum: &Einsum, values: &[Polynomial]) -> Polynomial {
    let mut next_id = next_private_id(&einsum.code);
    let mut terms = vec![Term {
        coeff: einsum.coeff.clone(),
        factors: Vec::new(),
    }];

    for (reference, code_input) in einsum.args.iter().zip(&einsum.code.inputs) {
        let input = code_indices(code_input, &einsum.code.domains);
        let polynomial = reference_polynomial(*reference, &input, values);
        let polynomial = rename_polynomial(polynomial, input, &mut next_id);
        terms = multiply_terms(terms, &polynomial.terms);
    }

    Polynomial {
        output: code_indices(&einsum.code.output, &einsum.code.domains),
        terms,
    }
}

fn expand_lincomb(lincomb: &LinComb, values: &[Polynomial]) -> Polynomial {
    let mut next_id = next_private_id(&lincomb.code);
    let mut terms = Vec::new();

    for ((coefficient, reference), code_input) in lincomb
        .coeff
        .iter()
        .zip(&lincomb.args)
        .zip(&lincomb.code.inputs)
    {
        let input = code_indices(code_input, &lincomb.code.domains);
        let polynomial = reference_polynomial(*reference, &input, values);
        let mut polynomial = rename_polynomial(polynomial, input, &mut next_id);
        for term in &mut polynomial.terms {
            term.coeff *= coefficient;
        }
        terms.extend(polynomial.terms);
    }

    Polynomial {
        output: code_indices(&lincomb.code.output, &lincomb.code.domains),
        terms,
    }
}

/// Turn one reference into a polynomial before renaming it for this use.
fn reference_polynomial(reference: ValueRef, input: &[Index], values: &[Polynomial]) -> Polynomial {
    match reference {
        ValueRef::Tensor(tensor) => Polynomial {
            output: input.to_vec(),
            terms: vec![Term {
                coeff: Coefficient::from_integer(1.into()),
                factors: vec![Factor {
                    tensor,
                    indices: input.to_vec(),
                }],
            }],
        },
        ValueRef::Value(value) => values[value.0 as usize].clone(),
    }
}

/// Wrap local IDs without changing their numeric identities.
fn code_indices(ids: &[IndexId], domains: &BTreeMap<IndexId, DomainId>) -> Vec<Index> {
    ids.iter()
        .map(|id| Index {
            id: u64::from(id.0),
            domain: domains[id],
        })
        .collect()
}

fn next_private_id(code: &Code) -> u64 {
    code.domains
        .keys()
        .map(|id| u64::from(id.0) + 1)
        .max()
        .unwrap_or(0)
}

/// Seed, grow, and apply one complete index rename.
fn rename_polynomial(polynomial: Polynomial, input: Vec<Index>, next_id: &mut u64) -> Polynomial {
    debug_assert_eq!(polynomial.output.len(), input.len());

    let seed = polynomial
        .output
        .iter()
        .copied()
        .zip(input.iter().copied())
        .collect::<BTreeMap<_, _>>();

    let mut terms = Vec::with_capacity(polynomial.terms.len());
    for term in polynomial.terms {
        let mut rename = seed.clone();
        for factor in &term.factors {
            for &index in &factor.indices {
                if let Entry::Vacant(entry) = rename.entry(index) {
                    entry.insert(Index {
                        id: *next_id,
                        domain: index.domain,
                    });
                    *next_id += 1;
                }
            }
        }

        terms.push(Term {
            coeff: term.coeff,
            factors: term
                .factors
                .into_iter()
                .map(|factor| Factor {
                    tensor: factor.tensor,
                    indices: factor
                        .indices
                        .into_iter()
                        .map(|index| rename[&index])
                        .collect(),
                })
                .collect(),
        });
    }

    Polynomial {
        output: input,
        terms,
    }
}

/// Multiply two term lists by distributing every left term over every right term.
fn multiply_terms(left: Vec<Term>, right: &[Term]) -> Vec<Term> {
    let mut product = Vec::with_capacity(left.len().saturating_mul(right.len()));
    for left_term in left {
        for right_term in right {
            let mut term = left_term.clone();
            term.coeff *= &right_term.coeff;
            term.factors.extend(right_term.factors.iter().cloned());
            product.push(term);
        }
    }
    product
}

#[cfg(test)]
#[path = "../tests/unit/expand.rs"]
mod tests;
