//! Canonicalization of expanded tensor polynomials.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::expand::{Factor, Index, Polynomial, Term};
use crate::model::{Coefficient, SymmetryGenerator, TensorId};

/// Canonicalize an expanded polynomial modulo dummy renaming, factor order,
/// and declared signed tensor symmetries.
pub(crate) fn canonicalize(
    polynomial: Polynomial,
    symmetries: &BTreeMap<TensorId, Vec<SymmetryGenerator>>,
) -> Polynomial {
    let output = polynomial.output;
    let mut terms = polynomial
        .terms
        .into_iter()
        .filter_map(|term| canonicalize_term(term, &output, symmetries))
        .collect::<Vec<_>>();

    terms.sort_by(compare_term_bodies);

    let mut merged = Vec::<Term>::with_capacity(terms.len());
    for term in terms {
        if let Some(previous) = merged.last_mut()
            && compare_term_bodies(previous, &term) == Ordering::Equal
        {
            previous.coeff += term.coeff;
        } else {
            merged.push(term);
        }
    }
    merged.retain(|term| !is_zero(&term.coeff));

    if merged.is_empty() {
        Polynomial {
            output: Vec::new(),
            terms: Vec::new(),
        }
    } else {
        Polynomial {
            output: canonical_output(&output),
            terms: merged,
        }
    }
}

fn canonicalize_term(
    term: Term,
    output: &[Index],
    symmetries: &BTreeMap<TensorId, Vec<SymmetryGenerator>>,
) -> Option<Term> {
    if is_zero(&term.coeff) {
        return None;
    }

    let orders = factor_order_candidates(&term.factors);
    let mut best = None::<Term>;
    let mut saw_positive = false;
    let mut saw_negative = false;

    for (factors, sign) in symmetry_candidates(&term.factors, symmetries) {
        for order in &orders {
            let ordered = order
                .iter()
                .map(|&position| factors[position].clone())
                .collect();
            let candidate = Term {
                coeff: term.coeff,
                factors: normalize_indices(ordered, output),
            };

            match best
                .as_ref()
                .map(|best| compare_term_bodies(&candidate, best))
            {
                None | Some(Ordering::Less) => {
                    best = Some(candidate);
                    saw_positive = sign == 1;
                    saw_negative = sign == -1;
                }
                Some(Ordering::Equal) => {
                    saw_positive |= sign == 1;
                    saw_negative |= sign == -1;
                }
                Some(Ordering::Greater) => {}
            }
        }
    }

    if saw_positive && saw_negative {
        return None;
    }

    let mut best = best.expect("identity symmetry and factor order always form a candidate");
    if saw_negative {
        best.coeff = -best.coeff;
    }

    Some(best)
}

/// Generate the signed Cartesian product of all factor symmetry variants.
fn symmetry_candidates(
    factors: &[Factor],
    symmetries: &BTreeMap<TensorId, Vec<SymmetryGenerator>>,
) -> Vec<(Vec<Factor>, i8)> {
    let mut product = vec![(Vec::new(), 1)];

    for factor in factors {
        let generators = symmetries
            .get(&factor.tensor)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let variants = factor_symmetry_candidates(factor, generators);
        let mut next = Vec::with_capacity(product.len().saturating_mul(variants.len()));

        for (prefix, prefix_sign) in product {
            for (variant, variant_sign) in &variants {
                let mut factors = prefix.clone();
                factors.push(variant.clone());
                next.push((factors, prefix_sign * variant_sign));
            }
        }

        product = next;
    }

    product
}

fn factor_symmetry_candidates(
    factor: &Factor,
    generators: &[SymmetryGenerator],
) -> Vec<(Factor, i8)> {
    let rank = factor.indices.len();
    let mut group = vec![((0..rank).collect::<Vec<_>>(), 1)];
    let mut position = 0;

    while position < group.len() {
        let (permutation, sign) = group[position].clone();
        for generator in generators {
            let element = (
                compose(&permutation, &generator.permutation),
                sign * generator.sign,
            );
            if !group.contains(&element) {
                group.push(element);
            }
        }
        position += 1;
    }

    let mut variants = Vec::new();
    for (permutation, sign) in group {
        let variant = Factor {
            tensor: factor.tensor,
            indices: permutation
                .iter()
                .map(|&position| factor.indices[position])
                .collect(),
        };
        if !variants.contains(&(variant.clone(), sign)) {
            variants.push((variant, sign));
        }
    }
    variants
}

/// Compose preimage permutations, applying `left` and then `right`.
fn compose(left: &[usize], right: &[usize]) -> Vec<usize> {
    right.iter().map(|&position| left[position]).collect()
}

/// Generate all factor orders that have sorted tensor IDs.
///
/// Factors with unequal tensor IDs have a fixed relative order. All orders
/// among occurrences of the same tensor are retained because index
/// normalization can distinguish them.
fn factor_order_candidates(factors: &[Factor]) -> Vec<Vec<usize>> {
    fn generate(
        factors: &[Factor],
        position: usize,
        current: &mut [usize],
        result: &mut Vec<Vec<usize>>,
    ) {
        if position == current.len() {
            result.push(current.to_vec());
            return;
        }

        let tensor = factors[current[position]].tensor;
        for next in position..current.len() {
            if factors[current[next]].tensor != tensor {
                continue;
            }
            current.swap(position, next);
            generate(factors, position + 1, current, result);
            current.swap(position, next);
        }
    }

    let mut current = (0..factors.len()).collect::<Vec<_>>();
    current.sort_by_key(|&position| factors[position].tensor);
    let mut result = Vec::new();
    generate(factors, 0, &mut current, &mut result);
    result
}

/// Fix output identities by position, then rename every dummy by first
/// occurrence in factor-and-slot order.
fn normalize_indices(mut factors: Vec<Factor>, output: &[Index]) -> Vec<Factor> {
    let mut rename = output
        .iter()
        .enumerate()
        .map(|(position, &index)| {
            (
                index,
                Index {
                    id: position as u64,
                    domain: index.domain,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut next_id = output.len() as u64;

    for factor in &mut factors {
        for index in &mut factor.indices {
            let canonical = *rename.entry(*index).or_insert_with(|| {
                let canonical = Index {
                    id: next_id,
                    domain: index.domain,
                };
                next_id += 1;
                canonical
            });
            *index = canonical;
        }
    }

    factors
}

fn canonical_output(output: &[Index]) -> Vec<Index> {
    output
        .iter()
        .enumerate()
        .map(|(position, index)| Index {
            id: position as u64,
            domain: index.domain,
        })
        .collect()
}

/// Compare term bodies by all tensor IDs first, then all factor indices.
fn compare_term_bodies(left: &Term, right: &Term) -> Ordering {
    left.factors
        .iter()
        .map(|factor| factor.tensor)
        .cmp(right.factors.iter().map(|factor| factor.tensor))
        .then_with(|| {
            left.factors
                .iter()
                .flat_map(|factor| &factor.indices)
                .cmp(right.factors.iter().flat_map(|factor| &factor.indices))
        })
}

fn is_zero(coeff: &Coefficient) -> bool {
    coeff == &Coefficient::from_integer(0)
}

#[cfg(test)]
#[path = "../tests/unit/canon.rs"]
mod tests;
