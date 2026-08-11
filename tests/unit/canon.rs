use std::collections::BTreeMap;

use super::*;
use crate::model::DomainId;

fn integer(value: i64) -> Coefficient {
    Coefficient::from_integer(value.into())
}

fn index(id: u64, domain: u32) -> Index {
    Index {
        id,
        domain: DomainId(domain),
    }
}

fn factor(tensor: u32, indices: Vec<Index>) -> Factor {
    Factor {
        tensor: TensorId(tensor),
        indices,
    }
}

#[test]
fn compares_all_tensor_ids_before_indices() {
    let left = Term {
        coeff: integer(1),
        factors: vec![factor(0, vec![index(0, 0)]), factor(2, vec![index(0, 0)])],
    };
    let right = Term {
        coeff: integer(1),
        factors: vec![factor(0, vec![index(1, 0)]), factor(1, vec![index(0, 0)])],
    };

    assert_eq!(compare_term_bodies(&left, &right), Ordering::Greater);
}

#[test]
fn canonicalizes_factor_order_and_dummy_names_then_merges() {
    let output = vec![index(40, 0), index(41, 0)];
    let polynomial = Polynomial {
        output,
        terms: vec![
            Term {
                coeff: integer(2),
                factors: vec![
                    factor(0, vec![index(40, 0), index(10, 0)]),
                    factor(0, vec![index(10, 0), index(41, 0)]),
                ],
            },
            Term {
                coeff: integer(3),
                factors: vec![
                    factor(0, vec![index(20, 0), index(41, 0)]),
                    factor(0, vec![index(40, 0), index(20, 0)]),
                ],
            },
        ],
    };

    assert_eq!(
        canonicalize(polynomial, &BTreeMap::new()),
        Polynomial {
            output: vec![index(0, 0), index(1, 0)],
            terms: vec![Term {
                coeff: integer(5),
                factors: vec![
                    factor(0, vec![index(0, 0), index(2, 0)]),
                    factor(0, vec![index(2, 0), index(1, 0)]),
                ],
            }],
        }
    );
}

#[test]
fn applies_signed_tensor_symmetry() {
    let i = index(10, 0);
    let j = index(11, 0);
    let polynomial = Polynomial {
        output: vec![i, j],
        terms: vec![Term {
            coeff: integer(2),
            factors: vec![factor(0, vec![j, i])],
        }],
    };
    let symmetries = BTreeMap::from([(
        TensorId(0),
        vec![SymmetryGenerator {
            permutation: vec![1, 0],
            sign: -1,
        }],
    )]);

    assert_eq!(
        canonicalize(polynomial, &symmetries),
        Polynomial {
            output: vec![index(0, 0), index(1, 0)],
            terms: vec![Term {
                coeff: integer(-2),
                factors: vec![factor(0, vec![index(0, 0), index(1, 0)])],
            }],
        }
    );
}

#[test]
fn negative_stabilizer_makes_a_term_zero() {
    let i = index(10, 0);
    let polynomial = Polynomial {
        output: vec![i],
        terms: vec![Term {
            coeff: integer(1),
            factors: vec![factor(0, vec![i, i])],
        }],
    };
    let symmetries = BTreeMap::from([(
        TensorId(0),
        vec![SymmetryGenerator {
            permutation: vec![1, 0],
            sign: -1,
        }],
    )]);

    assert_eq!(
        canonicalize(polynomial, &symmetries),
        Polynomial {
            output: Vec::new(),
            terms: Vec::new(),
        }
    );
}

#[test]
fn symmetry_generators_are_closed_into_a_group() {
    let input = factor(0, vec![index(2, 0), index(0, 0), index(1, 0)]);
    let generators = vec![SymmetryGenerator {
        permutation: vec![1, 2, 0],
        sign: 1,
    }];

    let variants = factor_symmetry_candidates(&input, &generators);

    assert_eq!(variants.len(), 3);
    assert!(variants.contains(&(factor(0, vec![index(0, 0), index(1, 0), index(2, 0)]), 1,)));
}

#[test]
fn sorts_and_cancels_canonical_polynomial_terms() {
    let i = index(10, 0);
    let polynomial = Polynomial {
        output: vec![i],
        terms: vec![
            Term {
                coeff: integer(4),
                factors: vec![factor(2, vec![i])],
            },
            Term {
                coeff: integer(3),
                factors: vec![factor(1, vec![i])],
            },
            Term {
                coeff: integer(-4),
                factors: vec![factor(2, vec![i])],
            },
            Term {
                coeff: integer(2),
                factors: vec![factor(0, vec![i])],
            },
        ],
    };

    assert_eq!(
        canonicalize(polynomial, &BTreeMap::new()),
        Polynomial {
            output: vec![index(0, 0)],
            terms: vec![
                Term {
                    coeff: integer(2),
                    factors: vec![factor(0, vec![index(0, 0)])],
                },
                Term {
                    coeff: integer(3),
                    factors: vec![factor(1, vec![index(0, 0)])],
                },
            ],
        }
    );
}
