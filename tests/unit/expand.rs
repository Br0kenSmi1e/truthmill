use std::collections::BTreeMap;

use super::*;
use crate::model::{Einsum, LinComb, ValueId};

fn integer(value: i64) -> Coefficient {
    Coefficient::from_integer(value.into())
}

fn index(id: u64, domain: DomainId) -> Index {
    Index { id, domain }
}

#[test]
fn expands_an_einsum_contraction() {
    let i = IndexId(0);
    let j = IndexId(1);
    let k = IndexId(2);
    let di = DomainId(0);
    let dj = DomainId(1);
    let dk = DomainId(2);
    let program = Program {
        values: vec![ValueDef::Einsum(Einsum {
            coeff: integer(2),
            code: Code {
                inputs: vec![vec![i, j], vec![j, k]],
                output: vec![i, k],
                domains: BTreeMap::from([(i, di), (j, dj), (k, dk)]),
            },
            args: vec![ValueRef::Tensor(TensorId(0)), ValueRef::Tensor(TensorId(1))],
        })],
        outputs: vec![ValueId(0)],
    };

    assert_eq!(
        expand(&program),
        vec![Polynomial {
            output: vec![index(0, di), index(2, dk)],
            terms: vec![Term {
                coeff: integer(2),
                factors: vec![
                    Factor {
                        tensor: TensorId(0),
                        indices: vec![index(0, di), index(1, dj)],
                    },
                    Factor {
                        tensor: TensorId(1),
                        indices: vec![index(1, dj), index(2, dk)],
                    },
                ],
            }],
        }]
    );
}

#[test]
fn expands_a_diagonal() {
    let i = IndexId(0);
    let domain = DomainId(0);
    let program = Program {
        values: vec![ValueDef::Einsum(Einsum {
            coeff: integer(1),
            code: Code {
                inputs: vec![vec![i, i]],
                output: vec![i],
                domains: BTreeMap::from([(i, domain)]),
            },
            args: vec![ValueRef::Tensor(TensorId(0))],
        })],
        outputs: vec![ValueId(0)],
    };

    let polynomial = &expand(&program)[0];
    assert_eq!(polynomial.output, [index(0, domain)]);
    assert_eq!(
        polynomial.terms[0].factors[0].indices,
        [index(0, domain), index(0, domain)]
    );
}

#[test]
fn expands_a_contracted_hyperedge() {
    let i = IndexId(0);
    let domain = DomainId(0);
    let program = Program {
        values: vec![ValueDef::Einsum(Einsum {
            coeff: integer(1),
            code: Code {
                inputs: vec![vec![i], vec![i], vec![i]],
                output: vec![],
                domains: BTreeMap::from([(i, domain)]),
            },
            args: vec![
                ValueRef::Tensor(TensorId(0)),
                ValueRef::Tensor(TensorId(1)),
                ValueRef::Tensor(TensorId(2)),
            ],
        })],
        outputs: vec![ValueId(0)],
    };

    let polynomial = &expand(&program)[0];
    let factors = &polynomial.terms[0].factors;
    assert!(polynomial.output.is_empty());
    assert_eq!(factors[0].indices[0], factors[1].indices[0]);
    assert_eq!(factors[1].indices[0], factors[2].indices[0]);
}

#[test]
fn expands_an_indexed_linear_combination() {
    let i = IndexId(0);
    let j = IndexId(1);
    let domain = DomainId(0);
    let program = Program {
        values: vec![ValueDef::LinComb(LinComb {
            coeff: vec![integer(2), integer(-3)],
            code: Code {
                inputs: vec![vec![i, j], vec![j, i]],
                output: vec![i, j],
                domains: BTreeMap::from([(i, domain), (j, domain)]),
            },
            args: vec![ValueRef::Tensor(TensorId(0)), ValueRef::Tensor(TensorId(1))],
        })],
        outputs: vec![ValueId(0)],
    };

    let polynomial = &expand(&program)[0];
    assert_eq!(polynomial.output, [index(0, domain), index(1, domain)]);
    assert_eq!(polynomial.terms.len(), 2);
    assert_eq!(polynomial.terms[0].coeff, integer(2));
    assert_eq!(
        polynomial.terms[0].factors[0].indices,
        [index(0, domain), index(1, domain)]
    );
    assert_eq!(polynomial.terms[1].coeff, integer(-3));
    assert_eq!(
        polynomial.terms[1].factors[0].indices,
        [index(1, domain), index(0, domain)]
    );
}

#[test]
fn reindexes_an_intermediate_by_its_explicit_output() {
    let i = IndexId(0);
    let j = IndexId(1);
    let domain = DomainId(0);
    let program = Program {
        values: vec![
            ValueDef::Einsum(Einsum {
                coeff: integer(1),
                code: Code {
                    inputs: vec![vec![i, j]],
                    output: vec![i, j],
                    domains: BTreeMap::from([(i, domain), (j, domain)]),
                },
                args: vec![ValueRef::Tensor(TensorId(0))],
            }),
            ValueDef::LinComb(LinComb {
                coeff: vec![integer(1)],
                code: Code {
                    inputs: vec![vec![j, i]],
                    output: vec![i, j],
                    domains: BTreeMap::from([(i, domain), (j, domain)]),
                },
                args: vec![ValueRef::Value(ValueId(0))],
            }),
        ],
        outputs: vec![ValueId(1)],
    };

    let polynomial = &expand(&program)[0];
    assert_eq!(polynomial.output, [index(0, domain), index(1, domain)]);
    assert_eq!(
        polynomial.terms[0].factors[0].indices,
        [index(1, domain), index(0, domain)]
    );
}

#[test]
fn distributes_einsum_over_a_linear_combination() {
    let i = IndexId(0);
    let domain = DomainId(0);
    let program = Program {
        values: vec![
            ValueDef::LinComb(LinComb {
                coeff: vec![integer(2), integer(-3)],
                code: Code {
                    inputs: vec![vec![i], vec![i]],
                    output: vec![i],
                    domains: BTreeMap::from([(i, domain)]),
                },
                args: vec![ValueRef::Tensor(TensorId(0)), ValueRef::Tensor(TensorId(1))],
            }),
            ValueDef::Einsum(Einsum {
                coeff: integer(5),
                code: Code {
                    inputs: vec![vec![i], vec![i]],
                    output: vec![i],
                    domains: BTreeMap::from([(i, domain)]),
                },
                args: vec![ValueRef::Value(ValueId(0)), ValueRef::Tensor(TensorId(2))],
            }),
        ],
        outputs: vec![ValueId(1)],
    };

    let polynomial = &expand(&program)[0];
    assert_eq!(polynomial.terms.len(), 2);
    assert_eq!(polynomial.terms[0].coeff, integer(10));
    assert_eq!(polynomial.terms[0].factors.len(), 2);
    assert_eq!(polynomial.terms[1].coeff, integer(-15));
    assert_eq!(polynomial.terms[1].factors.len(), 2);
}

#[test]
fn freshens_private_indices_when_reusing_an_intermediate() {
    let i = IndexId(0);
    let j = IndexId(1);
    let k = IndexId(4);
    let domain = DomainId(0);
    let program = Program {
        values: vec![
            ValueDef::Einsum(Einsum {
                coeff: integer(1),
                code: Code {
                    inputs: vec![vec![i, j]],
                    output: vec![i],
                    domains: BTreeMap::from([(i, domain), (j, domain)]),
                },
                args: vec![ValueRef::Tensor(TensorId(0))],
            }),
            ValueDef::Einsum(Einsum {
                coeff: integer(1),
                code: Code {
                    inputs: vec![vec![k], vec![k]],
                    output: vec![],
                    domains: BTreeMap::from([(k, domain)]),
                },
                args: vec![ValueRef::Value(ValueId(0)), ValueRef::Value(ValueId(0))],
            }),
        ],
        outputs: vec![ValueId(1)],
    };

    let polynomial = &expand(&program)[0];
    let factors = &polynomial.terms[0].factors;
    assert_eq!(factors[0].indices, [index(4, domain), index(5, domain)]);
    assert_eq!(factors[1].indices, [index(4, domain), index(6, domain)]);
}

#[test]
fn freshens_private_indices_across_terms() {
    let domain = DomainId(0);
    let output = index(0, domain);
    let private = index(1, domain);
    let polynomial = Polynomial {
        output: vec![output],
        terms: vec![
            Term {
                coeff: integer(1),
                factors: vec![Factor {
                    tensor: TensorId(0),
                    indices: vec![output, private],
                }],
            },
            Term {
                coeff: integer(1),
                factors: vec![Factor {
                    tensor: TensorId(1),
                    indices: vec![output, private],
                }],
            },
        ],
    };
    let input = vec![index(2, domain)];
    let mut next_id = 3;

    let renamed = rename_polynomial(polynomial, input, &mut next_id);

    assert_eq!(
        renamed.terms[0].factors[0].indices,
        [index(2, domain), index(3, domain)]
    );
    assert_eq!(
        renamed.terms[1].factors[0].indices,
        [index(2, domain), index(4, domain)]
    );
    assert_eq!(next_id, 5);
}

#[test]
fn retains_output_indices_while_expanding_zero() {
    let i = IndexId(0);
    let domain = DomainId(0);
    let program = Program {
        values: vec![ValueDef::LinComb(LinComb {
            coeff: vec![],
            code: Code {
                inputs: vec![],
                output: vec![i],
                domains: BTreeMap::from([(i, domain)]),
            },
            args: vec![],
        })],
        outputs: vec![ValueId(0)],
    };

    let polynomial = &expand(&program)[0];
    assert_eq!(polynomial.output, [index(0, domain)]);
    assert!(polynomial.terms.is_empty());
}
