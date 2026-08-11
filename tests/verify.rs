use std::collections::BTreeMap;

use truthmill::*;

fn integer(value: i64) -> Coefficient {
    Coefficient::from_integer(value.into())
}

fn contraction_problem() -> Problem {
    let i = IndexId(0);
    let j = IndexId(1);
    let k = IndexId(2);
    let domain = DomainId(0);

    Problem {
        sizes: BTreeMap::from([(domain, 10)]),
        symmetries: BTreeMap::from([(TensorId(0), vec![]), (TensorId(1), vec![])]),
        reference: Program {
            values: vec![ValueDef::Einsum(Einsum {
                coeff: integer(1),
                code: Code {
                    inputs: vec![vec![i, j], vec![j, k]],
                    output: vec![i, k],
                    domains: BTreeMap::from([(i, domain), (j, domain), (k, domain)]),
                },
                args: vec![ValueRef::Tensor(TensorId(0)), ValueRef::Tensor(TensorId(1))],
            })],
            outputs: vec![ValueId(0)],
        },
    }
}

#[test]
fn accepts_the_reference_program() {
    let problem = contraction_problem();

    assert_eq!(verify(&problem, &problem.reference), Ok(true));
}

#[test]
fn accepts_dummy_renaming_and_factor_reordering() {
    let problem = contraction_problem();
    let p = IndexId(7);
    let q = IndexId(8);
    let r = IndexId(9);
    let domain = DomainId(0);
    let candidate = Program {
        values: vec![ValueDef::Einsum(Einsum {
            coeff: integer(1),
            code: Code {
                inputs: vec![vec![q, r], vec![p, q]],
                output: vec![p, r],
                domains: BTreeMap::from([(p, domain), (q, domain), (r, domain)]),
            },
            args: vec![ValueRef::Tensor(TensorId(1)), ValueRef::Tensor(TensorId(0))],
        })],
        outputs: vec![ValueId(0)],
    };

    assert_eq!(verify(&problem, &candidate), Ok(true));
}

#[test]
fn accepts_declared_signed_symmetry() {
    let i = IndexId(0);
    let j = IndexId(1);
    let domain = DomainId(0);
    let problem = Problem {
        sizes: BTreeMap::from([(domain, 10)]),
        symmetries: BTreeMap::from([(
            TensorId(0),
            vec![SymmetryGenerator {
                permutation: vec![1, 0],
                sign: -1,
            }],
        )]),
        reference: Program {
            values: vec![ValueDef::Einsum(Einsum {
                coeff: integer(1),
                code: Code {
                    inputs: vec![vec![i, j]],
                    output: vec![i, j],
                    domains: BTreeMap::from([(i, domain), (j, domain)]),
                },
                args: vec![ValueRef::Tensor(TensorId(0))],
            })],
            outputs: vec![ValueId(0)],
        },
    };
    let candidate = Program {
        values: vec![ValueDef::Einsum(Einsum {
            coeff: integer(-1),
            code: Code {
                inputs: vec![vec![j, i]],
                output: vec![i, j],
                domains: BTreeMap::from([(i, domain), (j, domain)]),
            },
            args: vec![ValueRef::Tensor(TensorId(0))],
        })],
        outputs: vec![ValueId(0)],
    };

    assert_eq!(verify(&problem, &candidate), Ok(true));
}

#[test]
fn accepts_distribution_across_an_intermediate_linear_combination() {
    let i = IndexId(0);
    let domain = DomainId(0);
    let code = || Code {
        inputs: vec![vec![i], vec![i]],
        output: vec![i],
        domains: BTreeMap::from([(i, domain)]),
    };
    let problem = Problem {
        sizes: BTreeMap::from([(domain, 10)]),
        symmetries: BTreeMap::from([
            (TensorId(0), vec![]),
            (TensorId(1), vec![]),
            (TensorId(2), vec![]),
        ]),
        reference: Program {
            values: vec![
                ValueDef::LinComb(LinComb {
                    coeff: vec![integer(1), integer(1)],
                    code: code(),
                    args: vec![ValueRef::Tensor(TensorId(0)), ValueRef::Tensor(TensorId(1))],
                }),
                ValueDef::Einsum(Einsum {
                    coeff: integer(1),
                    code: code(),
                    args: vec![ValueRef::Value(ValueId(0)), ValueRef::Tensor(TensorId(2))],
                }),
            ],
            outputs: vec![ValueId(1)],
        },
    };
    let candidate = Program {
        values: vec![
            ValueDef::Einsum(Einsum {
                coeff: integer(1),
                code: code(),
                args: vec![ValueRef::Tensor(TensorId(0)), ValueRef::Tensor(TensorId(2))],
            }),
            ValueDef::Einsum(Einsum {
                coeff: integer(1),
                code: code(),
                args: vec![ValueRef::Tensor(TensorId(1)), ValueRef::Tensor(TensorId(2))],
            }),
            ValueDef::LinComb(LinComb {
                coeff: vec![integer(1), integer(1)],
                code: code(),
                args: vec![ValueRef::Value(ValueId(0)), ValueRef::Value(ValueId(1))],
            }),
        ],
        outputs: vec![ValueId(2)],
    };

    assert_eq!(verify(&problem, &candidate), Ok(true));
}

#[test]
fn rejects_a_different_polynomial() {
    let problem = contraction_problem();
    let mut candidate = problem.reference.clone();
    let ValueDef::Einsum(einsum) = &mut candidate.values[0] else {
        unreachable!()
    };
    einsum.coeff = integer(2);

    assert_eq!(verify(&problem, &candidate), Ok(false));
}

#[test]
fn rejects_an_invalid_candidate_program() {
    let problem = contraction_problem();
    let mut candidate = problem.reference.clone();
    let ValueDef::Einsum(einsum) = &mut candidate.values[0] else {
        unreachable!()
    };
    einsum.args[0] = ValueRef::Value(ValueId(0));

    assert_eq!(
        verify(&problem, &candidate),
        Err(ValidationError::InvalidValueReference {
            value: ValueId(0),
            referenced: ValueId(0),
        })
    );
}

#[test]
fn rejects_a_candidate_with_the_wrong_output_count() {
    let problem = contraction_problem();
    let mut candidate = problem.reference.clone();
    candidate.outputs.clear();

    assert_eq!(verify(&problem, &candidate), Ok(false));
}

#[test]
fn accepts_zeros_with_different_output_interfaces() {
    let i = IndexId(0);
    let domain = DomainId(0);
    let problem = Problem {
        sizes: BTreeMap::from([(domain, 10)]),
        symmetries: BTreeMap::new(),
        reference: Program {
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
        },
    };
    let candidate = Program {
        values: vec![ValueDef::LinComb(LinComb {
            coeff: vec![],
            code: Code {
                inputs: vec![],
                output: vec![],
                domains: BTreeMap::new(),
            },
            args: vec![],
        })],
        outputs: vec![ValueId(0)],
    };

    assert_eq!(verify(&problem, &candidate), Ok(true));
}

#[test]
fn rejects_a_candidate_tensor_rank_change() {
    let problem = contraction_problem();
    let i = IndexId(0);
    let domain = DomainId(0);
    let candidate = Program {
        values: vec![ValueDef::Einsum(Einsum {
            coeff: integer(1),
            code: Code {
                inputs: vec![vec![i]],
                output: vec![i],
                domains: BTreeMap::from([(i, domain)]),
            },
            args: vec![ValueRef::Tensor(TensorId(0))],
        })],
        outputs: vec![ValueId(0)],
    };

    assert_eq!(verify(&problem, &candidate), Ok(false));
}
