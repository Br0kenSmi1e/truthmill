use std::collections::BTreeMap;

use truthmill::ir::*;
use truthmill::{ValidationError, validate};

fn one() -> Coefficient {
    Coefficient::from_integer(1.into())
}

fn valid_problem() -> Problem {
    let i = IndexId(0);
    let j = IndexId(1);
    let k = IndexId(2);
    let domain = DomainId(0);

    Problem {
        sizes: BTreeMap::from([(domain, 10)]),
        symmetries: BTreeMap::from([
            (
                TensorId(0),
                vec![SymmetryGenerator {
                    permutation: vec![1, 0],
                    sign: -1,
                }],
            ),
            (TensorId(1), vec![]),
        ]),
        reference: Program {
            values: vec![
                ValueDef::Einsum(Einsum {
                    coeff: one(),
                    code: Code {
                        inputs: vec![vec![i, j], vec![j, k]],
                        output: vec![i, k],
                        domains: BTreeMap::from([(i, domain), (j, domain), (k, domain)]),
                    },
                    args: vec![ValueRef::Tensor(TensorId(0)), ValueRef::Tensor(TensorId(1))],
                }),
                ValueDef::LinComb(LinComb {
                    coeff: vec![one()],
                    code: Code {
                        inputs: vec![vec![i, k]],
                        output: vec![i, k],
                        domains: BTreeMap::from([(i, domain), (k, domain)]),
                    },
                    args: vec![ValueRef::Value(ValueId(0))],
                }),
            ],
            outputs: vec![ValueId(1)],
        },
    }
}

#[test]
fn accepts_a_valid_problem() {
    assert_eq!(validate(&valid_problem()), Ok(()));
}

#[test]
fn rejects_a_zero_domain_size() {
    let mut problem = valid_problem();
    problem.sizes.insert(DomainId(0), 0);

    assert_eq!(
        validate(&problem),
        Err(ValidationError::ZeroDomainSize {
            domain: DomainId(0)
        })
    );
}

#[test]
fn rejects_a_forward_value_reference() {
    let mut problem = valid_problem();
    let ValueDef::Einsum(einsum) = &mut problem.reference.values[0] else {
        unreachable!()
    };
    einsum.args[0] = ValueRef::Value(ValueId(0));

    assert_eq!(
        validate(&problem),
        Err(ValidationError::InvalidValueReference {
            value: ValueId(0),
            referenced: ValueId(0),
        })
    );
}

#[test]
fn rejects_an_undeclared_domain() {
    let mut problem = valid_problem();
    let ValueDef::Einsum(einsum) = &mut problem.reference.values[0] else {
        unreachable!()
    };
    einsum.code.domains.insert(IndexId(1), DomainId(9));

    assert_eq!(
        validate(&problem),
        Err(ValidationError::UndeclaredDomain {
            value: ValueId(0),
            index: IndexId(1),
            domain: DomainId(9),
        })
    );
}

#[test]
fn rejects_an_argument_count_mismatch() {
    let mut problem = valid_problem();
    let ValueDef::Einsum(einsum) = &mut problem.reference.values[0] else {
        unreachable!()
    };
    einsum.args.pop();

    assert_eq!(
        validate(&problem),
        Err(ValidationError::ArgumentCountMismatch {
            value: ValueId(0),
            inputs: 2,
            args: 1,
        })
    );
}

#[test]
fn rejects_a_missing_index_domain() {
    let mut problem = valid_problem();
    let ValueDef::Einsum(einsum) = &mut problem.reference.values[0] else {
        unreachable!()
    };
    einsum.code.domains.remove(&IndexId(1));

    assert_eq!(
        validate(&problem),
        Err(ValidationError::MissingIndexDomain {
            value: ValueId(0),
            index: IndexId(1),
        })
    );
}

#[test]
fn rejects_an_unused_index_domain() {
    let mut problem = valid_problem();
    let ValueDef::Einsum(einsum) = &mut problem.reference.values[0] else {
        unreachable!()
    };
    einsum.code.domains.insert(IndexId(9), DomainId(0));

    assert_eq!(
        validate(&problem),
        Err(ValidationError::UnusedIndexDomain {
            value: ValueId(0),
            index: IndexId(9),
        })
    );
}

#[test]
fn rejects_a_duplicate_output_index() {
    let mut problem = valid_problem();
    let ValueDef::Einsum(einsum) = &mut problem.reference.values[0] else {
        unreachable!()
    };
    einsum.code.output = vec![IndexId(0), IndexId(0)];

    assert_eq!(
        validate(&problem),
        Err(ValidationError::DuplicateOutputIndex {
            value: ValueId(0),
            index: IndexId(0),
        })
    );
}

#[test]
fn rejects_an_undeclared_tensor() {
    let mut problem = valid_problem();
    let ValueDef::Einsum(einsum) = &mut problem.reference.values[0] else {
        unreachable!()
    };
    einsum.args[1] = ValueRef::Tensor(TensorId(9));

    assert_eq!(
        validate(&problem),
        Err(ValidationError::UndeclaredTensor {
            value: ValueId(0),
            tensor: TensorId(9),
        })
    );
}

#[test]
fn rejects_an_empty_einsum() {
    let mut problem = valid_problem();
    let ValueDef::Einsum(einsum) = &mut problem.reference.values[0] else {
        unreachable!()
    };
    einsum.code.inputs.clear();
    einsum.code.output.clear();
    einsum.code.domains.clear();
    einsum.args.clear();

    assert_eq!(
        validate(&problem),
        Err(ValidationError::EmptyEinsum { value: ValueId(0) })
    );
}

#[test]
fn rejects_an_einsum_output_not_present_in_an_input() {
    let mut problem = valid_problem();
    let ValueDef::Einsum(einsum) = &mut problem.reference.values[0] else {
        unreachable!()
    };
    einsum.code.output = vec![IndexId(9)];
    einsum.code.domains.insert(IndexId(9), DomainId(0));

    assert_eq!(
        validate(&problem),
        Err(ValidationError::EinsumOutputNotInput {
            value: ValueId(0),
            index: IndexId(9),
        })
    );
}

#[test]
fn rejects_inconsistent_tensor_rank() {
    let mut problem = valid_problem();
    let ValueDef::LinComb(lincomb) = &mut problem.reference.values[1] else {
        unreachable!()
    };
    lincomb.code.inputs = vec![vec![IndexId(0)]];
    lincomb.code.output = vec![IndexId(0)];
    lincomb.code.domains = BTreeMap::from([(IndexId(0), DomainId(0))]);
    lincomb.args = vec![ValueRef::Tensor(TensorId(0))];

    assert_eq!(
        validate(&problem),
        Err(ValidationError::InconsistentTensorRank {
            value: ValueId(1),
            tensor: TensorId(0),
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn rejects_a_computed_value_interface_mismatch() {
    let mut problem = valid_problem();
    problem.sizes.insert(DomainId(1), 5);
    let ValueDef::LinComb(lincomb) = &mut problem.reference.values[1] else {
        unreachable!()
    };
    lincomb.code.domains.insert(IndexId(2), DomainId(1));

    assert_eq!(
        validate(&problem),
        Err(ValidationError::ValueInterfaceMismatch {
            value: ValueId(1),
            argument: 0,
            referenced: ValueId(0),
        })
    );
}

#[test]
fn rejects_contraction_in_a_linear_combination() {
    let mut problem = valid_problem();
    let ValueDef::LinComb(lincomb) = &mut problem.reference.values[1] else {
        unreachable!()
    };
    lincomb.code.inputs[0] = vec![IndexId(0), IndexId(0)];

    assert_eq!(
        validate(&problem),
        Err(ValidationError::InvalidLinCombInput {
            value: ValueId(1),
            input: 0,
        })
    );
}

#[test]
fn rejects_a_linear_combination_coefficient_mismatch() {
    let mut problem = valid_problem();
    let ValueDef::LinComb(lincomb) = &mut problem.reference.values[1] else {
        unreachable!()
    };
    lincomb.coeff.clear();

    assert_eq!(
        validate(&problem),
        Err(ValidationError::CoefficientCountMismatch {
            value: ValueId(1),
            coefficients: 0,
            args: 1,
        })
    );
}

#[test]
fn accepts_an_empty_linear_combination_as_a_typed_zero() {
    let index = IndexId(0);
    let domain = DomainId(0);
    let problem = Problem {
        sizes: BTreeMap::from([(domain, 10)]),
        symmetries: BTreeMap::new(),
        reference: Program {
            values: vec![ValueDef::LinComb(LinComb {
                coeff: vec![],
                code: Code {
                    inputs: vec![],
                    output: vec![index],
                    domains: BTreeMap::from([(index, domain)]),
                },
                args: vec![],
            })],
            outputs: vec![ValueId(0)],
        },
    };

    assert_eq!(validate(&problem), Ok(()));
}

#[test]
fn rejects_an_invalid_symmetry_sign() {
    let mut problem = valid_problem();
    problem.symmetries.get_mut(&TensorId(0)).unwrap()[0].sign = 0;

    assert_eq!(
        validate(&problem),
        Err(ValidationError::InvalidSymmetrySign {
            tensor: TensorId(0),
            generator: 0,
            sign: 0,
        })
    );
}

#[test]
fn rejects_a_symmetry_rank_mismatch() {
    let mut problem = valid_problem();
    problem.symmetries.get_mut(&TensorId(0)).unwrap()[0].permutation = vec![0];

    assert_eq!(
        validate(&problem),
        Err(ValidationError::SymmetryRankMismatch {
            tensor: TensorId(0),
            generator: 0,
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn rejects_an_invalid_symmetry_permutation() {
    let mut problem = valid_problem();
    problem.symmetries.get_mut(&TensorId(0)).unwrap()[0].permutation = vec![0, 0];

    assert_eq!(
        validate(&problem),
        Err(ValidationError::InvalidSymmetryPermutation {
            tensor: TensorId(0),
            generator: 0,
        })
    );
}

#[test]
fn accepts_a_declared_but_unused_tensor() {
    let mut problem = valid_problem();
    problem.symmetries.insert(TensorId(2), vec![]);

    assert_eq!(validate(&problem), Ok(()));
}

#[test]
fn rejects_an_invalid_output() {
    let mut problem = valid_problem();
    problem.reference.outputs = vec![ValueId(2)];

    assert_eq!(
        validate(&problem),
        Err(ValidationError::InvalidOutput {
            output: 0,
            value: ValueId(2),
        })
    );
}
