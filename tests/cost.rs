use std::collections::BTreeMap;

use truthmill::*;

fn integer(value: i64) -> Coefficient {
    Coefficient::from_integer(value)
}

fn problem(reference: Program, sizes: BTreeMap<DomainId, u64>, tensors: &[u32]) -> Problem {
    Problem {
        sizes,
        symmetries: tensors
            .iter()
            .map(|&tensor| (TensorId(tensor), Vec::new()))
            .collect(),
        reference,
    }
}

fn contraction() -> Program {
    let i = IndexId(0);
    let j = IndexId(1);
    let k = IndexId(2);
    let domain = DomainId(0);
    Program {
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
    }
}

#[test]
fn follows_gristmill_for_a_contraction() {
    let candidate = contraction();
    let problem = problem(
        candidate.clone(),
        BTreeMap::from([(DomainId(0), 10)]),
        &[0, 1],
    );

    assert_close(log_flops(&problem, &candidate).unwrap(), 2000.0_f64.ln());
}

#[test]
fn counts_an_indexed_linear_combination() {
    let i = IndexId(0);
    let j = IndexId(1);
    let domain = DomainId(0);
    let candidate = Program {
        values: vec![ValueDef::LinComb(LinComb {
            coeff: vec![integer(1), integer(1)],
            code: Code {
                inputs: vec![vec![i, j], vec![i, j]],
                output: vec![i, j],
                domains: BTreeMap::from([(i, domain), (j, domain)]),
            },
            args: vec![ValueRef::Tensor(TensorId(0)), ValueRef::Tensor(TensorId(1))],
        })],
        outputs: vec![ValueId(0)],
    };
    let problem = problem(candidate.clone(), BTreeMap::from([(domain, 10)]), &[0, 1]);

    assert_close(log_flops(&problem, &candidate).unwrap(), 100.0_f64.ln());
}

#[test]
fn counts_a_structural_reduction_for_a_size_one_sum() {
    let i = IndexId(0);
    let j = IndexId(1);
    let k = IndexId(2);
    let output_domain = DomainId(0);
    let unit_domain = DomainId(1);
    let candidate = Program {
        values: vec![ValueDef::Einsum(Einsum {
            coeff: integer(1),
            code: Code {
                inputs: vec![vec![i, j], vec![j, k]],
                output: vec![i, k],
                domains: BTreeMap::from([(i, output_domain), (j, unit_domain), (k, output_domain)]),
            },
            args: vec![ValueRef::Tensor(TensorId(0)), ValueRef::Tensor(TensorId(1))],
        })],
        outputs: vec![ValueId(0)],
    };
    let problem = problem(
        candidate.clone(),
        BTreeMap::from([(output_domain, 10), (unit_domain, 1)]),
        &[0, 1],
    );

    assert_close(log_flops(&problem, &candidate).unwrap(), 200.0_f64.ln());
}

#[test]
fn counts_a_hyperedge_index_once() {
    let i = IndexId(0);
    let domain = DomainId(0);
    let candidate = Program {
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
    let problem = problem(
        candidate.clone(),
        BTreeMap::from([(domain, 10)]),
        &[0, 1, 2],
    );

    assert_close(log_flops(&problem, &candidate).unwrap(), 30.0_f64.ln());
}

#[test]
fn charges_each_ssa_definition_once() {
    let shared = contraction();
    let shared = Program {
        outputs: vec![ValueId(0), ValueId(0)],
        ..shared
    };
    let mut recomputed = shared.clone();
    recomputed.values.push(recomputed.values[0].clone());
    recomputed.outputs = vec![ValueId(0), ValueId(1)];
    let problem = problem(shared.clone(), BTreeMap::from([(DomainId(0), 10)]), &[0, 1]);

    assert_close(log_flops(&problem, &shared).unwrap(), 2000.0_f64.ln());
    assert_close(log_flops(&problem, &recomputed).unwrap(), 4000.0_f64.ln());
}

#[test]
fn reports_zero_modeled_flops_for_copies() {
    let i = IndexId(0);
    let domain = DomainId(0);
    let candidate = Program {
        values: vec![ValueDef::Einsum(Einsum {
            coeff: integer(2),
            code: Code {
                inputs: vec![vec![i]],
                output: vec![i],
                domains: BTreeMap::from([(i, domain)]),
            },
            args: vec![ValueRef::Tensor(TensorId(0))],
        })],
        outputs: vec![ValueId(0)],
    };
    let problem = problem(candidate.clone(), BTreeMap::from([(domain, 10)]), &[0]);

    assert_eq!(
        log_flops(&problem, &candidate),
        Err(CostError::ZeroTotalFlops)
    );
}

#[test]
fn rejects_an_invalid_candidate() {
    let candidate = contraction();
    let problem = problem(
        candidate.clone(),
        BTreeMap::from([(DomainId(0), 0)]),
        &[0, 1],
    );

    assert_eq!(
        log_flops(&problem, &candidate),
        Err(CostError::Invalid(ValidationError::ZeroDomainSize {
            domain: DomainId(0),
        }))
    );
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "actual {actual}, expected {expected}"
    );
}
