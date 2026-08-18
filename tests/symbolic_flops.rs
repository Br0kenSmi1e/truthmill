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

fn contraction(domains: [DomainId; 3]) -> Program {
    let i = IndexId(0);
    let j = IndexId(1);
    let k = IndexId(2);
    Program {
        values: vec![ValueDef::Einsum(Einsum {
            coeff: integer(1),
            code: Code {
                inputs: vec![vec![i, j], vec![j, k]],
                output: vec![i, k],
                domains: BTreeMap::from([(i, domains[0]), (j, domains[1]), (k, domains[2])]),
            },
            args: vec![ValueRef::Tensor(TensorId(0)), ValueRef::Tensor(TensorId(1))],
        })],
        outputs: vec![ValueId(0)],
    }
}

#[test]
fn represents_scaling_as_a_sparse_polynomial() {
    let domain = DomainId(0);
    let mut candidate = contraction([domain; 3]);
    let i = IndexId(0);
    let k = IndexId(1);
    candidate.values.push(ValueDef::LinComb(LinComb {
        coeff: vec![integer(1), integer(1)],
        code: Code {
            inputs: vec![vec![i, k], vec![i, k]],
            output: vec![i, k],
            domains: BTreeMap::from([(i, domain), (k, domain)]),
        },
        args: vec![ValueRef::Value(ValueId(0)), ValueRef::Value(ValueId(0))],
    }));
    candidate.outputs = vec![ValueId(1)];
    let problem = problem(candidate.clone(), BTreeMap::from([(domain, 10)]), &[0, 1]);

    let cost = symbolic_flops(&problem, &candidate).unwrap();

    assert_eq!(cost.to_string(), "2 * D0^3 + D0^2");
    assert_eq!(cost.uniform_degree(), Some(3));
    assert_eq!(cost.highest_order_terms(&[]), vec![&cost.terms()[0]]);
    assert_eq!(cost.degree_in(domain), Some(3));
    assert_eq!(cost.terms().len(), 2);
    assert_eq!(cost.terms()[0].coefficient(), 2);
    assert_eq!(cost.terms()[0].powers(), &BTreeMap::from([(domain, 3)]));
    assert_eq!(cost.terms()[1].coefficient(), 1);
    assert_eq!(cost.terms()[1].exponent(domain), 2);
}

#[test]
fn keeps_distinct_domains_symbolic_even_when_their_current_size_is_one() {
    let outer = DomainId(0);
    let reduced = DomainId(1);
    let candidate = contraction([outer, reduced, outer]);
    let problem = problem(
        candidate.clone(),
        BTreeMap::from([(outer, 10), (reduced, 1)]),
        &[0, 1],
    );

    let cost = symbolic_flops(&problem, &candidate).unwrap();

    assert_eq!(cost.to_string(), "2 * D0^2 * D1");
    assert_eq!(cost.uniform_degree(), Some(3));
    assert_eq!(cost.degree_in(outer), Some(2));
    assert_eq!(cost.degree_in(reduced), Some(1));
}

#[test]
fn combines_equal_terms_from_independent_definitions() {
    let domain = DomainId(0);
    let mut candidate = contraction([domain; 3]);
    candidate.values.push(candidate.values[0].clone());
    candidate.outputs = vec![ValueId(0), ValueId(1)];
    let problem = problem(candidate.clone(), BTreeMap::from([(domain, 10)]), &[0, 1]);

    let cost = symbolic_flops(&problem, &candidate).unwrap();

    assert_eq!(cost.to_string(), "4 * D0^3");
    assert_eq!(cost.terms().len(), 1);
    assert_eq!(cost.terms()[0].coefficient(), 4);
}

#[test]
fn returns_every_term_tied_for_highest_order() {
    let first = DomainId(0);
    let second = DomainId(1);
    let mut candidate = contraction([first; 3]);
    let other_contraction = contraction([first, second, first]);
    candidate.values.push(other_contraction.values[0].clone());

    let i = IndexId(0);
    let j = IndexId(1);
    candidate.values.push(ValueDef::LinComb(LinComb {
        coeff: vec![integer(1), integer(1)],
        code: Code {
            inputs: vec![vec![i, j], vec![i, j]],
            output: vec![i, j],
            domains: BTreeMap::from([(i, first), (j, first)]),
        },
        args: vec![ValueRef::Tensor(TensorId(2)), ValueRef::Tensor(TensorId(3))],
    }));
    candidate.outputs = vec![ValueId(0), ValueId(1), ValueId(2)];
    let problem = problem(
        candidate.clone(),
        BTreeMap::from([(first, 10), (second, 5)]),
        &[0, 1, 2, 3],
    );

    let cost = symbolic_flops(&problem, &candidate).unwrap();
    let tied = cost.highest_order_terms(&[]);
    let first_has_priority = cost.highest_order_terms(&[first, second]);
    let second_has_priority = cost.highest_order_terms(&[second, first]);

    assert_eq!(cost.to_string(), "2 * D0^3 + 2 * D0^2 * D1 + D0^2");
    assert_eq!(tied.len(), 2);
    assert!(tied.iter().all(|term| term.total_degree() == 3));
    assert_eq!(first_has_priority, vec![&cost.terms()[0]]);
    assert_eq!(first_has_priority[0].to_string(), "2 * D0^3");
    assert_eq!(second_has_priority, vec![&cost.terms()[1]]);
    assert_eq!(second_has_priority[0].to_string(), "2 * D0^2 * D1");
}

#[test]
fn returns_zero_for_operations_without_modeled_flops() {
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

    let cost = symbolic_flops(&problem, &candidate).unwrap();

    assert!(cost.is_zero());
    assert!(cost.terms().is_empty());
    assert_eq!(cost.uniform_degree(), None);
    assert!(cost.highest_order_terms(&[]).is_empty());
    assert_eq!(cost.degree_in(domain), None);
    assert_eq!(cost.to_string(), "0");
}

#[test]
fn agrees_with_the_independent_numerical_model() {
    let outer = DomainId(0);
    let reduced = DomainId(1);
    let candidate = contraction([outer, reduced, outer]);
    let problem = problem(
        candidate.clone(),
        BTreeMap::from([(outer, 10), (reduced, 5)]),
        &[0, 1],
    );

    let symbolic = symbolic_flops(&problem, &candidate).unwrap();
    let evaluated = symbolic
        .terms()
        .iter()
        .map(|term| {
            term.powers()
                .iter()
                .fold(term.coefficient() as f64, |cost, (domain, exponent)| {
                    cost * (problem.sizes[domain] as f64).powi(*exponent as i32)
                })
        })
        .sum::<f64>();

    assert_close(log_flops(&problem, &candidate).unwrap(), evaluated.ln());
}

#[test]
fn rejects_an_invalid_candidate() {
    let domain = DomainId(0);
    let candidate = contraction([domain; 3]);
    let problem = problem(candidate.clone(), BTreeMap::from([(domain, 0)]), &[0, 1]);

    assert_eq!(
        symbolic_flops(&problem, &candidate),
        Err(ValidationError::ZeroDomainSize { domain })
    );
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "actual {actual}, expected {expected}"
    );
}
