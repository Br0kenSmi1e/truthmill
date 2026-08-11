use std::collections::BTreeMap;

use truthmill::{
    Code, Coefficient, DomainId, Einsum, IndexId, Problem, Program, SymmetryGenerator, TensorId,
    ValueDef, ValueId, ValueRef,
};

#[test]
fn represents_minimal_problem_semantics() {
    let antisymmetric = SymmetryGenerator {
        permutation: vec![1, 0],
        sign: -1,
    };
    let problem = Problem {
        sizes: BTreeMap::from([(DomainId(0), 10), (DomainId(1), 5)]),
        symmetries: BTreeMap::from([(TensorId(0), vec![antisymmetric.clone()])]),
        reference: Program {
            values: vec![ValueDef::Einsum(Einsum {
                coeff: Coefficient::from_integer(1.into()),
                code: Code {
                    inputs: vec![vec![IndexId(0), IndexId(1)]],
                    output: vec![IndexId(0), IndexId(1)],
                    domains: BTreeMap::from([(IndexId(0), DomainId(0)), (IndexId(1), DomainId(1))]),
                },
                args: vec![ValueRef::Tensor(TensorId(0))],
            })],
            outputs: vec![ValueId(0)],
        },
    };

    assert_eq!(problem.sizes[&DomainId(1)], 5);
    assert_eq!(problem.symmetries[&TensorId(0)], [antisymmetric]);
}
