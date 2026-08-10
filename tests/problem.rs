use std::collections::BTreeMap;

use truthmill::ir::{DomainId, Problem, Program, SymmetryGenerator, TensorId};

#[test]
fn represents_minimal_problem_semantics() {
    let antisymmetric = SymmetryGenerator {
        permutation: vec![1, 0],
        sign: -1,
    };
    let problem = Problem {
        sizes: BTreeMap::from([(DomainId(0), 10), (DomainId(1), 5)]),
        symmetries: BTreeMap::from([(TensorId(0), vec![antisymmetric.clone()])]),
        reference: Program::default(),
    };

    assert_eq!(problem.sizes[&DomainId(1)], 5);
    assert_eq!(problem.symmetries[&TensorId(0)], [antisymmetric]);
}
