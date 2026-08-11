use std::collections::BTreeMap;

use truthmill::*;

fn one() -> Coefficient {
    Coefficient::from_integer(1.into())
}

#[test]
fn represents_explicit_ssa_sharing() {
    let i = IndexId(0);
    let j = IndexId(1);
    let k = IndexId(2);
    let domain = DomainId(0);

    let program = Program {
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
                coeff: vec![one(), one()],
                code: Code {
                    inputs: vec![vec![i, k], vec![i, k]],
                    output: vec![i, k],
                    domains: BTreeMap::from([(i, domain), (k, domain)]),
                },
                args: vec![ValueRef::Value(ValueId(0)), ValueRef::Value(ValueId(0))],
            }),
        ],
        outputs: vec![ValueId(1)],
    };

    let ValueDef::LinComb(lincomb) = &program.values[1] else {
        panic!("the second value is a linear combination")
    };
    assert_eq!(
        lincomb.args,
        [ValueRef::Value(ValueId(0)), ValueRef::Value(ValueId(0))]
    );
}
