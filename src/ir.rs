//! The optimizer-facing tensor SSA representation.

use std::collections::BTreeMap;

use num_rational::BigRational;

/// An exact scalar coefficient.
pub type Coefficient = BigRational;

/// An input tensor declared by the problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub u32);

/// An index domain declared by the problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DomainId(pub u32);

/// An index local to one [`Code`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IndexId(pub u32);

/// A tensor value defined by one SSA operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub u32);

/// A reference to either a problem tensor or a computed SSA value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueRef {
    Tensor(TensorId),
    Value(ValueId),
}

/// A finite, acyclic tensor computation in SSA form.
///
/// The operation at position `n` defines `ValueId(n)` and may reference any
/// problem tensor but only computed values with smaller IDs. Reusing a value ID
/// represents sharing; duplicate value definitions represent recomputation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Program {
    pub values: Vec<ValueDef>,
    pub outputs: Vec<ValueRef>,
}

/// The single definition of one tensor value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueDef {
    Einsum(Einsum),
    LinComb(LinComb),
}

/// The indexing code shared by tensor operations.
///
/// Index IDs have meaning only within this code. `domains` assigns every used
/// index its problem-defined domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Code {
    pub inputs: Vec<Vec<IndexId>>,
    pub output: Vec<IndexId>,
    pub domains: BTreeMap<IndexId, DomainId>,
}

/// An exact scalar multiple of a generalized Einstein contraction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Einsum {
    pub coeff: Coefficient,
    pub code: Code,
    pub args: Vec<ValueRef>,
}

/// An exact indexed linear combination.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinComb {
    pub coeff: Vec<Coefficient>,
    pub code: Code,
    pub args: Vec<ValueRef>,
}

/// The immutable semantics against which programs are checked.
///
/// Domains are nominal in v0: equal sizes do not make two domain IDs equal,
/// and no ambient/subdomain relationship is represented or assumed. Tensor
/// ranks are inferred from the trusted reference program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    pub sizes: BTreeMap<DomainId, u64>,
    pub symmetries: BTreeMap<TensorId, Vec<SymmetryGenerator>>,
    pub reference: Program,
}

/// One generator of a tensor's signed axis-permutation symmetry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymmetryGenerator {
    pub permutation: Vec<usize>,
    pub sign: i8,
}
