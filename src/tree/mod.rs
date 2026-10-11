//! The VDB tree: a shallow, wide B+tree-like hierarchy of nodes.
//!
//! The standard configuration ([`Tree<T>`]) is Root → Internal(32³ children) →
//! Internal(16³ children) → Leaf(8³ voxels), the same as OpenVDB's `Tree4<T, 5, 4, 3>`.

pub mod accessor;
pub mod internal;
pub mod leaf;
pub mod node;
pub mod root;
#[allow(clippy::module_inception)]
pub mod tree;
pub mod value_source;

pub use accessor::{ValueAccessor, ValueAccessorMut};
pub use internal::InternalNode;
pub use leaf::LeafNode;
pub use node::{Lookup, Node};
pub use root::{RootEntry, RootNode};
pub use tree::{
    BoolTree, DoubleTree, FloatTree, Int32Tree, Int64Tree, Internal1, Internal2, Tree, TreeBase, ValueItem, Vec3DTree,
    Vec3ITree, Vec3STree,
};
pub use value_source::ValueSource;
