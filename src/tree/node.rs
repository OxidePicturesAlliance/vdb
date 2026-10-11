//! The common interface implemented by all non-root tree nodes.

use super::leaf::LeafNode;
use crate::math::{Coord, CoordBBox};
use crate::types::Value;

/// Interface shared by [`LeafNode`] and [`InternalNode`](super::InternalNode).
///
/// Nodes form a fixed-depth hierarchy; the trait's associated constants describe the node's
/// position in that hierarchy (all dimensions are powers of two):
///
/// * `LOG2DIM` — log2 of the number of child slots (or voxels) per axis in this node.
/// * `TOTAL` — log2 of the node's extent in voxels per axis.
/// * `LEVEL` — 0 for leaves, increasing toward the root.
pub trait Node: Clone + Send + Sync + 'static {
    type Value: Value;

    const LEVEL: u32;
    const LOG2DIM: u32;
    const TOTAL: u32;
    /// Extent of the node in voxels per axis.
    const DIM: i32 = 1 << Self::TOTAL;
    /// Number of values (or child slots) in the node.
    const NUM_VALUES: usize = 1 << (3 * Self::LOG2DIM);
    /// Number of voxels spanned by the node.
    const NUM_VOXELS: u64 = 1u64 << (3 * Self::TOTAL);

    /// Create a node whose origin is `xyz` rounded down to a multiple of `DIM`, with every
    /// value set to `value` and every active state set to `active`.
    fn new(xyz: Coord, value: Self::Value, active: bool) -> Self;

    /// The minimum corner of this node.
    fn origin(&self) -> Coord;
    /// Index-space bounds of this node.
    fn node_bbox(&self) -> CoordBBox {
        CoordBBox::create_cube(self.origin(), Self::DIM)
    }

    fn get_value(&self, xyz: &Coord) -> &Self::Value;
    fn is_value_on(&self, xyz: &Coord) -> bool;
    fn probe_value(&self, xyz: &Coord) -> (Self::Value, bool) {
        (*self.get_value(xyz), self.is_value_on(xyz))
    }
    /// The tree level at which the value at `xyz` is stored (0 = leaf voxel).
    fn get_value_level(&self, xyz: &Coord) -> u32;

    fn set_value_on(&mut self, xyz: &Coord, value: Self::Value);
    fn set_value_off(&mut self, xyz: &Coord, value: Self::Value);
    /// Change the value without changing the active state.
    fn set_value_only(&mut self, xyz: &Coord, value: Self::Value);
    fn set_active_state(&mut self, xyz: &Coord, on: bool);
    /// Apply `f` to the value at `xyz` and mark it active.
    fn modify_value<F: FnOnce(&mut Self::Value)>(&mut self, xyz: &Coord, f: F);
    /// Apply `f` to the value and active state at `xyz`.
    fn modify_value_and_active_state<F: FnOnce(&mut Self::Value, &mut bool)>(&mut self, xyz: &Coord, f: F);

    /// Set all voxels in `bbox` (clipped to this node) to `value` with the given state,
    /// using tiles wherever possible.
    fn fill(&mut self, bbox: &CoordBBox, value: Self::Value, active: bool);

    /// Mark every value in `bbox` (clipped to this node) active, keeping values unchanged.
    fn activate_region(&mut self, bbox: &CoordBBox);

    /// Return the leaf containing `xyz`, creating it (and any intermediate nodes) if needed.
    fn touch_leaf(&mut self, xyz: &Coord) -> &mut LeafNode<Self::Value>;
    fn probe_leaf(&self, xyz: &Coord) -> Option<&LeafNode<Self::Value>>;
    fn probe_leaf_mut(&mut self, xyz: &Coord) -> Option<&mut LeafNode<Self::Value>>;
    /// Insert a leaf, replacing any existing leaf or tile at its position.
    fn add_leaf(&mut self, leaf: Box<LeafNode<Self::Value>>);
    /// Insert a tile at the given level (`0` sets a single voxel), deleting any child there.
    fn add_tile(&mut self, level: u32, xyz: &Coord, value: Self::Value, active: bool);

    /// If every value is within `tolerance` of the first one, all active states agree, and
    /// the node has no children, return that first value and the shared state.
    fn is_constant(&self, tolerance: &Self::Value) -> Option<(Self::Value, bool)>;
    /// Replace descendant nodes that are constant (within `tolerance`) with tiles.
    fn prune(&mut self, tolerance: &Self::Value);
    /// Replace descendant nodes that have no active values with inactive tiles of `background`.
    fn prune_inactive(&mut self, background: &Self::Value);

    /// Expand `bbox` to include all active values; if `visit_voxels` is false, leaves count
    /// as wholly active.
    fn eval_active_bbox(&self, bbox: &mut CoordBBox, visit_voxels: bool);

    fn active_voxel_count(&self) -> u64;
    fn inactive_voxel_count(&self) -> u64;
    fn active_leaf_voxel_count(&self) -> u64;
    fn inactive_leaf_voxel_count(&self) -> u64;
    fn active_tile_count(&self) -> u64;
    fn leaf_count(&self) -> usize;
    /// Accumulate the number of nodes at each level into `counts` (indexed by level).
    fn node_count(&self, counts: &mut [usize]);
    /// True if the node contains no active values.
    fn is_inactive(&self) -> bool;

    /// Iterate over all leaves below (or equal to) this node, in index order.
    fn leaves(&self) -> Box<dyn Iterator<Item = &LeafNode<Self::Value>> + '_>;
    fn leaves_mut(&mut self) -> Box<dyn Iterator<Item = &mut LeafNode<Self::Value>> + '_>;

    /// Visit every tile (non-leaf value) below this node: `f(tile bbox, value, active, level)`.
    fn visit_tiles(&self, f: &mut dyn FnMut(CoordBBox, &Self::Value, bool, u32));
    /// Visit every tile mutably: `f(tile bbox, &mut value, &mut active, level)`.
    fn visit_tiles_mut(&mut self, f: &mut dyn FnMut(CoordBBox, &mut Self::Value, &mut bool, u32));

    /// Set all voxels outside `bbox` to inactive `background`.
    fn clip(&mut self, bbox: &CoordBBox, background: &Self::Value);
    /// Replace active tiles with leaves of active voxels.
    fn voxelize_active_tiles(&mut self);
    /// Mark every value active or inactive.
    fn set_values_on_off(&mut self, on: bool);
    /// Replace inactive values equal to `old` with `new` (used when changing the background).
    fn reset_background(&mut self, old: &Self::Value, new: &Self::Value);

    /// Approximate memory used by this node and its descendants, in bytes.
    fn mem_usage(&self) -> usize;
}
