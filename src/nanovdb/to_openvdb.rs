//! Conversion of NanoVDB grids back into the crate's [`Grid`] (port of
//! `nanovdb/tools/NanoToOpenVDB.h`).

use super::build_types::{BuildType, Fp16, Fp4, Fp8, FpN, IndexBuildType, NanoPod, ValueMask};
use super::grid::NanoGrid;
use super::tree::{NanoLeaf, NanoLower, NanoUpper};
use super::GridClass;
use crate::grid::{Grid, GridClass as VdbGridClass};
use crate::math::{Coord, Transform, Vec3d, Vec3f};
use crate::tree::{Internal1, Internal2, LeafNode};
use crate::types::Value;

/// Build types that can be converted back to a [`Grid`]: `Out` is the value type of the
/// resulting grid (floats for the quantized types, `bool` for masks).
pub trait ToOpenVdb: BuildType {
    /// Value type of the output grid.
    type Out: Value;
    /// Convert a tile or background value.
    fn to_out(v: Self::Value) -> Self::Out;
}

macro_rules! to_open_same {
    ($($t:ty),*) => {$(
        impl ToOpenVdb for $t {
            type Out = $t;
            #[inline]
            fn to_out(v: $t) -> $t { v }
        }
    )*};
}
to_open_same!(f32, f64, i32, i64, u32, Vec3f, Vec3d, bool);

macro_rules! to_open_float {
    ($($t:ty),*) => {$(
        impl ToOpenVdb for $t {
            type Out = f32;
            #[inline]
            fn to_out(v: f32) -> f32 { v }
        }
    )*};
}
to_open_float!(Fp4, Fp8, Fp16, FpN);

impl ToOpenVdb for ValueMask {
    type Out = bool;
    fn to_out(v: bool) -> bool {
        v
    }
}

fn to_vdb_class(c: GridClass) -> VdbGridClass {
    match c {
        GridClass::LevelSet => VdbGridClass::LevelSet,
        GridClass::FogVolume => VdbGridClass::FogVolume,
        GridClass::Staggered => VdbGridClass::Staggered,
        _ => VdbGridClass::Unknown,
    }
}

/// Generic conversion: `tile(value_loc_value)` maps stored tile values and `leaf` produces
/// the 512 voxel values of a leaf.
fn convert<B: BuildType, T: Value>(
    grid: &NanoGrid<'_, B>,
    background: T,
    tile: &dyn Fn(B::Value) -> T,
    leaf: &dyn Fn(&NanoLeaf<'_, B>) -> Vec<T>,
) -> Grid<T> {
    let mut out = Grid::<T>::new(background);
    out.set_name(&grid.grid_name());
    out.set_grid_class(to_vdb_class(grid.grid_class()));
    out.set_transform(Transform::create_linear_transform_from_matrix(grid.map().to_mat4()));
    let root = grid.tree().root();
    let tree = out.tree_mut();
    for t in root.tiles() {
        match t.child() {
            Some(upper) => {
                let node = convert_upper(&upper, t.origin(), tile, leaf);
                tree.root_mut().add_child(Box::new(node));
            }
            None => tree.root_mut().add_root_tile(&t.origin(), tile(t.value()), t.state() != 0),
        }
    }
    out
}

fn convert_upper<B: BuildType, T: Value>(
    src: &NanoUpper<'_, B>,
    origin: Coord,
    tile: &dyn Fn(B::Value) -> T,
    leaf: &dyn Fn(&NanoLeaf<'_, B>) -> Vec<T>,
) -> Internal2<T> {
    let size = NanoUpper::<B>::SIZE;
    let mut values = vec![T::zero(); size];
    let mut children: Vec<Option<Box<Internal1<T>>>> = (0..size).map(|_| None).collect();
    for n in 0..size {
        match src.child(n) {
            Some(lower) => {
                let o = NanoUpper::<B>::child_origin(origin, n);
                children[n] = Some(Box::new(convert_lower(&lower, o, tile, leaf)));
            }
            None => values[n] = tile(src.tile_value(n)),
        }
    }
    Internal2::from_parts(origin, src.child_mask().to_node_mask(), src.value_mask().to_node_mask(), values, children)
}

fn convert_lower<B: BuildType, T: Value>(
    src: &NanoLower<'_, B>,
    origin: Coord,
    tile: &dyn Fn(B::Value) -> T,
    leaf: &dyn Fn(&NanoLeaf<'_, B>) -> Vec<T>,
) -> Internal1<T> {
    let size = NanoLower::<B>::SIZE;
    let mut values = vec![T::zero(); size];
    let mut children: Vec<Option<Box<LeafNode<T>>>> = (0..size).map(|_| None).collect();
    for n in 0..size {
        match src.child(n) {
            Some(l) => {
                let o = NanoLower::<B>::child_origin(origin, n);
                children[n] = Some(Box::new(LeafNode::from_parts(o, l.value_mask().to_node_mask(), leaf(&l))));
            }
            None => values[n] = tile(src.tile_value(n)),
        }
    }
    Internal1::from_parts(origin, src.child_mask().to_node_mask(), src.value_mask().to_node_mask(), values, children)
}

/// Convert a NanoVDB grid into a [`Grid`] (`nanovdb::tools::nanoToOpenVDB`): the tree
/// topology, values and active states, the name, the grid class (level set, fog volume and
/// staggered only) and the transform are preserved; quantized values are decoded.
pub fn nano_to_openvdb<B: ToOpenVdb>(grid: &NanoGrid<'_, B>) -> Grid<B::Out> {
    let bg = B::to_out(grid.tree().background());
    convert(grid, bg, &|v| B::to_out(v), &|l| (0..512).map(|n| B::to_out(l.get_value_at(n))).collect())
}

/// Convert an index grid and a sidecar array of values (e.g. one of its channels) into a
/// [`Grid`] whose voxel and tile values are `values[index]`.
pub fn index_to_openvdb<B: IndexBuildType, T: Value + NanoPod>(
    grid: &NanoGrid<'_, B>,
    values: &[T],
    class: GridClass,
    name: Option<&str>,
) -> Grid<T> {
    let lookup = |i: u64| values.get(i as usize).copied().unwrap_or(values[0]);
    let mut g = convert(grid, values[0], &|i| lookup(i), &|l| (0..512).map(|n| lookup(l.get_value_at(n))).collect());
    g.set_grid_class(to_vdb_class(class));
    g.set_name(name.unwrap_or(""));
    g
}
