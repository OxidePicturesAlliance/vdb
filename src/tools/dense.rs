//! Dense voxel arrays and conversion to and from sparse trees (port of
//! `openvdb/tools/Dense.h`).

use crate::math::{Coord, CoordBBox};
use crate::tools::prune::prune_tiles;
use crate::tools::value_traits::{ToolError, ToolResult, ToolValue, TreeAdapter};
use crate::tree::{LeafNode, Node};
use crate::types::Value;
use rayon::prelude::*;

/// Memory layout of a [`Dense`] array (OpenVDB's `MemoryLayout`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MemoryLayout {
    /// x varies fastest (`LayoutXYZ`, as in most image formats).
    LayoutXYZ,
    /// z varies fastest (`LayoutZYX`, the same ordering as VDB leaf nodes).
    #[default]
    LayoutZYX,
}

/// A dense 3D array of values covering an index-space bounding box (`tools::Dense`).
#[derive(Clone, Debug, PartialEq)]
pub struct Dense<T> {
    bbox: CoordBBox,
    layout: MemoryLayout,
    /// Strides (x, y, z) in elements.
    strides: [usize; 3],
    data: Vec<T>,
}

impl<T: Value> Dense<T> {
    /// A dense array over `bbox` filled with the default value. Returns an error if the
    /// box is empty.
    pub fn new(bbox: CoordBBox, layout: MemoryLayout) -> ToolResult<Self> {
        Self::with_value(bbox, T::zero(), layout)
    }
    /// A dense array over `bbox` filled with `value`.
    pub fn with_value(bbox: CoordBBox, value: T, layout: MemoryLayout) -> ToolResult<Self> {
        if bbox.is_empty() {
            return Err(ToolError::Value("can't construct a dense grid with an empty bounding box".into()));
        }
        let d = bbox.dim();
        let strides = match layout {
            MemoryLayout::LayoutZYX => {
                let my = d[2] as usize;
                [my * d[1] as usize, my, 1]
            }
            MemoryLayout::LayoutXYZ => {
                let my = d[0] as usize;
                [1, my, my * d[1] as usize]
            }
        };
        Ok(Dense { bbox, layout, strides, data: vec![value; bbox.volume() as usize] })
    }
    /// A dense array with dimensions `dim` whose minimum corner is `min`.
    pub fn from_dim(dim: Coord, min: Coord, layout: MemoryLayout) -> ToolResult<Self> {
        Self::new(CoordBBox::new(min, min + dim.offset(-1)), layout)
    }
    pub fn memory_layout(&self) -> MemoryLayout {
        self.layout
    }
    pub fn bbox(&self) -> &CoordBBox {
        &self.bbox
    }
    pub fn origin(&self) -> Coord {
        self.bbox.min
    }
    pub fn value_count(&self) -> u64 {
        self.bbox.volume()
    }
    pub fn data(&self) -> &[T] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
    pub fn x_stride(&self) -> usize {
        self.strides[0]
    }
    pub fn y_stride(&self) -> usize {
        self.strides[1]
    }
    pub fn z_stride(&self) -> usize {
        self.strides[2]
    }
    /// Linear offset of the local coordinates `(i, j, k)` (relative to the minimum corner).
    #[inline]
    pub fn local_to_offset(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.strides[0] + j * self.strides[1] + k * self.strides[2]
    }
    /// Linear offset of the global coordinates `xyz`, which must lie inside the box.
    #[inline]
    pub fn coord_to_offset(&self, xyz: &Coord) -> usize {
        debug_assert!(self.bbox.is_inside(xyz));
        let m = self.bbox.min;
        self.local_to_offset((xyz[0] - m[0]) as usize, (xyz[1] - m[1]) as usize, (xyz[2] - m[2]) as usize)
    }
    /// Local coordinates of the linear offset `n`.
    pub fn offset_to_local_coord(&self, n: usize) -> Coord {
        match self.layout {
            MemoryLayout::LayoutZYX => {
                let x = n / self.strides[0];
                let r = n - x * self.strides[0];
                let y = r / self.strides[1];
                Coord::new(x as i32, y as i32, (r - y * self.strides[1]) as i32)
            }
            MemoryLayout::LayoutXYZ => {
                let z = n / self.strides[2];
                let r = n - z * self.strides[2];
                let y = r / self.strides[1];
                Coord::new((r - y * self.strides[1]) as i32, y as i32, z as i32)
            }
        }
    }
    /// Global coordinates of the linear offset `n`.
    pub fn offset_to_coord(&self, n: usize) -> Coord {
        self.offset_to_local_coord(n) + self.bbox.min
    }
    pub fn get_value(&self, xyz: &Coord) -> T {
        self.data[self.coord_to_offset(xyz)]
    }
    pub fn set_value(&mut self, xyz: &Coord, value: T) {
        let n = self.coord_to_offset(xyz);
        self.data[n] = value;
    }
    pub fn get_value_at(&self, n: usize) -> T {
        self.data[n]
    }
    pub fn set_value_at(&mut self, n: usize, value: T) {
        self.data[n] = value;
    }
    pub fn fill(&mut self, value: T) {
        self.data.iter_mut().for_each(|v| *v = value);
    }
    /// Memory footprint in bytes.
    pub fn mem_usage(&self) -> usize {
        std::mem::size_of::<Self>() + self.data.len() * std::mem::size_of::<T>()
    }
}

/// Copy the values (active or not) of a sparse grid or tree into `dense`, over the dense
/// array's bounding box (`tools::copyToDense`).
pub fn copy_to_dense<T: Value, G: TreeAdapter<T>>(sparse: &G, dense: &mut Dense<T>, serial: bool) {
    let tree = sparse.tree_ref();
    let bbox = dense.bbox;
    let min = bbox.min;
    let (sx, sy, sz) = (dense.strides[0], dense.strides[1], dense.strides[2]);
    let nx = bbox.dim()[0] as usize;
    // Split the work into x slabs; each slab owns a disjoint set of offsets only for the
    // ZYX layout, so compute values per slab and scatter them afterwards.
    let slab = |x: usize| -> Vec<T> {
        let acc = tree.accessor();
        let mut out = Vec::with_capacity((bbox.dim()[1] * bbox.dim()[2]) as usize);
        for y in bbox.min[1]..=bbox.max[1] {
            for z in bbox.min[2]..=bbox.max[2] {
                out.push(acc.get_value(&Coord::new(min[0] + x as i32, y, z)));
            }
        }
        out
    };
    let slabs: Vec<Vec<T>> =
        if serial { (0..nx).map(slab).collect() } else { (0..nx).into_par_iter().map(slab).collect() };
    let ny = bbox.dim()[1] as usize;
    let nz = bbox.dim()[2] as usize;
    for (x, s) in slabs.into_iter().enumerate() {
        for y in 0..ny {
            for z in 0..nz {
                dense.data[x * sx + y * sy + z * sz] = s[y * nz + z];
            }
        }
    }
}

/// Copy the values of `dense` into a sparse grid or tree (`tools::copyFromDense`).
///
/// Values within `tolerance` of the background become inactive background values; all
/// other values become active. Existing leaf nodes outside the dense region are kept, leaf
/// blocks that end up constant become tiles, and internal nodes made only of tiles are
/// pruned (`pruneTiles`).
pub fn copy_from_dense<T: ToolValue, G: TreeAdapter<T>>(dense: &Dense<T>, sparse: &mut G, tolerance: T, serial: bool) {
    let tree = sparse.tree_mut_ref();
    let bbox = dense.bbox;
    let background = *tree.background();
    let empty_target = tree.is_empty();

    // Blocks aligned with (potential) leaf nodes.
    let mut blocks = Vec::new();
    let mut x = bbox.min[0];
    while x <= bbox.max[0] {
        let xe = bbox.max[0].min((x & !7) + 7);
        let mut y = bbox.min[1];
        while y <= bbox.max[1] {
            let ye = bbox.max[1].min((y & !7) + 7);
            let mut z = bbox.min[2];
            while z <= bbox.max[2] {
                let ze = bbox.max[2].min((z & !7) + 7);
                blocks.push(CoordBBox::from_coords(x, y, z, xe, ye, ze));
                z = ze + 1;
            }
            y = ye + 1;
        }
        x = xe + 1;
    }

    enum BlockResult<T: Value> {
        Leaf(Box<LeafNode<T>>),
        Tile(T, bool),
    }
    let process = |b: &CoordBBox| -> BlockResult<T> {
        let t = &*tree;
        let mut leaf = if empty_target {
            LeafNode::new(b.min, background, false)
        } else if let Some(target) = t.probe_leaf(&b.min) {
            target.clone()
        } else {
            let (v, on) = t.probe_value(&b.min);
            LeafNode::new(b.min, v, on)
        };
        for xyz in *b {
            let v = dense.get_value(&xyz);
            let n = LeafNode::<T>::coord_to_offset(&xyz);
            if background.approx_eq(&v, &tolerance) {
                leaf.set_value_off_at(n, background);
            } else {
                leaf.set_value_on_at(n, v);
            }
        }
        match leaf.value_mask().is_constant() {
            Some(state) => {
                let first = *leaf.get_value_at(0);
                if leaf.buffer().iter().all(|v| v.approx_eq(&first, &tolerance)) {
                    BlockResult::Tile(first, state)
                } else {
                    BlockResult::Leaf(Box::new(leaf))
                }
            }
            None => BlockResult::Leaf(Box::new(leaf)),
        }
    };
    let results: Vec<BlockResult<T>> =
        if serial { blocks.iter().map(process).collect() } else { blocks.par_iter().map(process).collect() };
    for (b, r) in blocks.iter().zip(results) {
        match r {
            BlockResult::Leaf(leaf) => tree.add_leaf(leaf),
            BlockResult::Tile(v, true) => tree.add_tile(1, &b.min, v, true),
            BlockResult::Tile(_, false) => {}
        }
    }
    prune_tiles(tree, tolerance, !serial);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::FloatTree;

    #[test]
    fn dense_roundtrip() {
        for layout in [MemoryLayout::LayoutZYX, MemoryLayout::LayoutXYZ] {
            let bbox = CoordBBox::from_coords(-3, -2, -1, 12, 9, 20);
            let mut dense = Dense::<f32>::new(bbox, layout).unwrap();
            for n in 0..dense.value_count() as usize {
                let c = dense.offset_to_coord(n);
                assert_eq!(dense.coord_to_offset(&c), n);
                let v = if (c[0] + c[1] + c[2]) % 3 == 0 { 0.0 } else { (c[0] * 100 + c[1] * 10 + c[2]) as f32 };
                dense.set_value_at(n, v);
            }
            let mut tree = FloatTree::new(0.0);
            copy_from_dense(&dense, &mut tree, 0.0, false);
            let mut back = Dense::<f32>::new(bbox, layout).unwrap();
            copy_to_dense(&tree, &mut back, false);
            assert_eq!(dense, back);
            let nonzero = dense.data().iter().filter(|v| **v != 0.0).count() as u64;
            assert_eq!(tree.active_voxel_count(), nonzero);
        }
        // A constant active block becomes a tile.
        let dense =
            Dense::with_value(CoordBBox::from_coords(0, 0, 0, 15, 15, 7), 2.0f32, MemoryLayout::LayoutZYX).unwrap();
        let mut tree = FloatTree::new(0.0);
        copy_from_dense(&dense, &mut tree, 0.0, true);
        assert_eq!(tree.leaf_count(), 0);
        assert_eq!(tree.active_voxel_count(), 16 * 16 * 8);
        assert!(Dense::<f32>::new(CoordBBox::empty(), MemoryLayout::LayoutXYZ).is_err());
    }
}
