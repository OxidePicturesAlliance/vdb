//! Tree topology and buffer serialization (port of `readTopology`/`writeTopology`/
//! `readBuffers`/`writeBuffers` in `Tree.h`, `RootNode.h`, `InternalNode.h`, `LeafNode.h`,
//! `LeafNodeBool.h` and `LeafNodeMask.h`).
//!
//! A tree is written in two passes: the *topology* (root tiles, internal node masks and tile
//! values, leaf value masks) and then the *buffers* (leaf voxel values), both in
//! depth-first order with children visited in increasing slot order.

#[cfg(feature = "read")]
use super::bytes::Reader;
#[cfg(feature = "write")]
use super::bytes::WriteLe;
#[cfg(feature = "read")]
use super::compression::{
    read_compressed_values, read_compressed_values_into, read_mask, FILE_VERSION_NODE_MASK_COMPRESSION,
};
#[cfg(feature = "write")]
use super::compression::{write_compressed_values, write_mask};
use super::compression::{Compression, ValueCodec};
#[cfg(feature = "read")]
use super::error::{Error, Result};
use super::value_io::IoValue;
use crate::math::Coord;
#[cfg(feature = "read")]
use crate::tree::RootNode;
#[cfg(feature = "write")]
use crate::tree::Tree;
use crate::tree::{InternalNode, LeafNode, Node, RootEntry};
use crate::util::NodeMask;
#[cfg(feature = "read")]
use std::io::Read;

/// File format version from which internal nodes store a compressed value array.
#[cfg(feature = "read")]
const FILE_VERSION_INTERNALNODE_COMPRESSION: u32 = 214;

/// How leaf voxel values are encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LeafFormat {
    /// `LeafNode<T>`: a value mask and a (compressed) array of values.
    Standard,
    /// `LeafNode<bool>`: a value mask, the origin and a bit mask of values.
    Bool,
    /// `LeafNode<ValueMask>`: the value mask is also the value buffer.
    Mask,
}

/// Settings for reading or writing one tree.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TreeCtx<T> {
    pub codec: ValueCodec<T>,
    pub leaf_format: LeafFormat,
}

/// Serialization of a node and its descendants.
pub(crate) trait NodeIo<T: IoValue>: Node<Value = T> + Sized {
    #[cfg(feature = "read")]
    fn read_topology<R: Read>(r: &mut Reader<R>, origin: Coord, ctx: &TreeCtx<T>) -> Result<Self>;
    #[cfg(feature = "write")]
    #[cfg(feature = "write")]
    fn write_topology(&self, out: &mut Vec<u8>, ctx: &TreeCtx<T>);
    #[cfg(feature = "read")]
    #[cfg(feature = "read")]
    fn read_buffers<R: Read>(&mut self, r: &mut Reader<R>, ctx: &TreeCtx<T>) -> Result<()>;
    #[cfg(feature = "write")]
    #[cfg(feature = "write")]
    fn write_buffers(&self, out: &mut Vec<u8>, ctx: &TreeCtx<T>);
}

#[cfg(feature = "read")]
fn read_coord<R: Read>(r: &mut Reader<R>) -> Result<Coord> {
    Ok(Coord::new(r.read_i32()?, r.read_i32()?, r.read_i32()?))
}

#[cfg(feature = "write")]
fn write_coord(out: &mut Vec<u8>, c: Coord) {
    for i in 0..3 {
        out.put_i32(c[i]);
    }
}

impl<T: IoValue> NodeIo<T> for LeafNode<T> {
    #[cfg(feature = "read")]
    fn read_topology<R: Read>(r: &mut Reader<R>, origin: Coord, ctx: &TreeCtx<T>) -> Result<Self> {
        let mut mask = NodeMask::<3>::new();
        read_mask(r, &mut mask)?;
        let buffer = match ctx.leaf_format {
            LeafFormat::Mask => (0..512).map(|i| T::from_bit(mask.is_on(i))).collect(),
            _ => vec![ctx.codec.background; 512],
        };
        Ok(LeafNode::from_parts(origin, mask, buffer))
    }

    #[cfg(feature = "write")]
    fn write_topology(&self, out: &mut Vec<u8>, _ctx: &TreeCtx<T>) {
        write_mask(out, self.value_mask());
    }

    #[cfg(feature = "read")]
    fn read_buffers<R: Read>(&mut self, r: &mut Reader<R>, ctx: &TreeCtx<T>) -> Result<()> {
        match ctx.leaf_format {
            LeafFormat::Standard => {
                read_mask(r, self.value_mask_mut())?;
                let mut num_buffers = 1i8;
                if ctx.codec.file_version < FILE_VERSION_NODE_MASK_COMPRESSION {
                    let origin = read_coord(r)?;
                    self.set_origin(origin);
                    num_buffers = r.read_i8()?;
                }
                let (buffer, mask) = self.buffer_and_mask_mut();
                read_compressed_values_into(r, buffer, mask, &ctx.codec)?;
                // Auxiliary buffers written by old library versions are read and discarded.
                // They are never mask compressed.
                let zipped = ctx.codec.compression.contains(Compression::ZIP);
                let aux = ValueCodec {
                    compression: if zipped { Compression::ZIP } else { Compression::NONE },
                    file_version: 0,
                    ..ctx.codec
                };
                for _ in 1..num_buffers.max(1) {
                    read_compressed_values(r, 512, self.value_mask(), &aux)?;
                }
            }
            LeafFormat::Bool => {
                read_mask(r, self.value_mask_mut())?;
                let origin = read_coord(r)?;
                self.set_origin(origin);
                let mut data = NodeMask::<3>::new();
                read_mask(r, &mut data)?;
                for (i, v) in self.buffer_mut().iter_mut().enumerate() {
                    *v = T::from_bit(data.is_on(i));
                }
            }
            LeafFormat::Mask => {
                read_mask(r, self.value_mask_mut())?;
                let origin = read_coord(r)?;
                self.set_origin(origin);
                let (buffer, mask) = self.buffer_and_mask_mut();
                for (i, v) in buffer.iter_mut().enumerate() {
                    *v = T::from_bit(mask.is_on(i));
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "write")]
    fn write_buffers(&self, out: &mut Vec<u8>, ctx: &TreeCtx<T>) {
        write_mask(out, self.value_mask());
        match ctx.leaf_format {
            LeafFormat::Standard => {
                write_compressed_values(out, self.buffer(), self.value_mask(), &NodeMask::<3>::new(), &ctx.codec);
            }
            LeafFormat::Bool => {
                write_coord(out, self.origin());
                let mut data = NodeMask::<3>::new();
                for (i, v) in self.buffer().iter().enumerate() {
                    if v.to_bit() {
                        data.set_on(i);
                    }
                }
                write_mask(out, &data);
            }
            LeafFormat::Mask => write_coord(out, self.origin()),
        }
    }
}

impl<T: IoValue, C: NodeIo<T>, const L: u32> NodeIo<T> for InternalNode<C, L> {
    #[cfg(feature = "read")]
    fn read_topology<R: Read>(r: &mut Reader<R>, origin: Coord, ctx: &TreeCtx<T>) -> Result<Self> {
        let version = ctx.codec.file_version;
        if version < FILE_VERSION_INTERNALNODE_COMPRESSION {
            return Err(Error::UnsupportedVersion(version));
        }
        let size = NodeMask::<L>::SIZE;
        let mut child_mask = NodeMask::<L>::new();
        let mut value_mask = NodeMask::<L>::new();
        read_mask(r, &mut child_mask)?;
        read_mask(r, &mut value_mask)?;
        let old = version < FILE_VERSION_NODE_MASK_COMPRESSION;
        let num_values = if old { child_mask.count_off() as usize } else { size };
        let read = read_compressed_values(r, num_values, &value_mask, &ctx.codec)?;
        let values = if old {
            let mut it = read.into_iter();
            (0..size)
                .map(|i| if child_mask.is_off(i) { it.next().unwrap_or(ctx.codec.background) } else { T::zero() })
                .collect()
        } else {
            read
        };
        let origin = origin.align_down(Self::DIM);
        let mut children: Vec<Option<Box<C>>> = (0..size).map(|_| None).collect();
        for n in child_mask.iter_on() {
            let l = Self::offset_to_local_coord(n);
            let child_origin = origin + Coord::new(l[0] << C::TOTAL, l[1] << C::TOTAL, l[2] << C::TOTAL);
            children[n] = Some(Box::new(C::read_topology(r, child_origin, ctx)?));
        }
        Ok(InternalNode::from_parts(origin, child_mask, value_mask, values, children))
    }

    #[cfg(feature = "write")]
    fn write_topology(&self, out: &mut Vec<u8>, ctx: &TreeCtx<T>) {
        write_mask(out, self.child_mask());
        write_mask(out, self.value_mask());
        let values: Vec<T> =
            self.values().iter().enumerate().map(|(i, v)| if self.is_child(i) { T::zero() } else { *v }).collect();
        write_compressed_values(out, &values, self.value_mask(), self.child_mask(), &ctx.codec);
        for (_, child) in self.children() {
            child.write_topology(out, ctx);
        }
    }

    #[cfg(feature = "read")]
    fn read_buffers<R: Read>(&mut self, r: &mut Reader<R>, ctx: &TreeCtx<T>) -> Result<()> {
        for child in self.children_mut() {
            child.read_buffers(r, ctx)?;
        }
        Ok(())
    }

    #[cfg(feature = "write")]
    fn write_buffers(&self, out: &mut Vec<u8>, ctx: &TreeCtx<T>) {
        for (_, child) in self.children() {
            child.write_buffers(out, ctx);
        }
    }
}

/// Read a tree's topology and buffers.
#[cfg(feature = "read")]
pub(crate) fn read_tree<T, C, R>(
    r: &mut Reader<R>,
    file_version: u32,
    compression: Compression,
    half: bool,
    leaf_format: LeafFormat,
) -> Result<crate::tree::TreeBase<C>>
where
    T: IoValue,
    C: NodeIo<T>,
    R: Read,
{
    let (mut tree, ctx) = read_tree_topology::<T, C, R>(r, file_version, compression, half, leaf_format)?;
    for child in tree.root_mut().children_mut() {
        child.read_buffers(r, &ctx)?;
    }
    Ok(tree)
}

/// Read a tree's topology only (buffer count, root, internal nodes and leaf masks); also
/// returns the context for reading its buffers.
#[cfg(feature = "read")]
pub(crate) fn read_tree_topology<T, C, R>(
    r: &mut Reader<R>,
    file_version: u32,
    compression: Compression,
    half: bool,
    leaf_format: LeafFormat,
) -> Result<(crate::tree::TreeBase<C>, TreeCtx<T>)>
where
    T: IoValue,
    C: NodeIo<T>,
    R: Read,
{
    let buffer_count = r.read_i32()?;
    if buffer_count != 1 {
        // Multi-buffer trees are no longer supported by OpenVDB either; only the first is used.
    }
    let background = T::from_le(&r.read_bytes(T::SIZE)?);
    let ctx = TreeCtx { codec: ValueCodec { file_version, compression, half, background }, leaf_format };
    let mut root = RootNode::<C>::new(background);
    let num_tiles = r.read_u32()?;
    let num_children = r.read_u32()?;
    for _ in 0..num_tiles {
        let key = read_coord(r)?;
        let value = T::from_le(&r.read_bytes(T::SIZE)?);
        let active = r.read_bool()?;
        root.table_mut().insert(key, RootEntry::Tile { value, active });
    }
    for _ in 0..num_children {
        let key = read_coord(r)?;
        let child = C::read_topology(r, key, &ctx)?;
        root.table_mut().insert(key, RootEntry::Child(Box::new(child)));
    }
    Ok((crate::tree::TreeBase::from_root(root), ctx))
}

/// Write a tree's topology (buffer count, root, internal nodes and leaf masks).
#[cfg(feature = "write")]
pub(crate) fn write_tree_topology<T: IoValue, C: NodeIo<T>>(
    out: &mut Vec<u8>,
    tree: &crate::tree::TreeBase<C>,
    ctx: &TreeCtx<T>,
) {
    out.put_i32(1); // buffer count
    let root = tree.root();
    let bg = *root.background();
    let bg = if ctx.codec.half { bg.truncate_half() } else { bg };
    bg.write_le(out);
    out.put_u32(root.tile_count() as u32);
    out.put_u32(root.child_count() as u32);
    for (key, entry) in root.table() {
        if let RootEntry::Tile { value, active } = entry {
            write_coord(out, *key);
            value.write_le(out);
            out.put_u8(*active as u8);
        }
    }
    for (key, entry) in root.table() {
        if let RootEntry::Child(child) = entry {
            write_coord(out, *key);
            child.write_topology(out, ctx);
        }
    }
}

/// Write a tree's leaf buffers.
#[cfg(feature = "write")]
pub(crate) fn write_tree_buffers<T: IoValue, C: NodeIo<T>>(
    out: &mut Vec<u8>,
    tree: &crate::tree::TreeBase<C>,
    ctx: &TreeCtx<T>,
) {
    for child in tree.root().children() {
        child.write_buffers(out, ctx);
    }
}

/// Context for writing a tree with the given background and settings.
#[cfg(feature = "write")]
pub(crate) fn write_ctx<T: IoValue>(
    tree: &Tree<T>,
    compression: Compression,
    half: bool,
    leaf_format: LeafFormat,
) -> TreeCtx<T> {
    TreeCtx {
        codec: ValueCodec { file_version: super::FILE_VERSION, compression, half, background: *tree.background() },
        leaf_format,
    }
}

#[cfg(all(test, feature = "read", feature = "write"))]
mod tests {
    use super::*;
    use crate::math::CoordBBox;
    use crate::tree::Internal2;

    fn roundtrip<T: IoValue>(tree: &Tree<T>, c: Compression, half: bool, fmt: LeafFormat) -> Tree<T> {
        let ctx = write_ctx(tree, c, half, fmt);
        let mut out = Vec::new();
        write_tree_topology(&mut out, tree, &ctx);
        write_tree_buffers(&mut out, tree, &ctx);
        let mut r = Reader::new(&out[..]);
        let back = read_tree::<T, Internal2<T>, _>(&mut r, super::super::FILE_VERSION, c, half, fmt).unwrap();
        assert_eq!(r.pos() as usize, out.len());
        back
    }

    #[test]
    fn float_tree_roundtrip() {
        let mut t = Tree::<f32>::new(1.5);
        t.fill(&CoordBBox::from_coords(-300, 0, 0, -1, 200, 140), 0.25, true);
        for i in 0..200 {
            t.set_value(&Coord::new(i * 3, -i, i % 7), i as f32);
            t.set_value_off(&Coord::new(i * 3 + 1, -i, i % 7), -1.5);
        }
        t.root_mut().add_root_tile(&Coord::new(8192, 0, 0), 3.0, true);
        for c in [
            Compression::NONE,
            Compression::ZIP,
            Compression::ACTIVE_MASK,
            Compression::ZIP | Compression::ACTIVE_MASK,
            Compression::BLOSC | Compression::ACTIVE_MASK,
        ] {
            let back = roundtrip(&t, c, false, LeafFormat::Standard);
            assert!(back == t, "compression {}", c);
        }
    }

    #[test]
    fn bool_and_mask_roundtrip() {
        let mut t = Tree::<bool>::new(false);
        t.fill(&CoordBBox::from_coords(0, 0, 0, 255, 127, 127), true, true);
        for i in 0..100 {
            t.set_value(&Coord::new(-i, i * 2, 5), i % 2 == 0);
            t.set_value_off(&Coord::new(-i, i * 2, 6), true);
        }
        let back = roundtrip(&t, Compression::ZIP | Compression::ACTIVE_MASK, false, LeafFormat::Bool);
        assert!(back == t);

        // In a mask tree, values equal active states.
        let mut m = Tree::<bool>::new(false);
        for i in 0..100 {
            m.set_value(&Coord::new(i, -i, 2 * i), true);
        }
        let back = roundtrip(&m, Compression::ACTIVE_MASK, false, LeafFormat::Mask);
        assert!(back == m);
    }
}
