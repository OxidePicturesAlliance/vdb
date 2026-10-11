//! The root node: a sparse, unbounded map of top-level children and tiles
//! (port of `openvdb/tree/RootNode.h`).

use super::leaf::LeafNode;
use super::node::Node;
use crate::math::{Coord, CoordBBox};
use crate::types::Value;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// An entry in the root table: either a child node or a tile.
#[derive(Clone)]
pub enum RootEntry<C: Node> {
    Child(Box<C>),
    Tile { value: C::Value, active: bool },
}

impl<C: Node> RootEntry<C> {
    pub fn is_child(&self) -> bool {
        matches!(self, RootEntry::Child(_))
    }
    pub fn is_tile(&self) -> bool {
        !self.is_child()
    }
    pub fn child(&self) -> Option<&C> {
        match self {
            RootEntry::Child(c) => Some(c),
            _ => None,
        }
    }
    pub fn child_mut(&mut self) -> Option<&mut C> {
        match self {
            RootEntry::Child(c) => Some(c),
            _ => None,
        }
    }
}

/// The root of a tree. Children are keyed by their origin, in lexicographic coordinate order.
pub struct RootNode<C: Node> {
    table: BTreeMap<Coord, RootEntry<C>>,
    background: C::Value,
}

impl<C: Node> Clone for RootNode<C> {
    fn clone(&self) -> Self {
        // Children are cloned in parallel (each is itself cloned in parallel below).
        let entries: Vec<(&Coord, &RootEntry<C>)> = self.table.iter().collect();
        let table = entries.into_par_iter().map(|(k, e)| (*k, e.clone())).collect::<Vec<_>>().into_iter().collect();
        RootNode { table, background: self.background }
    }
}

impl<C: Node> Default for RootNode<C> {
    fn default() -> Self {
        Self::new(C::Value::zero())
    }
}

impl<C: Node> RootNode<C> {
    pub const LEVEL: u32 = C::LEVEL + 1;

    pub fn new(background: C::Value) -> Self {
        RootNode { table: BTreeMap::new(), background }
    }
    /// Table key for the child containing `xyz`.
    #[inline]
    pub fn coord_to_key(xyz: &Coord) -> Coord {
        xyz.align_down(C::DIM)
    }
    #[inline]
    pub fn background(&self) -> &C::Value {
        &self.background
    }
    /// Change the background value, updating inactive tiles and voxels that had the old one.
    pub fn set_background(&mut self, background: C::Value, update_child_nodes: bool) {
        let old = self.background;
        if old == background {
            return;
        }
        if update_child_nodes {
            for e in self.table.values_mut() {
                match e {
                    RootEntry::Child(c) => c.reset_background(&old, &background),
                    RootEntry::Tile { value, active } => {
                        if !*active && *value == old {
                            *value = background;
                        }
                    }
                }
            }
        }
        self.background = background;
    }
    /// Set the background value without touching any nodes.
    pub fn set_background_unchecked(&mut self, background: C::Value) {
        self.background = background;
    }
    pub fn table(&self) -> &BTreeMap<Coord, RootEntry<C>> {
        &self.table
    }
    pub fn table_mut(&mut self) -> &mut BTreeMap<Coord, RootEntry<C>> {
        &mut self.table
    }
    pub fn clear(&mut self) {
        self.table.clear();
    }
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
    /// True if the table only holds inactive background tiles.
    pub fn is_background_only(&self) -> bool {
        self.table.values().all(|e| matches!(e, RootEntry::Tile { value, active: false } if *value == self.background))
    }
    pub fn child_count(&self) -> usize {
        self.table.values().filter(|e| e.is_child()).count()
    }
    pub fn tile_count(&self) -> usize {
        self.table.values().filter(|e| e.is_tile()).count()
    }
    /// Iterate over the top-level children.
    pub fn children(&self) -> impl Iterator<Item = &C> + '_ {
        self.table.values().filter_map(|e| e.child())
    }
    pub fn children_mut(&mut self) -> impl Iterator<Item = &mut C> + '_ {
        self.table.values_mut().filter_map(|e| e.child_mut())
    }
    /// Iterate over `(origin, value, active)` for root-level tiles.
    pub fn tiles(&self) -> impl Iterator<Item = (Coord, &C::Value, bool)> + '_ {
        self.table.iter().filter_map(|(k, e)| match e {
            RootEntry::Tile { value, active } => Some((*k, value, *active)),
            _ => None,
        })
    }
    /// Return the top-level child containing `xyz`, if any.
    pub fn probe_child(&self, xyz: &Coord) -> Option<&C> {
        self.table.get(&Self::coord_to_key(xyz))?.child()
    }
    pub fn probe_child_mut(&mut self, xyz: &Coord) -> Option<&mut C> {
        self.table.get_mut(&Self::coord_to_key(xyz))?.child_mut()
    }
    /// Insert a top-level child, replacing anything at its position.
    pub fn add_child(&mut self, child: Box<C>) {
        let key = Self::coord_to_key(&child.origin());
        self.table.insert(key, RootEntry::Child(child));
    }
    /// Insert a root-level tile at the key containing `xyz`.
    pub fn add_root_tile(&mut self, xyz: &Coord, value: C::Value, active: bool) {
        self.table.insert(Self::coord_to_key(xyz), RootEntry::Tile { value, active });
    }
    /// Remove inactive tiles whose value equals the background.
    pub fn erase_background_tiles(&mut self) {
        let bg = self.background;
        self.table.retain(|_, e| !matches!(e, RootEntry::Tile { value, active: false } if *value == bg));
    }

    /// Return the top-level child at `xyz`, creating it from the tile or background if needed.
    pub fn touch_child(&mut self, xyz: &Coord) -> &mut C {
        let key = Self::coord_to_key(xyz);
        let bg = self.background;
        let e = self.table.entry(key).or_insert(RootEntry::Tile { value: bg, active: false });
        if let RootEntry::Tile { value, active } = *e {
            *e = RootEntry::Child(Box::new(C::new(key, value, active)));
        }
        e.child_mut().unwrap()
    }

    #[inline]
    pub fn get_value(&self, xyz: &Coord) -> &C::Value {
        match self.table.get(&Self::coord_to_key(xyz)) {
            None => &self.background,
            Some(RootEntry::Tile { value, .. }) => value,
            Some(RootEntry::Child(c)) => c.get_value(xyz),
        }
    }
    #[inline]
    pub fn is_value_on(&self, xyz: &Coord) -> bool {
        match self.table.get(&Self::coord_to_key(xyz)) {
            None => false,
            Some(RootEntry::Tile { active, .. }) => *active,
            Some(RootEntry::Child(c)) => c.is_value_on(xyz),
        }
    }
    pub fn probe_value(&self, xyz: &Coord) -> (C::Value, bool) {
        match self.table.get(&Self::coord_to_key(xyz)) {
            None => (self.background, false),
            Some(RootEntry::Tile { value, active }) => (*value, *active),
            Some(RootEntry::Child(c)) => c.probe_value(xyz),
        }
    }
    /// Level at which the value at `xyz` is stored; -1 if it is the background.
    pub fn get_value_depth(&self, xyz: &Coord) -> i32 {
        match self.table.get(&Self::coord_to_key(xyz)) {
            None => -1,
            Some(RootEntry::Tile { .. }) => 0,
            Some(RootEntry::Child(c)) => (Self::LEVEL - c.get_value_level(xyz)) as i32,
        }
    }
    pub fn get_value_level(&self, xyz: &Coord) -> u32 {
        match self.table.get(&Self::coord_to_key(xyz)) {
            None | Some(RootEntry::Tile { .. }) => Self::LEVEL,
            Some(RootEntry::Child(c)) => c.get_value_level(xyz),
        }
    }
    pub fn set_value_on(&mut self, xyz: &Coord, value: C::Value) {
        if let Some(RootEntry::Tile { value: v, active: true }) = self.table.get(&Self::coord_to_key(xyz)) {
            if *v == value {
                return;
            }
        }
        self.touch_child(xyz).set_value_on(xyz, value);
    }
    pub fn set_value_off(&mut self, xyz: &Coord, value: C::Value) {
        match self.table.get(&Self::coord_to_key(xyz)) {
            None if value == self.background => return,
            Some(RootEntry::Tile { value: v, active: false }) if *v == value => return,
            _ => {}
        }
        self.touch_child(xyz).set_value_off(xyz, value);
    }
    pub fn set_value_only(&mut self, xyz: &Coord, value: C::Value) {
        match self.table.get(&Self::coord_to_key(xyz)) {
            None if value == self.background => return,
            Some(RootEntry::Tile { value: v, .. }) if *v == value => return,
            _ => {}
        }
        self.touch_child(xyz).set_value_only(xyz, value);
    }
    pub fn set_active_state(&mut self, xyz: &Coord, on: bool) {
        match self.table.get(&Self::coord_to_key(xyz)) {
            None if !on => return,
            Some(RootEntry::Tile { active, .. }) if *active == on => return,
            _ => {}
        }
        self.touch_child(xyz).set_active_state(xyz, on);
    }
    pub fn modify_value<F: FnOnce(&mut C::Value)>(&mut self, xyz: &Coord, f: F) {
        match self.table.get(&Self::coord_to_key(xyz)) {
            Some(RootEntry::Child(_)) => {}
            other => {
                let (old, on) = match other {
                    Some(RootEntry::Tile { value, active }) => (*value, *active),
                    _ => (self.background, false),
                };
                let mut v = old;
                f(&mut v);
                if on && v == old {
                    return;
                }
                self.touch_child(xyz).set_value_on(xyz, v);
                return;
            }
        }
        self.touch_child(xyz).modify_value(xyz, f);
    }
    pub fn modify_value_and_active_state<F: FnOnce(&mut C::Value, &mut bool)>(&mut self, xyz: &Coord, f: F) {
        match self.table.get(&Self::coord_to_key(xyz)) {
            Some(RootEntry::Child(_)) => {}
            other => {
                let (old, old_on) = match other {
                    Some(RootEntry::Tile { value, active }) => (*value, *active),
                    _ => (self.background, false),
                };
                let (mut v, mut on) = (old, old_on);
                f(&mut v, &mut on);
                if on == old_on && v == old {
                    return;
                }
                self.touch_child(xyz).add_tile(0, xyz, v, on);
                return;
            }
        }
        self.touch_child(xyz).modify_value_and_active_state(xyz, f);
    }

    /// Fill `bbox` with a value, using tiles where possible.
    pub fn fill(&mut self, bbox: &CoordBBox, value: C::Value, active: bool) {
        if bbox.is_empty() {
            return;
        }
        let dim = C::DIM as i64;
        let align = |v: i32| -> i64 { (v as i64).div_euclid(dim) * dim };
        let mut x = align(bbox.min[0]);
        while x <= bbox.max[0] as i64 {
            let mut y = align(bbox.min[1]);
            while y <= bbox.max[1] as i64 {
                let mut z = align(bbox.min[2]);
                while z <= bbox.max[2] as i64 {
                    let tile_min = Coord::new(x as i32, y as i32, z as i32);
                    let tile = CoordBBox::create_cube(tile_min, C::DIM);
                    let sub = tile.intersection(bbox);
                    if sub == tile {
                        self.table.insert(tile_min, RootEntry::Tile { value, active });
                    } else {
                        self.touch_child(&tile_min).fill(&sub, value, active);
                    }
                    z += dim;
                }
                y += dim;
            }
            x += dim;
        }
    }

    /// Mark every value in `bbox` active, keeping values unchanged.
    pub fn activate_region(&mut self, bbox: &CoordBBox) {
        if bbox.is_empty() {
            return;
        }
        let dim = C::DIM as i64;
        let align = |v: i32| -> i64 { (v as i64).div_euclid(dim) * dim };
        let mut x = align(bbox.min[0]);
        while x <= bbox.max[0] as i64 {
            let mut y = align(bbox.min[1]);
            while y <= bbox.max[1] as i64 {
                let mut z = align(bbox.min[2]);
                while z <= bbox.max[2] as i64 {
                    let key = Coord::new(x as i32, y as i32, z as i32);
                    let tile = CoordBBox::create_cube(key, C::DIM);
                    let bg = self.background;
                    let full = bbox.is_inside_bbox(&tile);
                    let e = self.table.entry(key).or_insert(RootEntry::Tile { value: bg, active: false });
                    match e {
                        RootEntry::Child(c) => c.activate_region(bbox),
                        RootEntry::Tile { active: true, .. } => {}
                        RootEntry::Tile { active, value } => {
                            if full {
                                *active = true;
                            } else {
                                let mut c = C::new(key, *value, false);
                                c.activate_region(bbox);
                                *e = RootEntry::Child(Box::new(c));
                            }
                        }
                    }
                    z += dim;
                }
                y += dim;
            }
            x += dim;
        }
    }
    pub fn touch_leaf(&mut self, xyz: &Coord) -> &mut LeafNode<C::Value> {
        self.touch_child(xyz).touch_leaf(xyz)
    }
    pub fn probe_leaf(&self, xyz: &Coord) -> Option<&LeafNode<C::Value>> {
        self.probe_child(xyz)?.probe_leaf(xyz)
    }
    pub fn probe_leaf_mut(&mut self, xyz: &Coord) -> Option<&mut LeafNode<C::Value>> {
        self.probe_child_mut(xyz)?.probe_leaf_mut(xyz)
    }
    pub fn add_leaf(&mut self, leaf: Box<LeafNode<C::Value>>) {
        let xyz = leaf.origin();
        self.touch_child(&xyz).add_leaf(leaf);
    }
    /// Add a tile at the given level (`Self::LEVEL` for a root tile, 0 for a voxel).
    pub fn add_tile(&mut self, level: u32, xyz: &Coord, value: C::Value, active: bool) {
        if level >= Self::LEVEL {
            self.add_root_tile(xyz, value, active);
        } else {
            self.touch_child(xyz).add_tile(level, xyz, value, active);
        }
    }

    pub fn prune(&mut self, tolerance: &C::Value) {
        for e in self.table.values_mut() {
            if let RootEntry::Child(c) = e {
                c.prune(tolerance);
                if let Some((value, active)) = c.is_constant(tolerance) {
                    *e = RootEntry::Tile { value, active };
                }
            }
        }
        self.erase_background_tiles();
    }
    pub fn prune_inactive(&mut self) {
        let bg = self.background;
        for e in self.table.values_mut() {
            if let RootEntry::Child(c) = e {
                c.prune_inactive(&bg);
                if c.is_inactive() {
                    *e = RootEntry::Tile { value: bg, active: false };
                }
            }
        }
        self.erase_background_tiles();
    }
    pub fn eval_active_bbox(&self, bbox: &mut CoordBBox, visit_voxels: bool) {
        for (k, e) in &self.table {
            match e {
                RootEntry::Child(c) => c.eval_active_bbox(bbox, visit_voxels),
                RootEntry::Tile { active: true, .. } => bbox.expand_cube(*k, C::DIM),
                _ => {}
            }
        }
    }
    pub fn active_voxel_count(&self) -> u64 {
        let entries: Vec<&RootEntry<C>> = self.table.values().collect();
        entries
            .into_par_iter()
            .map(|e| match e {
                RootEntry::Child(c) => c.active_voxel_count(),
                RootEntry::Tile { active: true, .. } => C::NUM_VOXELS,
                _ => 0,
            })
            .sum()
    }
    pub fn inactive_voxel_count(&self) -> u64 {
        self.table
            .values()
            .map(|e| match e {
                RootEntry::Child(c) => c.inactive_voxel_count(),
                RootEntry::Tile { active: false, value } if *value != self.background => C::NUM_VOXELS,
                _ => 0,
            })
            .sum()
    }
    pub fn active_leaf_voxel_count(&self) -> u64 {
        self.children().map(|c| c.active_leaf_voxel_count()).sum()
    }
    pub fn inactive_leaf_voxel_count(&self) -> u64 {
        self.children().map(|c| c.inactive_leaf_voxel_count()).sum()
    }
    pub fn active_tile_count(&self) -> u64 {
        self.table
            .values()
            .map(|e| match e {
                RootEntry::Child(c) => c.active_tile_count(),
                RootEntry::Tile { active, .. } => *active as u64,
            })
            .sum()
    }
    pub fn leaf_count(&self) -> usize {
        let children: Vec<&C> = self.children().collect();
        children.into_par_iter().map(|c| c.leaf_count()).sum()
    }
    /// Number of nodes at each level, from leaves (index 0) up to and including the root.
    pub fn node_count(&self) -> Vec<usize> {
        let mut counts = vec![0usize; Self::LEVEL as usize + 1];
        counts[Self::LEVEL as usize] = 1;
        for c in self.children() {
            c.node_count(&mut counts);
        }
        counts
    }
    pub fn leaves(&self) -> impl Iterator<Item = &LeafNode<C::Value>> + '_ {
        self.children().flat_map(|c| c.leaves())
    }
    pub fn leaves_mut(&mut self) -> impl Iterator<Item = &mut LeafNode<C::Value>> + '_ {
        self.children_mut().flat_map(|c| c.leaves_mut())
    }
    /// Visit every tile in the tree, including root tiles: `f(bbox, value, active, level)`.
    pub fn visit_tiles(&self, f: &mut dyn FnMut(CoordBBox, &C::Value, bool, u32)) {
        for (k, e) in &self.table {
            match e {
                RootEntry::Child(c) => c.visit_tiles(f),
                RootEntry::Tile { value, active } => f(CoordBBox::create_cube(*k, C::DIM), value, *active, Self::LEVEL),
            }
        }
    }
    pub fn visit_tiles_mut(&mut self, f: &mut dyn FnMut(CoordBBox, &mut C::Value, &mut bool, u32)) {
        for (k, e) in self.table.iter_mut() {
            match e {
                RootEntry::Child(c) => c.visit_tiles_mut(f),
                RootEntry::Tile { value, active } => f(CoordBBox::create_cube(*k, C::DIM), value, active, Self::LEVEL),
            }
        }
    }
    pub fn clip(&mut self, bbox: &CoordBBox) {
        let bg = self.background;
        let keys: Vec<Coord> = self.table.keys().copied().collect();
        for k in keys {
            let tile_bbox = CoordBBox::create_cube(k, C::DIM);
            if bbox.is_inside_bbox(&tile_bbox) {
                continue;
            }
            if !bbox.has_overlap(&tile_bbox) {
                self.table.remove(&k);
                continue;
            }
            match self.table.get_mut(&k).unwrap() {
                RootEntry::Child(c) => c.clip(bbox, &bg),
                RootEntry::Tile { value, active } => {
                    let (v, on) = (*value, *active);
                    let mut child = C::new(k, bg, false);
                    child.fill(&tile_bbox.intersection(bbox), v, on);
                    self.table.insert(k, RootEntry::Child(Box::new(child)));
                }
            }
        }
        self.prune_inactive_background_children();
    }
    fn prune_inactive_background_children(&mut self) {
        let bg = self.background;
        self.table.retain(|_, e| match e {
            RootEntry::Child(c) => !(c.is_inactive() && c.is_constant(&bg).is_some_and(|(v, _)| v == bg)),
            RootEntry::Tile { value, active } => *active || *value != bg,
        });
    }
    pub fn voxelize_active_tiles(&mut self) {
        let keys: Vec<Coord> = self
            .table
            .iter()
            .filter(|(_, e)| matches!(e, RootEntry::Tile { active: true, .. }))
            .map(|(k, _)| *k)
            .collect();
        for k in keys {
            self.touch_child(&k);
        }
        for c in self.children_mut() {
            c.voxelize_active_tiles();
        }
    }
    pub fn set_values_on_off(&mut self, on: bool) {
        for e in self.table.values_mut() {
            match e {
                RootEntry::Child(c) => c.set_values_on_off(on),
                RootEntry::Tile { active, .. } => *active = on,
            }
        }
    }
    pub fn mem_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.table.len() * (std::mem::size_of::<Coord>() + std::mem::size_of::<RootEntry<C>>())
            + self.children().map(|c| c.mem_usage()).sum::<usize>()
    }
}
