//! Utilities.

pub mod node_mask;

pub use node_mask::NodeMask;

use crate::math::Coord;

/// `util::INVALID_IDX` (the maximum 32-bit unsigned value).
pub const INVALID_IDX: u32 = u32::MAX;

/// OpenVDB's `util::COORD_OFFSETS`: the 26 neighbour offsets (6 face, 12 edge, 8 corner).
pub const COORD_OFFSETS: [Coord; 26] = [
    Coord::new(1, 0, 0),
    Coord::new(-1, 0, 0),
    Coord::new(0, 1, 0),
    Coord::new(0, -1, 0),
    Coord::new(0, 0, 1),
    Coord::new(0, 0, -1),
    Coord::new(1, 0, -1),
    Coord::new(-1, 0, -1),
    Coord::new(1, 0, 1),
    Coord::new(-1, 0, 1),
    Coord::new(1, 1, 0),
    Coord::new(-1, 1, 0),
    Coord::new(1, -1, 0),
    Coord::new(-1, -1, 0),
    Coord::new(0, -1, 1),
    Coord::new(0, -1, -1),
    Coord::new(0, 1, 1),
    Coord::new(0, 1, -1),
    Coord::new(-1, -1, -1),
    Coord::new(-1, -1, 1),
    Coord::new(1, -1, 1),
    Coord::new(1, -1, -1),
    Coord::new(-1, 1, -1),
    Coord::new(-1, 1, 1),
    Coord::new(1, 1, 1),
    Coord::new(1, 1, -1),
];
