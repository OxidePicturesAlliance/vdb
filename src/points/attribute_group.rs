//! Point group membership stored as bits of `uint8` group arrays (port of
//! `openvdb/points/AttributeGroup.h`).
//!
//! Each group array (`("uint8", "grp")`) holds eight groups, one bit per group. A group is
//! identified by a [`GroupIndex`]: the position of its array in the attribute set and its
//! bit offset within that array.

use super::attribute_array::AttributeArray;
use super::{Error, Result};

/// The storage type of group arrays (OpenVDB's `GroupType`).
pub type GroupType = u8;

/// Number of groups per group array.
pub const GROUP_BITS: usize = 8;

/// Location of a group: (attribute array position, bit offset) (OpenVDB's `GroupIndex`).
pub type GroupIndex = (usize, u8);

fn check_group(array: &AttributeArray) -> Result<()> {
    if array.is_group() {
        Ok(())
    } else {
        Err(Error::Type(format!("attribute array of type {} is not a group array", array.attribute_type())))
    }
}

/// Read access to one group of a group array (OpenVDB's `GroupHandle`).
#[derive(Clone, Copy)]
pub struct GroupHandle<'a> {
    array: &'a AttributeArray,
    mask: GroupType,
}

impl<'a> GroupHandle<'a> {
    /// A handle for the group at bit `offset` of `array`.
    pub fn new(array: &'a AttributeArray, offset: u8) -> Result<Self> {
        check_group(array)?;
        Ok(GroupHandle { array, mask: 1u8 << offset })
    }
    /// A handle testing all bits of `mask`.
    pub fn with_mask(array: &'a AttributeArray, mask: GroupType) -> Result<Self> {
        check_group(array)?;
        Ok(GroupHandle { array, mask })
    }
    /// Number of points.
    pub fn size(&self) -> u32 {
        self.array.size()
    }
    /// True if the group array is uniform.
    pub fn is_uniform(&self) -> bool {
        self.array.is_uniform()
    }
    /// True if point `n` is in the group.
    #[inline]
    pub fn get(&self, n: u32) -> bool {
        (self.array.storage_at(n)[0] & self.mask) == self.mask
    }
}

/// Read-write access to one group of a group array (OpenVDB's `GroupWriteHandle`).
pub struct GroupWriteHandle<'a> {
    array: &'a mut AttributeArray,
    mask: GroupType,
}

impl<'a> GroupWriteHandle<'a> {
    /// A handle for the group at bit `offset` of `array`.
    pub fn new(array: &'a mut AttributeArray, offset: u8) -> Result<Self> {
        check_group(array)?;
        Ok(GroupWriteHandle { array, mask: 1u8 << offset })
    }
    /// True if point `n` is in the group.
    #[inline]
    pub fn get(&self, n: u32) -> bool {
        (self.array.storage_at(n)[0] & self.mask) == self.mask
    }
    /// True if the group array is uniform.
    pub fn is_uniform(&self) -> bool {
        self.array.is_uniform()
    }
    /// Add point `n` to (or remove it from) the group, expanding the array if needed.
    pub fn set(&mut self, n: u32, on: bool) {
        let v = self.array.storage_at(n)[0];
        let nv = if on { v | self.mask } else { v & !self.mask };
        if nv != v {
            self.array.set::<u8>(n, nv);
        }
    }
    /// Set the membership of every point; returns true if the array is uniform afterwards
    /// (`GroupWriteHandle::collapse`).
    pub fn collapse(&mut self, on: bool) -> bool {
        self.array.compact();
        if self.array.is_uniform() {
            let v = self.array.storage()[0];
            self.array.storage_mut()[0] = if on { v | self.mask } else { v & !self.mask };
            return true;
        }
        let mask = self.mask;
        for b in self.array.storage_mut() {
            *b = if on { *b | mask } else { *b & !mask };
        }
        false
    }
    /// Collapse the array if all values are equal.
    pub fn compact(&mut self) -> bool {
        self.array.compact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::points::Codec;

    #[test]
    fn group_bits() {
        let mut a = AttributeArray::with_size::<u8>(Codec::Group, 5).unwrap();
        {
            let mut w = GroupWriteHandle::new(&mut a, 3).unwrap();
            assert!(w.is_uniform());
            w.set(2, true);
            assert!(!w.is_uniform());
            assert!(w.get(2) && !w.get(1));
        }
        let h = GroupHandle::new(&a, 3).unwrap();
        assert!(h.get(2));
        assert!(!GroupHandle::new(&a, 2).unwrap().get(2));
        {
            let mut w = GroupWriteHandle::new(&mut a, 3).unwrap();
            w.set(2, false);
            assert!(w.collapse(true));
        }
        assert!(a.is_uniform());
        assert_eq!(a.storage()[0], 8);
        let f = AttributeArray::with_size::<f32>(Codec::Null, 2).unwrap();
        assert!(GroupHandle::new(&f, 0).is_err());
    }
}
