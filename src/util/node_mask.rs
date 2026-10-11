//! Bit masks used by tree nodes (port of `openvdb/util/NodeMasks.h`).

use std::fmt;

/// Number of words stored inline; larger masks (internal nodes) go on the heap.
const INLINE_WORDS: usize = 8;

/// A bit mask with `1 << (3 * LOG2DIM)` bits, stored as 64-bit words.
///
/// The word layout matches OpenVDB's `util::NodeMask`, so masks can be (de)serialized
/// directly as little-endian `u64` arrays. Masks of up to 512 bits (leaf masks) are stored
/// inline, so leaves need no separate allocation for their mask.
#[derive(Clone)]
pub struct NodeMask<const LOG2DIM: u32> {
    inline: [u64; INLINE_WORDS],
    heap: Box<[u64]>,
}

impl<const LOG2DIM: u32> PartialEq for NodeMask<LOG2DIM> {
    fn eq(&self, o: &Self) -> bool {
        self.words() == o.words()
    }
}
impl<const LOG2DIM: u32> Eq for NodeMask<LOG2DIM> {}
impl<const LOG2DIM: u32> std::hash::Hash for NodeMask<LOG2DIM> {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.words().hash(h)
    }
}

impl<const LOG2DIM: u32> NodeMask<LOG2DIM> {
    /// Number of bits in the mask.
    pub const SIZE: usize = 1 << (3 * LOG2DIM);
    /// Number of 64-bit words.
    pub const WORD_COUNT: usize = if Self::SIZE >= 64 { Self::SIZE >> 6 } else { 1 };

    /// A mask with all bits off.
    const IS_INLINE: bool = Self::WORD_COUNT <= INLINE_WORDS;

    pub fn new() -> Self {
        let heap = if Self::IS_INLINE { Box::default() } else { vec![0u64; Self::WORD_COUNT].into_boxed_slice() };
        NodeMask { inline: [0; INLINE_WORDS], heap }
    }
    /// A mask with all bits set to `on`.
    pub fn new_with(on: bool) -> Self {
        let mut m = Self::new();
        if on {
            m.set_all_on();
        }
        m
    }
    #[inline]
    fn tail_mask() -> u64 {
        if Self::SIZE >= 64 {
            !0
        } else {
            (1u64 << Self::SIZE) - 1
        }
    }
    #[inline]
    pub fn is_on(&self, n: usize) -> bool {
        debug_assert!(n < Self::SIZE);
        (self.w()[n >> 6] >> (n & 63)) & 1 != 0
    }
    #[inline]
    pub fn is_off(&self, n: usize) -> bool {
        !self.is_on(n)
    }
    #[inline]
    pub fn set_on(&mut self, n: usize) {
        debug_assert!(n < Self::SIZE);
        self.w_mut()[n >> 6] |= 1u64 << (n & 63);
    }
    #[inline]
    pub fn set_off(&mut self, n: usize) {
        debug_assert!(n < Self::SIZE);
        self.w_mut()[n >> 6] &= !(1u64 << (n & 63));
    }
    #[inline]
    pub fn set(&mut self, n: usize, on: bool) {
        if on {
            self.set_on(n)
        } else {
            self.set_off(n)
        }
    }
    #[inline]
    pub fn toggle(&mut self, n: usize) {
        self.w_mut()[n >> 6] ^= 1u64 << (n & 63);
    }
    pub fn set_all_on(&mut self) {
        let t = Self::tail_mask();
        for w in self.w_mut().iter_mut() {
            *w = t;
        }
    }
    pub fn set_all_off(&mut self) {
        for w in self.w_mut().iter_mut() {
            *w = 0;
        }
    }
    pub fn set_all(&mut self, on: bool) {
        if on {
            self.set_all_on()
        } else {
            self.set_all_off()
        }
    }
    pub fn toggle_all(&mut self) {
        let t = Self::tail_mask();
        for w in self.w_mut().iter_mut() {
            *w = !*w & t;
        }
    }
    /// Number of bits that are on.
    pub fn count_on(&self) -> u32 {
        self.w().iter().map(|w| w.count_ones()).sum()
    }
    pub fn count_off(&self) -> u32 {
        Self::SIZE as u32 - self.count_on()
    }
    /// True if all bits are on.
    pub fn is_all_on(&self) -> bool {
        let t = Self::tail_mask();
        self.w().iter().all(|w| *w == t)
    }
    /// True if all bits are off.
    pub fn is_all_off(&self) -> bool {
        self.w().iter().all(|w| *w == 0)
    }
    /// True if all bits are on or all are off; returns `Some(state)` in that case.
    pub fn is_constant(&self) -> Option<bool> {
        if self.is_all_on() {
            Some(true)
        } else if self.is_all_off() {
            Some(false)
        } else {
            None
        }
    }
    pub fn find_first_on(&self) -> Option<usize> {
        self.find_next_on(0)
    }
    pub fn find_first_off(&self) -> Option<usize> {
        self.find_next_off(0)
    }
    /// Index of the first on bit at or after `start`.
    pub fn find_next_on(&self, start: usize) -> Option<usize> {
        if start >= Self::SIZE {
            return None;
        }
        let mut wi = start >> 6;
        let mut w = self.w()[wi] & (!0u64 << (start & 63));
        loop {
            if w != 0 {
                let n = (wi << 6) + w.trailing_zeros() as usize;
                return if n < Self::SIZE { Some(n) } else { None };
            }
            wi += 1;
            if wi >= Self::WORD_COUNT {
                return None;
            }
            w = self.w()[wi];
        }
    }
    /// Index of the first off bit at or after `start`.
    pub fn find_next_off(&self, start: usize) -> Option<usize> {
        if start >= Self::SIZE {
            return None;
        }
        let mut wi = start >> 6;
        let mut w = !self.w()[wi] & (!0u64 << (start & 63));
        loop {
            if w != 0 {
                let n = (wi << 6) + w.trailing_zeros() as usize;
                return if n < Self::SIZE { Some(n) } else { None };
            }
            wi += 1;
            if wi >= Self::WORD_COUNT {
                return None;
            }
            w = !self.w()[wi];
        }
    }
    /// Iterate over the indices of the on bits.
    pub fn iter_on(&self) -> MaskOnIter<'_> {
        MaskOnIter { words: self.w(), wi: 0, cur: self.w().first().copied().unwrap_or(0), size: Self::SIZE }
    }
    /// Iterate over the indices of the off bits.
    pub fn iter_off(&self) -> impl Iterator<Item = usize> + '_ {
        (0..Self::SIZE).filter(move |&i| self.is_off(i))
    }
    /// Raw words (for serialization).
    #[inline]
    pub fn words(&self) -> &[u64] {
        self.w()
    }
    #[inline]
    pub fn words_mut(&mut self) -> &mut [u64] {
        self.w_mut()
    }
    #[inline(always)]
    fn w(&self) -> &[u64] {
        if Self::IS_INLINE {
            &self.inline[..Self::WORD_COUNT]
        } else {
            &self.heap
        }
    }
    #[inline(always)]
    fn w_mut(&mut self) -> &mut [u64] {
        if Self::IS_INLINE {
            &mut self.inline[..Self::WORD_COUNT]
        } else {
            &mut self.heap
        }
    }
    /// Bitwise union.
    pub fn union_with(&mut self, o: &Self) {
        for (a, b) in self.w_mut().iter_mut().zip(o.w().iter()) {
            *a |= *b;
        }
    }
    /// Bitwise intersection.
    pub fn intersect_with(&mut self, o: &Self) {
        for (a, b) in self.w_mut().iter_mut().zip(o.w().iter()) {
            *a &= *b;
        }
    }
    /// Bitwise difference (`self & !o`).
    pub fn difference_with(&mut self, o: &Self) {
        for (a, b) in self.w_mut().iter_mut().zip(o.w().iter()) {
            *a &= !*b;
        }
    }
    /// Bitwise symmetric difference.
    pub fn xor_with(&mut self, o: &Self) {
        for (a, b) in self.w_mut().iter_mut().zip(o.w().iter()) {
            *a ^= *b;
        }
    }
    /// Size of the mask in bytes.
    pub fn mem_usage(&self) -> usize {
        Self::WORD_COUNT * 8
    }
}

impl<const LOG2DIM: u32> Default for NodeMask<LOG2DIM> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const LOG2DIM: u32> fmt::Debug for NodeMask<LOG2DIM> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeMask<{}>({} on)", LOG2DIM, self.count_on())
    }
}

/// Iterator over the on bits of a [`NodeMask`].
pub struct MaskOnIter<'a> {
    words: &'a [u64],
    wi: usize,
    cur: u64,
    size: usize,
}

impl Iterator for MaskOnIter<'_> {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.cur != 0 {
                let b = self.cur.trailing_zeros() as usize;
                self.cur &= self.cur - 1;
                let n = (self.wi << 6) + b;
                return if n < self.size { Some(n) } else { None };
            }
            self.wi += 1;
            if self.wi >= self.words.len() {
                return None;
            }
            self.cur = self.words[self.wi];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mask_ops() {
        let mut m = NodeMask::<3>::new();
        assert_eq!(NodeMask::<3>::SIZE, 512);
        assert!(m.is_all_off());
        m.set_on(0);
        m.set_on(70);
        m.set_on(511);
        assert_eq!(m.count_on(), 3);
        assert_eq!(m.iter_on().collect::<Vec<_>>(), vec![0, 70, 511]);
        assert_eq!(m.find_next_on(1), Some(70));
        assert_eq!(m.find_first_off(), Some(1));
        m.set_all_on();
        assert!(m.is_all_on());
        assert_eq!(m.find_first_off(), None);
        let mut s = NodeMask::<1>::new();
        assert_eq!(NodeMask::<1>::SIZE, 8);
        s.set_all_on();
        assert_eq!(s.count_on(), 8);
        assert!(s.is_all_on());
        s.toggle_all();
        assert!(s.is_all_off());
    }
}
