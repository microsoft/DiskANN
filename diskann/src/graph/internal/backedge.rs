/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt::Debug, ops::Deref};

use diskann_vector::contains::ContainsSimd;

use crate::graph::AdjacencyList;

/// A specialized vector-like like data structure for holding unique values, targeting the
/// "backedge aggregation" phase of `multi-insert`.
///
/// A key characteristic of these backedges is that the distribution in backedge length is
/// heavily skewed towards lower values (e.g. one or two).
///
/// To optimize for this common case, this data structure can hold up to four unique values
/// without allocating.
#[derive(Debug, Clone)]
pub(crate) struct BackedgeBuffer<I> {
    inner: Inner<I>,
}

impl<I> Default for BackedgeBuffer<I> {
    fn default() -> Self {
        Self { inner: Inner::None }
    }
}

// Private inner enum.
#[derive(Debug, Clone)]
enum Inner<I> {
    None,
    One(I),
    Two([I; 2]),
    Three([I; 3]),
    Four([I; 4]),
    Many(AdjacencyList<I>),
}

impl<I> BackedgeBuffer<I> {
    /// Construct a new [`BackedgeBuffer`] of length 1 containing `value`.
    ///
    /// This method does not allocate.
    pub(crate) fn new(value: I) -> Self {
        Self {
            inner: Inner::One(value),
        }
    }

    /// Return the number of unique items contained in the buffer.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        match &self.inner {
            Inner::None => 0,
            Inner::One(_) => 1,
            Inner::Two(_) => 2,
            Inner::Three(_) => 3,
            Inner::Four(_) => 4,
            Inner::Many(adj) => adj.len(),
        }
    }

    /// Return `true` if the buffer is empty.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the contents of the buffer as a slice.
    #[cfg(test)]
    pub(crate) fn as_slice(&self) -> &[I] {
        self
    }
}

impl<I> BackedgeBuffer<I>
where
    I: Copy + Debug + Default + PartialEq + ContainsSimd,
{
    /// Append `value` to the buffer if it does not compare equal to any value already in
    /// the buffer. Return `true` if `value` was inserted.
    ///
    /// If `self.len() <= 3` before calling this method, insertion will not allocate.
    pub(crate) fn push(&mut self, value: I) -> bool {
        // NOTE: Structuring the replacement this way by replacing `self.inner` and then
        // overwriting again results in much better code generation than working through
        // a mutable reference.
        let (new, inserted) = match std::mem::replace(&mut self.inner, Inner::None) {
            Inner::None => (Inner::One(value), true),
            Inner::One(v) => {
                if v != value {
                    (Inner::Two([v, value]), true)
                } else {
                    (Inner::One(v), false)
                }
            }
            Inner::Two([v0, v1]) => {
                if v0 != value && v1 != value {
                    (Inner::Three([v0, v1, value]), true)
                } else {
                    (Inner::Two([v0, v1]), false)
                }
            }
            Inner::Three([v0, v1, v2]) => {
                if v0 != value && v1 != value && v2 != value {
                    (Inner::Four([v0, v1, v2, value]), true)
                } else {
                    (Inner::Three([v0, v1, v2]), false)
                }
            }
            Inner::Four([v0, v1, v2, v3]) => {
                if v0 != value && v1 != value && v2 != value && v3 != value {
                    let mut list = AdjacencyList::with_capacity(5);
                    let mut guard = list.resize(5);

                    // SAFETY: We just sized the underlying slice to be 5 long. All accesses
                    // are in-bounds.
                    unsafe {
                        *guard.get_unchecked_mut(0) = v0;
                        *guard.get_unchecked_mut(1) = v1;
                        *guard.get_unchecked_mut(2) = v2;
                        *guard.get_unchecked_mut(3) = v3;
                        *guard.get_unchecked_mut(4) = value
                    };
                    guard.finish(5);
                    (Inner::Many(list), true)
                } else {
                    (Inner::Four([v0, v1, v2, v3]), false)
                }
            }
            Inner::Many(mut adj_list) => {
                let inserted = push_slow(&mut adj_list, value);
                (Inner::Many(adj_list), inserted)
            }
        };
        self.inner = new;
        inserted
    }
}

// Outline to prevent codegen leakage into `push`.
#[inline(never)]
fn push_slow<I>(list: &mut AdjacencyList<I>, v: I) -> bool
where
    I: ContainsSimd + Copy + Debug,
{
    list.push(v)
}

impl<I> Deref for BackedgeBuffer<I> {
    type Target = [I];
    fn deref(&self) -> &[I] {
        match &self.inner {
            Inner::None => &[],
            Inner::One(v) => std::slice::from_ref(v),
            Inner::Two(v) => v,
            Inner::Three(v) => v,
            Inner::Four(v) => v,
            Inner::Many(v) => v,
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sized() {
        assert_eq!(
            std::mem::size_of::<BackedgeBuffer<u32>>(),
            std::mem::size_of::<Vec<u32>>()
        );
    }

    #[test]
    fn test_buffer_new() {
        let mut buf = BackedgeBuffer::<u32>::new(10);

        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());
        assert_eq!(buf.as_slice(), &[10]);

        assert!(!buf.push(10), "10 should already be in the buffer");
    }

    #[test]
    fn test_buffer() {
        let values = [1u32, 2, 3, 4, 5, 6, 7, 8, 9];

        let mut buf = BackedgeBuffer::default();
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        assert_eq!(buf.as_slice(), &[] as &[u32]);

        for (i, v) in values.iter().enumerate() {
            assert!(
                buf.push(*v),
                "push should succeed since {} is not in the buffer",
                v,
            );

            assert_eq!(buf.len(), i + 1);
            assert!(!buf.is_empty());
            assert_eq!(buf.as_slice(), &values[..(i + 1)]);

            for j in values.iter().take(i + 1) {
                assert!(!buf.push(*j), "repeat elements are not allowed");
            }
        }
    }
}
