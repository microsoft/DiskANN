/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Adjacency List

use std::{
    ops::{Deref, DerefMut, Index},
    slice,
};

use byteorder::{ByteOrder, LittleEndian};
use diskann::{ANNError, ANNResult};
use diskann_vector::contains::ContainsSimd;

#[derive(Debug, Eq, Clone)]
/// Represents the out neighbors of a vertex
pub struct AdjacencyList<VectorIdType> {
    edges: Box<[VectorIdType]>,
    length: usize,
}

impl<VectorIdType> AdjacencyList<VectorIdType>
where
    VectorIdType: Default + Clone + std::marker::Copy,
{
    /// Create AdjacencyList with capacity slack for a range.
    pub fn for_range(range: usize, graph_slack_factor: f32) -> Self {
        let capacity = (range as f32 * graph_slack_factor) as usize;
        Self {
            edges: (0..capacity).map(|_| VectorIdType::default()).collect(),
            length: 0,
        }
    }

    /// Push a node to the list of neighbors for the given node.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() == self.capacity()`.
    pub fn push(&mut self, node_id: VectorIdType) {
        self.edges[self.length] = node_id;
        self.length += 1;
    }

    /// Clear the slice.
    pub fn clear(&mut self) {
        self.length = 0;
    }

    /// Get the length of the slice.
    pub fn capacity(&self) -> usize {
        self.edges.len()
    }

    /// Get the length of the slice.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Copy data from another adjacency list
    /// Will panic if `to_copy.len()` is longer than the list's capacity
    /// Will not panic if it is shorter
    pub fn copy_from(&mut self, to_copy: AdjacencyList<VectorIdType>) {
        let new_len = to_copy.len();
        self.edges[..new_len].copy_from_slice(&to_copy);
        self.length = new_len;
    }

    /// Check if the slice contains the given node.
    pub fn contains(&self, node_id: VectorIdType) -> bool
    where
        VectorIdType: ContainsSimd,
    {
        VectorIdType::contains_simd(self.as_slice(), node_id)
    }

    /// Try adding the node_id to the neighbors.
    /// If the node_id is already in the neighbors, return None.
    /// if the adjacency list is full, return the new list of neighbors in a vector.
    pub fn add_to_neighbors(&mut self, node_id: VectorIdType) -> Option<Vec<VectorIdType>>
    where
        VectorIdType: ContainsSimd + Copy,
    {
        // Check if n is already in the graph entry
        if self.contains(node_id) {
            return None;
        }

        let neighbor_len = self.len();

        // If not, check if there is capacity to add the node.
        if neighbor_len < self.capacity() {
            // If yes, add n to the graph entry
            self.push(node_id);
            return None;
        }

        let mut copy_of_neighbors = self.as_slice().to_vec();
        copy_of_neighbors.reserve_exact(1);
        copy_of_neighbors.push(node_id);

        Some(copy_of_neighbors)
    }

    /// Consume `self` and transfer its heap allocation into a `Vec`.
    pub fn into_vec(self) -> Vec<VectorIdType> {
        let len = self.len();
        let mut v: Vec<VectorIdType> = self.edges.into();
        v.truncate(len);
        v
    }

    /// Create an `AdjacencyList` from an existing `Vec`.
    /// Unlikd the `From<Vec<T>>` impl, this one keeps the `Vec`s capacity unchanged
    /// by filling unused elements with default values.
    pub fn from_vec(mut v: Vec<VectorIdType>) -> Self {
        let len = v.len();
        let cap = v.capacity();
        v.resize(cap, VectorIdType::default());
        Self {
            edges: v.into(),
            length: len,
        }
    }
}

impl<VectorIdType> PartialEq for AdjacencyList<VectorIdType>
where
    VectorIdType: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<VectorIdType> Index<usize> for AdjacencyList<VectorIdType> {
    type Output = VectorIdType;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(
            index < self.len(),
            "index {} must be less than length {}",
            index,
            self.len()
        );
        &self.edges[index]
    }
}

impl<VectorIdType> Deref for AdjacencyList<VectorIdType> {
    type Target = [VectorIdType];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<VectorIdType> DerefMut for AdjacencyList<VectorIdType> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: The slice is valid for the length of the slice.
        unsafe { std::slice::from_raw_parts_mut(self.edges.as_mut_ptr(), self.length) }
    }
}

/// Consume the adjacency list and creates an `Vec<u8>` from it.
/// An implementation of `From<AdjacencyList> for Vec<u8>` is preferred over `Into<Vec<u8>>` for
/// AdjacencyList since the former gives us `Into<_>` for free where the reverse isn't true.
/// For more details, see <https://rust-lang.github.io/rust-clippy/master/index.html#from_over_into>
impl<VectorIdType> From<AdjacencyList<VectorIdType>> for Vec<u8> {
    fn from(adjacency_list: AdjacencyList<VectorIdType>) -> Self {
        let len = adjacency_list.len();
        let mut bytes = Vec::with_capacity(4 + len * 4);
        bytes.extend_from_slice(&(len as u32).to_le_bytes());

        let ptr = adjacency_list.as_ptr() as *const u8;
        // SAFETY: The slice is valid for the length of the slice because the actual length is always smaller than capacity.
        let slice = unsafe { std::slice::from_raw_parts(ptr, len * 4) };
        bytes.extend_from_slice(slice);
        bytes
    }
}

/// Creates an adjacency list from a slice of bytes.
impl<VectorIdType> TryFrom<&[u8]> for AdjacencyList<VectorIdType>
where
    VectorIdType: Default + Clone,
{
    type Error = ANNError;
    fn try_from(bytes: &[u8]) -> ANNResult<Self> {
        if bytes.len() < 4 {
            return Err(ANNError::log_adjacency_list_conversion_error(
                "The given bytes are not long enough to create a valid adjacency list.".to_string(),
            ));
        }

        let nbr_count = LittleEndian::read_u32(bytes) as usize;

        if bytes.len() < 4 + nbr_count * 4 {
            return Err(ANNError::log_adjacency_list_conversion_error(
                "The given bytes are not long enough to create a valid adjacency list.".to_string(),
            ));
        }

        let vec = vec![VectorIdType::default(); nbr_count];
        let mut adjacency_list = AdjacencyList {
            edges: vec.into_boxed_slice(),
            length: nbr_count,
        };

        // Copy the bytes from the given slice to the edges of the adjacency list.
        // SAFETY: We have checked the length of slice above, and adjacency_list.edges is initialized with nbr_count length.
        unsafe {
            let src_ptr = bytes.as_ptr().add(4);
            std::ptr::copy_nonoverlapping(
                src_ptr,
                adjacency_list.edges.as_mut_ptr() as *mut u8,
                nbr_count * 4,
            );
        }

        Ok(adjacency_list)
    }
}

impl<VectorIdType> From<Vec<VectorIdType>> for AdjacencyList<VectorIdType> {
    fn from(edges: Vec<VectorIdType>) -> Self {
        let len = edges.len();
        Self {
            edges: edges.into(),
            length: len,
        }
    }
}

impl<VectorIdType> AdjacencyList<VectorIdType> {
    /// Return the slice of the adjacency list that is initialized.
    pub fn as_slice(&self) -> &[VectorIdType] {
        // SAFETY: The slice is valid for the length of the slice.
        unsafe { slice::from_raw_parts(self.edges.as_ptr(), self.length) }
    }
}

#[cfg(test)]
mod tests_adjacency_list {
    use super::*;

    #[test]
    fn test_serde_adjacency_list() {
        // Create an instance of AdjacencyList for testing
        let original_list = AdjacencyList::from(vec![1, 2, 3, 4, 5, 8, 7]);
        let original_list_cloned = AdjacencyList::from(vec![1, 2, 3, 4, 5, 8, 7]);
        // Serialize the AdjacencyList
        let serialized_data: Vec<u8> = original_list.into();
        assert!(serialized_data.len() == 4 + original_list_cloned.len() * 4);

        // Deserialize the binary data back to an AdjacencyList
        let deserialized_list: AdjacencyList<u32> =
            AdjacencyList::try_from(serialized_data.as_slice()).unwrap();

        // Check if the original and deserialized lists are equal
        assert_eq!(original_list_cloned, deserialized_list);
    }

    #[test]
    fn test_deref_mut() {
        let mut adj_list = AdjacencyList::from(vec![1, 2, 3]);
        let slice: &mut [u32] = &mut adj_list;
        slice[1] = 4;

        assert_eq!(adj_list[1], 4);
    }

    #[test]
    fn test_adjacency_list_try_from() {
        // Test case 1: valid input
        let bytes = [2u8, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0];
        let adjacency_list = AdjacencyList::<u32>::try_from(&bytes[..]).unwrap();
        assert_eq!(adjacency_list.len(), 2);
        assert_eq!(adjacency_list[0], 1);
        assert_eq!(adjacency_list[1], 3);

        // Test case 2: invalid input
        let bytes = [0u8, 0, 0, 1, 0, 0, 0, 1, 1];
        let result = AdjacencyList::<u32>::try_from(&bytes[..]);
        assert!(result.is_err());
    }

    #[test]
    fn test_adjacency_list_copy_from_should_succeed() {
        //Test case 1: copy a list of the same size
        let mut adj_list_test1 = AdjacencyList::from(vec![1, 2, 3]);
        let new_adj_list_test1 = AdjacencyList::from(vec![4, 5, 6]);
        adj_list_test1.copy_from(new_adj_list_test1);
        assert_eq!(adj_list_test1.len(), 3);
        assert_eq!(adj_list_test1[0], 4);
        assert_eq!(adj_list_test1[1], 5);
        assert_eq!(adj_list_test1[2], 6);

        //Test case 2: copy a shorter list
        let mut adj_list_test2 = AdjacencyList::from(vec![1, 2, 3]);
        let new_adj_list_test2 = AdjacencyList::from(vec![4, 5]);
        adj_list_test2.copy_from(new_adj_list_test2);
        assert_eq!(adj_list_test2.len(), 2);
        assert_eq!(adj_list_test2[0], 4);
        assert_eq!(adj_list_test2[1], 5);
    }

    #[test]
    #[should_panic]
    fn test_adjacency_list_copy_from_should_fail() {
        //Failing test case: try to copy a longer list to a shorter one, should panic
        let mut adj_list_test = AdjacencyList::from(vec![1, 2, 3]);
        let new_adj_list_test = AdjacencyList::from(vec![4, 5, 6, 7]);
        adj_list_test.copy_from(new_adj_list_test);
    }

    #[test]
    #[should_panic]
    fn test_push_panics() {
        let mut list = AdjacencyList::<u32>::for_range(2, 1.0);
        assert_eq!(list.capacity(), 2);
        list.push(1);
        list.push(2);
        list.push(3);
    }

    #[test]
    #[should_panic(expected = "index 3 must be less than length 3")]
    fn indexing_panics() {
        let mut list = AdjacencyList::<u32>::for_range(4, 1.0);
        assert_eq!(list.capacity(), 4);
        list.push(0);
        list.push(1);
        list.push(2);
        assert_eq!(list.len(), 3);
        let _: u32 = list[3];
    }

    #[test]
    fn test_into_vec() {
        {
            // Full.
            let list = AdjacencyList::<u32>::from(vec![1, 3, 4]);
            let ptr = list.edges.as_ptr();
            assert_eq!(list.len(), 3);
            assert_eq!(list.capacity(), 3);
            let v: Vec<u32> = list.into_vec();
            assert_eq!(v.len(), 3);
            assert_eq!(v, vec![1, 3, 4]);
            assert_eq!(v.as_ptr(), ptr, "heap allocation was not preservedc");
        }

        {
            // Partially full
            let mut list = AdjacencyList::<u32>::for_range(4, 1.0);
            list.push(1);
            list.push(2);
            list.push(3);
            let ptr = list.edges.as_ptr();
            assert_eq!(list.len(), 3);
            assert_eq!(list.capacity(), 4);
            let v: Vec<u32> = list.into_vec();
            assert_eq!(v.len(), 3);
            assert_eq!(v, vec![1, 2, 3]);
            assert_eq!(v.as_ptr(), ptr, "heap allocation was not preservedc");
        }

        {
            // Empty
            let list = AdjacencyList::<u32>::for_range(4, 1.0);
            let ptr = list.edges.as_ptr();
            assert_eq!(list.len(), 0);
            assert_eq!(list.capacity(), 4);
            let v: Vec<u32> = list.into_vec();
            assert_eq!(v.len(), 0);
            assert_eq!(v.as_ptr(), ptr, "heap allocation was not preservedc");
        }
    }
}
