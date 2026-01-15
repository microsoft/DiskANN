/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::attribute::AttributeValue;
use roaring::RoaringBitmap;

/// A read-only trait for querying an inverted index.
///
/// This trait provides methods for retrieving posting lists from an inverted index
/// without modifying the index structure. It separates read operations from write
/// operations (insert/delete/update), following the principle of interface segregation.
///
/// Implementations of this trait should provide efficient lookup of posting lists
/// for specific field-value pairs.
pub trait PostingListAccessor {
    /// The error type returned by read operations on this inverted index.
    type Error: std::error::Error + Send + Sync + 'static;

    /// The posting list type used to store document IDs.
    type PostingList: PostingList;

    /// The document ID type used to identify documents in the index.
    type DocId: Copy + Into<usize> + From<usize> + std::fmt::Debug;

    /// Retrieves the posting list for a specific field-value pair.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name to query.
    /// * `value` - The value to look up in the specified field.
    ///
    /// # Returns
    ///
    /// Returns `Some(PostingList)` if the field-value pair exists in the index,
    /// `None` if it doesn't exist, or an error if the retrieval fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let posting_list = index.get_posting_list("color", &AttributeValue::String("red".into()))?;
    /// if let Some(pl) = posting_list {
    ///     println!("Found {} documents with color=red", pl.len());
    /// }
    /// ```
    fn get_posting_list(
        &self,
        field: &str,
        value: &AttributeValue,
    ) -> std::result::Result<Option<Self::PostingList>, Self::Error>;
}

/// A posting list that stores a set of document IDs.
///
/// This trait defines operations for managing and combining sets of document IDs,
/// typically used in inverted index implementations. It supports set operations
/// like union, intersection, and difference, as well as serialization.
pub trait PostingList: Clone + Sized + Send + Sync {
    /// The error type returned by operations on this posting list.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Returns the number of document IDs in the posting list.
    ///
    /// # Returns
    ///
    /// The count of document IDs in this posting list.
    fn len(&self) -> usize;

    /// Checks if a document ID is present in the posting list.
    ///
    /// # Arguments
    ///
    /// * `id` - The document ID to check.
    ///
    /// # Returns
    ///
    /// `true` if the document ID is in the posting list, `false` otherwise.
    fn contains(&self, id: usize) -> bool;

    /// Serializes the posting list to a byte vector.
    ///
    /// # Returns
    ///
    /// A byte vector representing the serialized posting list.
    fn serialize(&self) -> Vec<u8>;

    /// Deserializes a posting list from a byte slice.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The byte slice containing the serialized posting list.
    ///
    /// # Returns
    ///
    /// Returns the deserialized posting list, or an error if deserialization fails.
    fn deserialize(bytes: &[u8]) -> std::result::Result<Self, Self::Error>;

    /// Creates an empty posting list.
    ///
    /// # Returns
    ///
    /// A new, empty posting list.
    fn empty() -> Self;

    /// Checks if the posting list is empty.
    ///
    /// # Returns
    ///
    /// `true` if the posting list contains no document IDs, `false` otherwise.
    fn is_empty(&self) -> bool;

    /// Inserts a document ID into the posting list.
    ///
    /// # Arguments
    ///
    /// * `id` - The document ID to insert.
    ///
    /// # Returns
    ///
    /// `true` if the document ID was newly inserted (not already present),
    /// `false` if it was already in the list.
    fn insert(&mut self, id: usize) -> bool;

    /// Removes a document ID from the posting list.
    ///
    /// # Arguments
    ///
    /// * `id` - The document ID to remove.
    ///
    /// # Returns
    ///
    /// `true` if the document ID was present and removed, `false` if it wasn't present.
    fn remove(&mut self, id: usize) -> bool;

    /// Computes the union of this posting list with another.
    ///
    /// # Arguments
    ///
    /// * `other` - The other posting list to union with.
    ///
    /// # Returns
    ///
    /// A new posting list containing all document IDs from both lists.
    fn union(&self, other: &Self) -> Self;

    /// Computes the intersection of this posting list with another.
    ///
    /// # Arguments
    ///
    /// * `other` - The other posting list to intersect with.
    ///
    /// # Returns
    ///
    /// A new posting list containing only document IDs present in both lists.
    fn intersect(&self, other: &Self) -> Self;

    /// Computes the difference of this posting list with another.
    ///
    /// # Arguments
    ///
    /// * `other` - The other posting list to subtract.
    ///
    /// # Returns
    ///
    /// A new posting list containing document IDs in this list but not in the other.
    fn difference(&self, other: &Self) -> Self;
}

/// A posting list implementation using RoaringBitmap for efficient storage.
///
/// This implementation uses the Roaring bitmap data structure which provides
/// excellent compression and performance for sparse sets of integers.
#[derive(Clone, Debug)]
pub struct RoaringPostingList {
    /// The underlying Roaring bitmap storing document IDs.
    pub rb: RoaringBitmap,
}

impl RoaringPostingList {
    /// Creates a new posting list from a RoaringBitmap.
    ///
    /// # Arguments
    ///
    /// * `rb` - The RoaringBitmap to wrap.
    ///
    /// # Returns
    ///
    /// A new `RoaringPostingList` instance.
    pub fn new(rb: RoaringBitmap) -> Self {
        Self { rb }
    }
}

/// Error type for posting list operations.
#[derive(Debug)]
pub struct PostingListError(pub String);

impl std::fmt::Display for PostingListError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PostingListError {}

impl PostingList for RoaringPostingList {
    type Error = PostingListError;

    fn len(&self) -> usize {
        self.rb.len() as usize
    }
    fn contains(&self, id: usize) -> bool {
        self.rb.contains(id as u32)
    }
    fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.rb.serialize_into(&mut buf).expect("roaring serialize");
        buf
    }
    fn deserialize(bytes: &[u8]) -> std::result::Result<Self, Self::Error> {
        let rb = RoaringBitmap::deserialize_from(&mut &*bytes)
            .map_err(|e| PostingListError(format!("Failed to deserialize RoaringBitmap: {}", e)))?;
        Ok(Self { rb })
    }
    fn empty() -> Self {
        Self {
            rb: RoaringBitmap::new(),
        }
    }
    fn insert(&mut self, id: usize) -> bool {
        let before = self.rb.contains(id as u32);
        self.rb.insert(id as u32);
        !before
    }
    fn remove(&mut self, id: usize) -> bool {
        self.rb.remove(id as u32)
    }
    fn union(&self, other: &Self) -> Self {
        let mut rb = self.rb.clone();
        rb |= &other.rb;
        Self { rb }
    }
    fn intersect(&self, other: &Self) -> Self {
        let mut rb = self.rb.clone();
        rb &= &other.rb;
        Self { rb }
    }

    fn difference(&self, other: &Self) -> Self {
        let mut rb = self.rb.clone();
        rb -= &other.rb;
        Self { rb }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
