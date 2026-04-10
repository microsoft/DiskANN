/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! `SaveWith` / `LoadWith` implementations for [`RoaringAttributeStore`] and
//! [`DocumentProvider`].
//!
//! # File format
//!
//! Label data is persisted to `{prefix}.labels.bin`.  The format uses
//! little-endian byte order throughout.
//!
//! ```text
//! Header (17 bytes)
//!   [u64] num_attribute_entries
//!   [u64] forward_index_offset   (byte offset from file start to Section 2)
//!   [u8]  vector_id_type_tag     (0 = u32, 1 = u64)
//!
//! Section 1 – Attribute Dictionary  (repeated num_attribute_entries times)
//!   [u64] attribute_id
//!   [u32] field_name_byte_len
//!   [u8…] UTF-8 field name
//!   [u8]  type_tag  (0=Bool, 1=Integer, 2=Real, 3=String, 4=Empty)
//!   value bytes:
//!     Bool:    1 byte  (0=false, 1=true)
//!     Integer: 8 bytes (i64 little-endian)
//!     Real:    8 bytes (f64 little-endian)
//!     String: [u32 byte_len][u8…  UTF-8]
//!     Empty:   0 bytes
//!
//! Section 2 – Forward Index
//!   [u64] num_nodes_with_labels
//!   (repeated num_nodes_with_labels times)
//!     [N bytes] node_internal_id   (4 bytes for u32, 8 bytes for u64)
//!     [u32]     num_attribute_ids
//!     [u64…]    attribute IDs (in RoaringTreemap iteration order, ascending)
//! ```
//!
//! The inverted index is **not** persisted; it is rebuilt from the forward
//! index during [`LoadWith::load_with`].

use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::mem;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use diskann::{utils::VectorId, ANNError, ANNErrorKind};
use diskann_providers::storage::{
    AsyncIndexMetadata, AsyncQuantLoadContext, LoadWith, SaveWith, StorageReadProvider,
    StorageWriteProvider,
};
use diskann_utils::future::AsyncFriendly;
use roaring::RoaringTreemap;

use crate::set::traits::SetProvider;
use crate::{
    attribute::{Attribute, AttributeValue},
    encoded_attribute_provider::{
        attribute_encoder::InternalAttribute, document_provider::DocumentProvider,
        roaring_attribute_store::RoaringAttributeStore,
    },
    traits::attribute_store::AttributeStore,
};

// ────────────────────────────────────────────────────────────────────────────
// Internal helper: derive the label file path from the auxiliary context.
// ────────────────────────────────────────────────────────────────────────────

/// Trait implemented for the auxiliary types recognised by this module.
/// Returns the path at which label data should be written / read.
pub(crate) trait LabelFilePath {
    fn label_file_path(&self) -> String;
}

impl LabelFilePath for String {
    fn label_file_path(&self) -> String {
        format!("{}.labels.bin", self)
    }
}

impl LabelFilePath for AsyncIndexMetadata {
    fn label_file_path(&self) -> String {
        format!("{}.labels.bin", self.prefix())
    }
}

impl LabelFilePath for (u32, AsyncIndexMetadata) {
    fn label_file_path(&self) -> String {
        format!("{}.labels.bin", self.1.prefix())
    }
}

impl LabelFilePath for AsyncQuantLoadContext {
    fn label_file_path(&self) -> String {
        format!("{}.labels.bin", self.metadata.prefix())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Type-tag helpers
// ────────────────────────────────────────────────────────────────────────────

/// Wire tag for the vector-ID integer width stored in the label file header.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
enum VectorIdTypeTag {
    U32 = 0,
    U64 = 1,
}

impl VectorIdTypeTag {
    fn from_u8(byte: u8) -> Result<Self, ANNError> {
        match byte {
            0 => Ok(Self::U32),
            1 => Ok(Self::U64),
            tag => Err(ANNError::message(
                ANNErrorKind::IndexError,
                format!("unknown vector_id_type_tag: {tag}"),
            )),
        }
    }
}

fn vector_id_type_tag<IT: VectorId>() -> VectorIdTypeTag {
    match mem::size_of::<IT>() {
        4 => VectorIdTypeTag::U32,
        8 => VectorIdTypeTag::U64,
        _ => panic!("unsupported VectorId width: {}", mem::size_of::<IT>()),
    }
}

/// Wire tag for the attribute value type stored in Section 1 entries.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
enum AttrTypeTag {
    Bool = 0,
    Integer = 1,
    Real = 2,
    String = 3,
    Empty = 4,
}

impl AttrTypeTag {
    fn from_u8(byte: u8) -> Result<Self, ANNError> {
        match byte {
            0 => Ok(Self::Bool),
            1 => Ok(Self::Integer),
            2 => Ok(Self::Real),
            3 => Ok(Self::String),
            4 => Ok(Self::Empty),
            tag => Err(ANNError::message(
                ANNErrorKind::IndexError,
                format!("unknown attribute type tag: {tag}"),
            )),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// RoaringAttributeStore – SaveWith
// ────────────────────────────────────────────────────────────────────────────

impl<IT, T> SaveWith<T> for RoaringAttributeStore<IT>
where
    IT: VectorId + AsyncFriendly,
    T: LabelFilePath + Send + Sync,
{
    type Ok = ();
    type Error = ANNError;

    async fn save_with<P>(&self, provider: &P, auxiliary: &T) -> Result<(), ANNError>
    where
        P: StorageWriteProvider,
    {
        let path = auxiliary.label_file_path();

        // Bind the Arcs to named variables so the guards don't hold a reference
        // to a temporary that is immediately dropped.
        let attr_map_arc = self.attribute_map();
        let attr_map_guard = attr_map_arc
            .read()
            .map_err(|_| lock_poison("attribute_map (read)"))?;
        let index_arc = self.get_index();
        let index_guard = index_arc.read().map_err(|_| lock_poison("index (read)"))?;

        let num_attribute_entries = attr_map_guard.len() as u64;

        let file = provider
            .create_for_write(&path)
            .map_err(|e| io_error(e, "create label file"))?;
        let mut w = BufWriter::new(file);

        // ------- Header -------
        // Byte 0–7:  num_attribute_entries (u64)
        // Byte 8–15: forward_index_offset  (u64, patched after section 1)
        // Byte 16:   vector_id_type_tag    (u8)
        w.write_u64::<LittleEndian>(num_attribute_entries)
            .map_err(|e| io_error(e, "write num_attribute_entries"))?;
        w.write_u64::<LittleEndian>(0_u64) // placeholder for forward_index_offset
            .map_err(|e| io_error(e, "write forward_index_offset placeholder"))?;
        let type_tag = vector_id_type_tag::<IT>();
        w.write_u8(type_tag as u8)
            .map_err(|e| io_error(e, "write vector_id_type_tag"))?;

        // ------- Section 1: Attribute Dictionary -------
        attr_map_guard.for_each(|attr, id| write_dict_entry(&mut w, id, attr))?;

        // Flush so the inner writer's position reflects all bytes written so far.
        w.flush()
            .map_err(|e| io_error(e, "flush before forward index"))?;

        // ------- Patch forward_index_offset and write Section 2 -------
        {
            let mut inner = w
                .into_inner()
                .map_err(|e| io_error(e.into_error(), "flush BufWriter into inner"))?;

            // Record where section 2 will start (current end of section 1).
            let fwd_offset = inner
                .stream_position()
                .map_err(|e| io_error(e, "seek to current position"))?;

            // Patch the placeholder at byte offset 8.
            inner
                .seek(SeekFrom::Start(8))
                .map_err(|e| io_error(e, "seek to forward_index_offset field"))?;
            inner
                .write_u64::<LittleEndian>(fwd_offset)
                .map_err(|e| io_error(e, "write forward_index_offset"))?;

            // Seek back to end to append section 2.
            inner
                .seek(SeekFrom::Start(fwd_offset))
                .map_err(|e| io_error(e, "seek back to section 2"))?;

            let mut w2 = BufWriter::new(inner);

            // Section 2 header: number of nodes with labels.
            let num_nodes = index_guard.count().map_err(|e| {
                ANNError::new(ANNErrorKind::IndexError, e).context("count forward index")
            })?;
            w2.write_u64::<LittleEndian>(num_nodes as u64)
                .map_err(|e| io_error(e, "write num_nodes_with_labels"))?;

            index_guard.for_each(|node_id, set| -> Result<(), ANNError> {
                write_forward_entry::<IT, _>(&mut w2, type_tag, *node_id, set)
            })?;

            w2.flush().map_err(|e| io_error(e, "flush section 2"))?;
        }

        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// RoaringAttributeStore – LoadWith
// ────────────────────────────────────────────────────────────────────────────

impl<IT, T> LoadWith<T> for RoaringAttributeStore<IT>
where
    IT: VectorId + AsyncFriendly,
    T: LabelFilePath + Send + Sync,
{
    type Error = ANNError;

    async fn load_with<P>(provider: &P, auxiliary: &T) -> Result<Self, ANNError>
    where
        P: StorageReadProvider,
    {
        let path = auxiliary.label_file_path();
        let file = provider
            .open_reader(&path)
            .map_err(|e| io_error(e, "open label file"))?;
        let mut r = BufReader::new(file);

        // ------- Header -------
        let num_attribute_entries = r
            .read_u64::<LittleEndian>()
            .map_err(|e| io_error(e, "read num_attribute_entries"))?;
        let forward_index_offset = r
            .read_u64::<LittleEndian>()
            .map_err(|e| io_error(e, "read forward_index_offset"))?;
        let type_tag_byte = r
            .read_u8()
            .map_err(|e| io_error(e, "read vector_id_type_tag"))?;
        let type_tag = VectorIdTypeTag::from_u8(type_tag_byte)?;

        // Validate that the stored type tag matches the current IT.
        let expected_tag = vector_id_type_tag::<IT>();
        if type_tag != expected_tag {
            return Err(ANNError::message(
                ANNErrorKind::IndexError,
                format!(
                    "Label file type tag mismatch: file has {:?}, expected {:?}",
                    type_tag, expected_tag
                ),
            ));
        }

        // ------- Section 1: Attribute Dictionary -------
        let store = RoaringAttributeStore::<IT>::new();
        {
            let attr_map_arc = store.attribute_map();
            let mut attr_map = attr_map_arc
                .write()
                .map_err(|_| lock_poison("attribute_map (write)"))?;
            for _ in 0..num_attribute_entries {
                let (attr_id, attr) = read_dict_entry(&mut r)?;
                // Re-insert with the exact same id that was persisted.
                attr_map.insert_with_id(&attr, attr_id);
            }
        }

        // ------- Seek to Section 2 -------
        r.seek(SeekFrom::Start(forward_index_offset))
            .map_err(|e| io_error(e, "seek to forward index"))?;

        // ------- Section 2: Forward Index -------
        let num_nodes = r
            .read_u64::<LittleEndian>()
            .map_err(|e| io_error(e, "read num_nodes_with_labels"))?;

        {
            let index_arc = store.get_index();
            let inv_index_arc = store.get_inv_index();
            let mut index = index_arc
                .write()
                .map_err(|_| lock_poison("index (write)"))?;
            let mut inv_index = inv_index_arc
                .write()
                .map_err(|_| lock_poison("inv_index (write)"))?;

            for _ in 0..num_nodes {
                let (node_id, attr_ids) = read_forward_entry::<IT, _>(&mut r, type_tag)?;
                let node_u64: u64 = node_id.into();
                for &attr_id in &attr_ids {
                    index.insert(&node_id, &attr_id).map_err(|e| {
                        ANNError::new(ANNErrorKind::IndexError, e).context("rebuild forward index")
                    })?;
                    inv_index.insert(&attr_id, &node_u64).map_err(|e| {
                        ANNError::new(ANNErrorKind::IndexError, e).context("rebuild inverted index")
                    })?;
                }
            }
        }

        Ok(store)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// DocumentProvider – SaveWith
// ────────────────────────────────────────────────────────────────────────────

impl<DP, AS, T> SaveWith<T> for DocumentProvider<DP, AS>
where
    DP: diskann::provider::DataProvider + SaveWith<T, Error = ANNError>,
    AS: AttributeStore<DP::InternalId> + SaveWith<T, Ok = (), Error = ANNError> + AsyncFriendly,
    T: Send + Sync,
{
    type Ok = ();
    type Error = ANNError;

    async fn save_with<P>(&self, provider: &P, auxiliary: &T) -> Result<(), ANNError>
    where
        P: StorageWriteProvider,
    {
        // Delegate to inner provider.
        self.inner_provider().save_with(provider, auxiliary).await?;

        // Persist the attribute store.
        self.attribute_store()
            .save_with(provider, auxiliary)
            .await?;

        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// DocumentProvider – LoadWith
// ────────────────────────────────────────────────────────────────────────────

impl<DP, AS, T> LoadWith<T> for DocumentProvider<DP, AS>
where
    DP: diskann::provider::DataProvider + LoadWith<T, Error = ANNError>,
    AS: AttributeStore<DP::InternalId> + LoadWith<T, Error = ANNError> + AsyncFriendly,
    T: Send + Sync,
{
    type Error = ANNError;

    async fn load_with<P>(provider: &P, auxiliary: &T) -> Result<Self, ANNError>
    where
        P: StorageReadProvider,
    {
        // Load the inner provider.
        let inner_provider = DP::load_with(provider, auxiliary).await?;

        // Load the attribute store.
        let attribute_store = AS::load_with(provider, auxiliary).await?;

        Ok(Self::new(inner_provider, attribute_store))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Private helpers
// ────────────────────────────────────────────────────────────────────────────

fn lock_poison(field: &str) -> ANNError {
    ANNError::message(
        ANNErrorKind::LockPoisonError,
        format!("poisoned lock on {field}"),
    )
}

fn io_error(e: std::io::Error, context: &str) -> ANNError {
    ANNError::message(
        ANNErrorKind::IndexError,
        format!("IO error while {context}: {e}"),
    )
}

fn write_dict_entry<W: Write>(
    w: &mut W,
    attr_id: u64,
    attr: &InternalAttribute,
) -> Result<(), ANNError> {
    w.write_u64::<LittleEndian>(attr_id)
        .map_err(|e| io_error(e, "write attribute_id"))?;

    let field = attr.field_name();
    let field_bytes = field.as_bytes();
    w.write_u32::<LittleEndian>(field_bytes.len() as u32)
        .map_err(|e| io_error(e, "write field_name_byte_len"))?;
    w.write_all(field_bytes)
        .map_err(|e| io_error(e, "write field_name bytes"))?;

    match attr.attr_value() {
        AttributeValue::Bool(b) => {
            w.write_u8(AttrTypeTag::Bool as u8)
                .map_err(|e| io_error(e, "write Bool tag"))?;
            w.write_u8(if *b { 1 } else { 0 })
                .map_err(|e| io_error(e, "write Bool value"))?;
        }
        AttributeValue::Integer(i) => {
            w.write_u8(AttrTypeTag::Integer as u8)
                .map_err(|e| io_error(e, "write Integer tag"))?;
            w.write_i64::<LittleEndian>(*i)
                .map_err(|e| io_error(e, "write Integer value"))?;
        }
        AttributeValue::Real(f) => {
            w.write_u8(AttrTypeTag::Real as u8)
                .map_err(|e| io_error(e, "write Real tag"))?;
            w.write_f64::<LittleEndian>(*f)
                .map_err(|e| io_error(e, "write Real value"))?;
        }
        AttributeValue::String(s) => {
            w.write_u8(AttrTypeTag::String as u8)
                .map_err(|e| io_error(e, "write String tag"))?;
            let sb = s.as_bytes();
            w.write_u32::<LittleEndian>(sb.len() as u32)
                .map_err(|e| io_error(e, "write String byte_len"))?;
            w.write_all(sb)
                .map_err(|e| io_error(e, "write String bytes"))?;
        }
        AttributeValue::Empty => {
            w.write_u8(AttrTypeTag::Empty as u8)
                .map_err(|e| io_error(e, "write Empty tag"))?;
        }
    }
    Ok(())
}

fn read_dict_entry<R: Read>(r: &mut R) -> Result<(u64, Attribute), ANNError> {
    let attr_id = r
        .read_u64::<LittleEndian>()
        .map_err(|e| io_error(e, "read attribute_id"))?;

    let field_len = r
        .read_u32::<LittleEndian>()
        .map_err(|e| io_error(e, "read field_name_byte_len"))? as usize;
    let mut field_bytes = vec![0u8; field_len];
    r.read_exact(&mut field_bytes)
        .map_err(|e| io_error(e, "read field_name bytes"))?;
    let field_name = String::from_utf8(field_bytes).map_err(|e| {
        ANNError::message(
            ANNErrorKind::IndexError,
            format!("invalid UTF-8 in field name: {e}"),
        )
    })?;

    let type_tag = AttrTypeTag::from_u8(r.read_u8().map_err(|e| io_error(e, "read type_tag"))?)?;

    let value = match type_tag {
        AttrTypeTag::Bool => {
            let b = r.read_u8().map_err(|e| io_error(e, "read Bool value"))?;
            AttributeValue::Bool(b != 0)
        }
        AttrTypeTag::Integer => {
            let i = r
                .read_i64::<LittleEndian>()
                .map_err(|e| io_error(e, "read Integer value"))?;
            AttributeValue::Integer(i)
        }
        AttrTypeTag::Real => {
            let f = r
                .read_f64::<LittleEndian>()
                .map_err(|e| io_error(e, "read Real value"))?;
            AttributeValue::Real(f)
        }
        AttrTypeTag::String => {
            let slen = r
                .read_u32::<LittleEndian>()
                .map_err(|e| io_error(e, "read String byte_len"))? as usize;
            let mut sb = vec![0u8; slen];
            r.read_exact(&mut sb)
                .map_err(|e| io_error(e, "read String bytes"))?;
            let s = String::from_utf8(sb).map_err(|e| {
                ANNError::message(
                    ANNErrorKind::IndexError,
                    format!("invalid UTF-8 in String value: {e}"),
                )
            })?;
            AttributeValue::String(s)
        }
        AttrTypeTag::Empty => AttributeValue::Empty,
    };

    Ok((attr_id, Attribute::from_value(field_name, value)))
}

fn write_forward_entry<IT, W>(
    w: &mut W,
    type_tag: VectorIdTypeTag,
    node_id: IT,
    set: &RoaringTreemap,
) -> Result<(), ANNError>
where
    IT: VectorId,
    W: Write,
{
    match type_tag {
        VectorIdTypeTag::U32 => {
            let id32: u32 = node_id.into() as u32;
            w.write_u32::<LittleEndian>(id32)
                .map_err(|e| io_error(e, "write node_id u32"))?;
        }
        VectorIdTypeTag::U64 => {
            let id64: u64 = node_id.into();
            w.write_u64::<LittleEndian>(id64)
                .map_err(|e| io_error(e, "write node_id u64"))?;
        }
    }

    w.write_u32::<LittleEndian>(set.len() as u32)
        .map_err(|e| io_error(e, "write num_attribute_ids"))?;
    for id in set.iter() {
        w.write_u64::<LittleEndian>(id)
            .map_err(|e| io_error(e, "write attribute_id in forward entry"))?;
    }
    Ok(())
}

fn read_forward_entry<IT, R>(
    r: &mut R,
    type_tag: VectorIdTypeTag,
) -> Result<(IT, Vec<u64>), ANNError>
where
    IT: VectorId,
    R: Read,
{
    let node_id: IT = match type_tag {
        VectorIdTypeTag::U32 => {
            let raw = r
                .read_u32::<LittleEndian>()
                .map_err(|e| io_error(e, "read node_id u32"))?;
            IT::from_u32(raw).ok_or_else(|| {
                ANNError::message(ANNErrorKind::IndexError, "node_id u32 conversion failed")
            })?
        }
        VectorIdTypeTag::U64 => {
            let raw = r
                .read_u64::<LittleEndian>()
                .map_err(|e| io_error(e, "read node_id u64"))?;
            IT::from_u64(raw).ok_or_else(|| {
                ANNError::message(ANNErrorKind::IndexError, "node_id u64 conversion failed")
            })?
        }
    };

    let num_attr = r
        .read_u32::<LittleEndian>()
        .map_err(|e| io_error(e, "read num_attribute_ids"))? as usize;
    let mut attr_ids = Vec::with_capacity(num_attr);
    for _ in 0..num_attr {
        let id = r
            .read_u64::<LittleEndian>()
            .map_err(|e| io_error(e, "read attribute_id in forward entry"))?;
        attr_ids.push(id);
    }
    Ok((node_id, attr_ids))
}
