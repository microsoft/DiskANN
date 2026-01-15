/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::attribute::{Attribute, AttributeValue};
use std::collections::HashMap;
use std::hash::Hash;

///We have the common [`Attribute`] struct that is used both by the clients and
/// internally within filter search to represent attributes. We want this
/// struct to be client-friendly, so we want to support features like adding
/// it to a set and looking up an attribute by only its name. Therefore we
/// implement Eq, Hash and Borrow for Attribute that only use the field name.
/// This in-turn causes an issue if the struct needs to be used in a mapped provider
/// where we want to map the entire attribute into a u32 (both the name
/// and the value). Therefore we create a copy of the Attribute which is the
/// struct below and define hash/eq, e.t.c for it to work with our scenario.
/// The caveat is that the original attribute object cannot be used for lookups
/// which should be ok.
#[derive(Debug, Hash, PartialEq, Eq)]
pub(crate) struct InternalAttribute {
    field_name: String,
    attr_value: AttributeValue,
}

impl InternalAttribute {
    pub(crate) fn new(attr: &Attribute) -> Self {
        Self {
            field_name: attr.field_name().clone(),
            attr_value: attr.value().clone(),
        }
    }

    /// Get the field name
    #[expect(dead_code, reason = "Callers will be added in the next PR")]
    pub(crate) fn field_name(&self) -> &str {
        &self.field_name
    }

    /// Get the attribute value
    #[expect(dead_code, reason = "Callers will be added in the next PR")]
    pub(crate) fn attr_value(&self) -> &AttributeValue {
        &self.attr_value
    }
}

/// Struct which maps a string attribute either in the form "field=value" or "string"
/// into an integral id.
/// This is used to improve the performance of search-time filter matching by replacing
/// string matches with id matches. Of course this works only for a subset of queries that
/// do not use relational operators (>, <, <=, >=), but that subset is large enough for us
/// to add this feature.
pub(crate) struct AttributeEncoder {
    attribute_index: HashMap<InternalAttribute, u64>,
    running_index: u64,
}

impl AttributeEncoder {
    pub fn new() -> Self {
        Self {
            attribute_index: HashMap::new(),
            running_index: 0,
        }
    }

    /// Insert a new attribute in the attribute map and returns its
    /// id. IF the attribute already exists, its existing id is
    /// returned.
    /// Returns: the id of the attribute
    pub fn insert(&mut self, attribute: &Attribute) -> u64 {
        let current_index = self.running_index;
        *self
            .attribute_index
            .entry(InternalAttribute::new(attribute))
            .or_insert_with(|| {
                self.running_index += 1;
                current_index
            })
    }

    /// Return the id for a given attribute. Converts the attribute into
    /// an "internal attribute"  looks it up in the attribute map and
    /// returns the matching id if it iexists.
    /// Returns: Some(id) if the attribute is known, None otherwise.
    pub fn get(&self, attribute: &Attribute) -> Option<u64> {
        //Since this is test code, we are not worrying about performance. But if
        //we need this function in the hot path, we'll have to do some borrow
        //tricks to optimize perf.
        let intr_attr = InternalAttribute::new(attribute);
        if let Some(&id) = self.attribute_index.get(&intr_attr) {
            Some(id)
        } else {
            None
        }
    }

    ///Return the number of entries in the attribute map.
    #[expect(dead_code, reason = "Will be used in the next PR")]
    pub(crate) fn len(&self) -> usize {
        self.attribute_index.len()
    }

    /// Apply a function to each entry in the attribute map
    /// This allows iteration over the internal attribute mappings
    #[expect(dead_code, reason = "Will be used in the next PR")]
    pub(crate) fn for_each<F>(&self, mut func: F)
    where
        F: FnMut(&InternalAttribute, u64),
    {
        for (attr, &id) in &self.attribute_index {
            func(attr, id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute::AttributeValue;

    #[test]
    fn test_new_attribute_map_is_empty() {
        let attribute_map: AttributeEncoder = AttributeEncoder::new();
        assert_eq!(attribute_map.attribute_index.len(), 0);
        assert_eq!(attribute_map.running_index, 0);
    }

    #[test]
    fn test_insert_new_attribute() {
        let mut attribute_map: AttributeEncoder = AttributeEncoder::new();
        let attribute = Attribute::from_value(
            "test_field",
            AttributeValue::String("test_value".to_string()),
        );
        let id = attribute_map.insert(&attribute);

        assert_eq!(id, 0); // First attribute should get ID 0
        assert_eq!(attribute_map.attribute_index.len(), 1);
        assert_eq!(attribute_map.running_index, 1);
    }

    #[test]
    fn test_insert_existing_attribute() {
        let mut attribute_map: AttributeEncoder = AttributeEncoder::new();
        let attribute = Attribute::from_value(
            "test_field",
            AttributeValue::String("test_value".to_string()),
        );

        let id1 = attribute_map.insert(&attribute);
        let id2 = attribute_map.insert(&attribute); // Insert same attribute again

        assert_eq!(id1, id2); // Should return the same ID
        assert_eq!(attribute_map.attribute_index.len(), 1); // Map should still have only one entry
        assert_eq!(attribute_map.running_index, 1); // Running index should only increment once
    }

    #[test]
    fn test_multiple_inserts() {
        let mut attribute_map: AttributeEncoder = AttributeEncoder::new();

        let attribute1 =
            Attribute::from_value("field1", AttributeValue::String("value1".to_string()));
        let attribute2 = Attribute::from_value("field2", AttributeValue::Integer(42));
        let attribute3 = Attribute::from_value("field3", AttributeValue::Bool(true));

        let id1 = attribute_map.insert(&attribute1);
        let id2 = attribute_map.insert(&attribute2);
        let id3 = attribute_map.insert(&attribute3);

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);
        assert_eq!(attribute_map.attribute_index.len(), 3);
        assert_eq!(attribute_map.running_index, 3);
    }

    #[test]
    fn test_get_existing_attribute() {
        let mut attribute_map: AttributeEncoder = AttributeEncoder::new();
        let attribute = Attribute::from_value(
            "test_field",
            AttributeValue::String("test_value".to_string()),
        );

        let inserted_id = attribute_map.insert(&attribute);
        let retrieved_id = attribute_map.get(&attribute).unwrap();

        assert_eq!(inserted_id, retrieved_id);
    }

    #[test]
    fn test_get_nonexistent_attribute() {
        let attribute_map: AttributeEncoder = AttributeEncoder::new();
        let attribute =
            Attribute::from_value("nonexistent", AttributeValue::String("value".to_string()));

        let result = attribute_map.get(&attribute);
        assert!(result.is_none());
    }
}
