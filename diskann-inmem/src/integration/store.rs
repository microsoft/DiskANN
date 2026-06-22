/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::store;

#[derive(Debug)]
pub struct Store {
    store: store::Store,
}

impl Store {
    pub fn acquire(&self) -> Option<Writer<'_>> {
        self.store.acquire().map(Writer::new)
    }

    #[must_use = "result indicates success or failure"]
    pub fn retire(&self, i: usize) -> bool {
        self.store.retire(i).is_ok()
    }

    pub fn reader(&self) -> Option<Reader<'_>> {
        match self.store.reader() {
            Ok(reader) => Some(Reader::new(reader)),
            Err(crate::epoch::Unavailable) => None,
        }
    }
}

pub struct Reader<'a> {
    reader: store::Reader<'a>,
}

impl<'a> Reader<'a> {
    fn new(reader: store::Reader<'a>) -> Self {
        Self { reader }
    }

    pub fn read(&self, i: usize) -> Option<&[u8]> {
        self.reader.read(i)
    }
}

pub struct Writer<'a> {
    slot: store::Slot<'a>,
}

impl<'a> Writer<'a> {
    fn new(slot: store::Slot<'a>) -> Self {
        Self { slot }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.slot.as_mut_slice()
    }

    pub fn slot(&self) -> u32 {
        self.slot.slot()
    }
}
