/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Pipelined IO reader using io_uring with non-blocking submit/poll semantics.

use std::{
    fs::OpenOptions,
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
};

use diskann::{ANNError, ANNResult};
use diskann_providers::common::AlignedBoxWithSlice;
use io_uring::IoUring;

/// Maximum number of concurrent IO operations supported by the ring.
pub const MAX_IO_CONCURRENCY: usize = 128;

/// Configuration for io_uring-based pipelined reader.
#[derive(Debug, Clone, Default)]
pub struct PipelinedReaderConfig {
    /// Enable kernel-side SQ polling. If `Some(idle_ms)`, a kernel thread polls
    /// the submission queue, eliminating the syscall per submit. After `idle_ms`
    /// milliseconds of inactivity the kernel thread sleeps (resumed automatically
    /// on next `submit()`). Requires Linux kernel >= 5.11 (>= 5.13 unprivileged).
    pub sqpoll_idle_ms: Option<u32>,
}

/// A pipelined IO reader that wraps `io_uring` for non-blocking submit/poll.
///
/// Unlike `LinuxAlignedFileReader` which uses `submit_and_wait` (blocking),
/// this reader submits reads and polls completions independently, enabling
/// IO/compute overlap within a single search query.
pub struct PipelinedReader {
    ring: IoUring,
    /// Pre-allocated sector-aligned read buffers, one per slot.
    slot_bufs: AlignedBoxWithSlice<u8>,
    /// Size of each slot buffer in bytes.
    slot_size: usize,
    /// Maximum number of slots available.
    max_slots: usize,
    /// Number of currently in-flight (submitted but not completed) reads.
    in_flight: usize,
    /// Keep the file handle alive for the lifetime of the reader.
    _file: std::fs::File,
}

impl PipelinedReader {
    /// Create a new pipelined reader.
    ///
    /// # Arguments
    /// * `file_path` - Path to the disk index file.
    /// * `max_slots` - Number of buffer slots (must be <= MAX_IO_CONCURRENCY).
    /// * `slot_size` - Size of each buffer slot in bytes (should be sector-aligned).
    /// * `alignment` - Memory alignment for the buffer (typically 4096 for O_DIRECT).
    pub fn new(
        file_path: &str,
        max_slots: usize,
        slot_size: usize,
        alignment: usize,
        config: &PipelinedReaderConfig,
    ) -> ANNResult<Self> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(file_path)
            .map_err(ANNError::log_io_error)?;

        let entries = max_slots.min(MAX_IO_CONCURRENCY) as u32;
        let ring = if config.sqpoll_idle_ms.is_some() {
            let mut builder = IoUring::builder();
            if let Some(idle_ms) = config.sqpoll_idle_ms {
                builder.setup_sqpoll(idle_ms);
            }
            builder.build(entries)?
        } else {
            IoUring::new(entries)?
        };
        let fd = file.as_raw_fd();
        ring.submitter().register_files(std::slice::from_ref(&fd))?;

        let slot_bufs = AlignedBoxWithSlice::new(max_slots * slot_size, alignment)?;

        Ok(Self {
            ring,
            slot_bufs,
            slot_size,
            max_slots,
            in_flight: 0,
            _file: file,
        })
    }

    /// Submit an asynchronous read into the buffer at `slot_id`.
    ///
    /// The read will fetch `slot_size` bytes from `sector_offset` (in bytes) into
    /// the pre-allocated buffer for the given slot. The `slot_id` is stored as
    /// `user_data` in the CQE for later retrieval.
    pub fn submit_read(&mut self, sector_offset: u64, slot_id: usize) -> ANNResult<()> {
        assert!(slot_id < self.max_slots, "slot_id out of range");

        let buf_start = slot_id * self.slot_size;
        let buf_ptr = self.slot_bufs[buf_start..buf_start + self.slot_size].as_mut_ptr();

        let read_op = io_uring::opcode::Read::new(
            io_uring::types::Fixed(0),
            buf_ptr,
            self.slot_size as u32,
        )
        .offset(sector_offset)
        .build()
        .user_data(slot_id as u64);

        // SAFETY: The buffer at slot_id is pre-allocated and will remain valid
        // for the duration of the IO operation. Each slot is used exclusively
        // (caller must not reuse a slot while it is in-flight).
        unsafe {
            self.ring
                .submission()
                .push(&read_op)
                .map_err(ANNError::log_push_error)?;
        }

        self.ring.submit()?;
        self.in_flight += 1;
        Ok(())
    }

    /// Poll for completed IO operations (non-blocking).
    ///
    /// Drains already-completed CQEs from the io_uring completion queue.
    pub fn poll_completions(&mut self) -> ANNResult<Vec<usize>> {
        self.drain_cqes()
    }

    /// Block until at least one IO completes, then drain all available CQEs.
    ///
    /// Use this when [`poll_completions`] returned an empty vec but there are
    /// in-flight reads — avoids busy-spinning while waiting for the kernel.
    pub fn wait_completions(&mut self) -> ANNResult<Vec<usize>> {
        if self.in_flight == 0 {
            return Ok(Vec::new());
        }
        self.ring.submit_and_wait(1)?;
        self.drain_cqes()
    }

    /// Drain all available CQEs from the completion queue.
    fn drain_cqes(&mut self) -> ANNResult<Vec<usize>> {
        let mut completed = Vec::new();
        for cqe in self.ring.completion() {
            if cqe.result() < 0 {
                self.in_flight = self.in_flight.saturating_sub(1);
                return Err(ANNError::log_io_error(std::io::Error::from_raw_os_error(
                    -cqe.result(),
                )));
            }
            let slot_id = cqe.user_data() as usize;
            completed.push(slot_id);
            self.in_flight = self.in_flight.saturating_sub(1);
        }
        Ok(completed)
    }

    /// Returns the read buffer for a completed slot.
    pub fn get_slot_buf(&self, slot_id: usize) -> &[u8] {
        let start = slot_id * self.slot_size;
        &self.slot_bufs[start..start + self.slot_size]
    }

    /// Reset the reader for reuse: clear in-flight count and drain remaining CQEs.
    pub fn reset(&mut self) {
        self.in_flight = 0;
        // Drain any remaining completions from the ring.
        for _cqe in self.ring.completion() {}
    }

    /// Returns the number of submitted but not yet completed reads.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight
    }

    /// Returns the slot size in bytes.
    pub fn slot_size(&self) -> usize {
        self.slot_size
    }

    /// Returns the maximum number of buffer slots.
    pub fn max_slots(&self) -> usize {
        self.max_slots
    }

    /// Drain all in-flight IOs, blocking until they complete.
    /// Must be called before freeing the slot buffers.
    fn drain_all(&mut self) {
        while self.in_flight > 0 {
            let _ = self.ring.submit_and_wait(1);
            for cqe in self.ring.completion() {
                let _ = cqe;
                self.in_flight = self.in_flight.saturating_sub(1);
            }
        }
    }
}

impl Drop for PipelinedReader {
    fn drop(&mut self) {
        // Must wait for all in-flight kernel IOs to complete before freeing
        // the slot buffers — otherwise the kernel may DMA into freed memory.
        self.drain_all();
    }
}
