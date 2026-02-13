/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Pipelined IO reader using io_uring with non-blocking submit/poll semantics.
//!
//! # Safety model
//!
//! The kernel writes to slot buffers via DMA, which is invisible to the Rust
//! compiler. To avoid aliasing UB we **never** form `&[u8]` or `&mut [u8]`
//! references to the backing allocation while any IO is in-flight. Instead we:
//!
//! 1. Obtain the base raw pointer (`*mut u8`) **once** at construction — before
//!    any IO is submitted — and store it for later use.
//! 2. Pass raw pointers to io_uring for kernel DMA targets.
//! 3. Only materialise `&[u8]` slices via [`std::slice::from_raw_parts`] for
//!    slots whose state is [`SlotState::Completed`] (kernel has finished writing).
//!
//! Slot lifecycle: `Free → InFlight → Completed → Free`.
//!
//! [`PipelinedReader`] owns the free-list and state machine so callers never
//! need `unsafe` for normal operation.

use std::{
    collections::VecDeque,
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

/// State of each buffer slot in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    /// Slot is available for a new IO submission.
    Free,
    /// SQE has been pushed (and possibly submitted). Kernel may be DMA-ing.
    InFlight,
    /// CQE has been reaped — data is ready. Safe to create `&[u8]`.
    Completed,
}

/// A pipelined IO reader that wraps `io_uring` for non-blocking submit/poll.
///
/// Unlike `LinuxAlignedFileReader` which uses `submit_and_wait` (blocking),
/// this reader submits reads and polls completions independently, enabling
/// IO/compute overlap within a single search query.
///
/// The reader owns both the ring buffer allocation and the slot state machine.
/// Callers interact through a safe API:
///
/// 1. [`enqueue_read`](Self::enqueue_read) — push an SQE, returns `slot_id`.
/// 2. [`flush`](Self::flush) — submit all enqueued SQEs to the kernel (one syscall).
/// 3. [`poll_completions`](Self::poll_completions) /
///    [`wait_completions`](Self::wait_completions) — drain CQEs.
/// 4. [`get_slot_buf`](Self::get_slot_buf) — borrow data for a `Completed` slot.
/// 5. [`release_slot`](Self::release_slot) — return a `Completed` slot to `Free`.
pub struct PipelinedReader {
    ring: IoUring,
    /// Owns the aligned allocation. **Must not be dereferenced** while any IO is
    /// in-flight — see the module-level safety discussion.
    _slot_bufs: AlignedBoxWithSlice<u8>,
    /// Raw pointer to the start of the buffer, obtained once at construction.
    /// All subsequent slot access goes through pointer arithmetic on this base.
    buf_base: *mut u8,
    /// Size of each slot buffer in bytes.
    slot_size: usize,
    /// Maximum number of slots available.
    max_slots: usize,
    /// Per-slot state.
    slot_states: Vec<SlotState>,
    /// FIFO free-list for O(1) slot allocation.
    free_slots: VecDeque<usize>,
    /// Number of slots whose SQEs have been submitted to the kernel (InFlight).
    in_flight: usize,
    /// Keep the file handle alive for the lifetime of the reader.
    _file: std::fs::File,
}

// SAFETY: The raw pointer `buf_base` is derived from an owned allocation
// (`_slot_bufs`) and is never shared — all mutable access requires `&mut self`.
// The io_uring ring and file descriptor are kernel-side resources with no
// thread-affinity. Moving the reader between threads is safe.
unsafe impl Send for PipelinedReader {}
// SAFETY: `&self` methods only access completed slot data (kernel has finished
// writing). All mutation requires `&mut self`.
unsafe impl Sync for PipelinedReader {}

impl PipelinedReader {
    /// Create a new pipelined reader.
    ///
    /// # Arguments
    /// * `file_path` - Path to the disk index file.
    /// * `max_slots` - Number of buffer slots (clamped to [`MAX_IO_CONCURRENCY`]).
    /// * `slot_size` - Size of each buffer slot in bytes (should be sector-aligned).
    /// * `alignment` - Memory alignment for the buffer (typically 4096 for O_DIRECT).
    /// * `config`    - Optional io_uring tuning (e.g. SQPOLL).
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

        let max_slots = max_slots.min(MAX_IO_CONCURRENCY);
        let entries = max_slots as u32;
        let ring = if let Some(idle_ms) = config.sqpoll_idle_ms {
            let mut builder = IoUring::builder();
            builder.setup_sqpoll(idle_ms);
            builder.build(entries)?
        } else {
            IoUring::new(entries)?
        };
        let fd = file.as_raw_fd();
        ring.submitter().register_files(std::slice::from_ref(&fd))?;

        let mut slot_bufs = AlignedBoxWithSlice::new(max_slots * slot_size, alignment)?;

        // SAFETY: No IOs are in-flight yet, so creating a `&mut [u8]` is sound.
        // We extract the raw pointer here and never form a reference again.
        let buf_base: *mut u8 = slot_bufs.as_mut_slice().as_mut_ptr();

        Ok(Self {
            ring,
            _slot_bufs: slot_bufs,
            buf_base,
            slot_size,
            max_slots,
            slot_states: vec![SlotState::Free; max_slots],
            free_slots: (0..max_slots).collect(),
            in_flight: 0,
            _file: file,
        })
    }

    // ------------------------------------------------------------------
    // Submission
    // ------------------------------------------------------------------

    /// Enqueue an asynchronous read for `sector_offset` into a newly-allocated
    /// buffer slot. Returns the `slot_id` on success.
    ///
    /// The SQE is pushed to the submission queue but **not submitted** to the
    /// kernel. Call [`flush`](Self::flush) after enqueuing a batch to submit
    /// them all in a single syscall.
    ///
    /// Returns an error if no free slots are available.
    pub fn enqueue_read(&mut self, sector_offset: u64) -> ANNResult<usize> {
        let slot_id = self.free_slots.pop_front().ok_or_else(|| {
            ANNError::log_index_error(format_args!(
                "PipelinedReader: no free slots (max_slots={})",
                self.max_slots
            ))
        })?;
        debug_assert_eq!(self.slot_states[slot_id], SlotState::Free);

        // Raw pointer arithmetic — no reference to the backing buffer.
        let buf_ptr = unsafe { self.buf_base.add(slot_id * self.slot_size) };

        let read_op =
            io_uring::opcode::Read::new(io_uring::types::Fixed(0), buf_ptr, self.slot_size as u32)
                .offset(sector_offset)
                .build()
                .user_data(slot_id as u64);

        // SAFETY: `buf_ptr` points into a pre-allocated, aligned region that
        // outlives the reader. The slot is being transitioned to InFlight so no
        // other code will access this memory region.
        let push_result = unsafe { self.ring.submission().push(&read_op) };
        if let Err(e) = push_result {
            // SQE queue full — return slot to free-list.
            self.free_slots.push_back(slot_id);
            return Err(ANNError::log_push_error(e));
        }

        self.slot_states[slot_id] = SlotState::InFlight;
        self.in_flight += 1;
        Ok(slot_id)
    }

    /// Submit all enqueued SQEs to the kernel in a single syscall.
    ///
    /// Retries automatically on `EINTR`. On fatal errors the enqueued slots
    /// remain `InFlight` and will be drained on [`Drop`].
    pub fn flush(&mut self) -> ANNResult<()> {
        loop {
            match self.ring.submit() {
                Ok(_) => return Ok(()),
                Err(ref e) if e.raw_os_error() == Some(libc::EINTR) => continue,
                Err(e) => return Err(ANNError::log_io_error(e)),
            }
        }
    }

    // ------------------------------------------------------------------
    // Completion
    // ------------------------------------------------------------------

    /// Poll for completed IO operations (non-blocking).
    ///
    /// Appends completed `slot_id`s to `completed`. Slots transition from
    /// `InFlight` → `Completed`. The caller must eventually call
    /// [`release_slot`](Self::release_slot) for each returned slot.
    ///
    /// On IO errors or short reads the affected slot is freed automatically and
    /// an error is returned. Successfully completed slots in `completed` are
    /// still valid and should be processed first.
    pub fn poll_completions(&mut self, completed: &mut Vec<usize>) -> ANNResult<()> {
        self.drain_cqes(completed)
    }

    /// Block until at least one IO completes, then drain all available CQEs.
    ///
    /// Same contract as [`poll_completions`](Self::poll_completions).
    pub fn wait_completions(&mut self, completed: &mut Vec<usize>) -> ANNResult<()> {
        if self.in_flight == 0 {
            completed.clear();
            return Ok(());
        }
        // submit_and_wait also flushes any un-submitted SQEs.
        loop {
            match self.ring.submit_and_wait(1) {
                Ok(_) => break,
                Err(ref e) if e.raw_os_error() == Some(libc::EINTR) => continue,
                Err(e) => return Err(ANNError::log_io_error(e)),
            }
        }
        self.drain_cqes(completed)
    }

    /// Drain all available CQEs from the completion queue.
    ///
    /// Processes every available CQE. On error or short-read the affected slot
    /// is returned to `Free` and the first error is propagated after all CQEs
    /// have been consumed (so no CQEs are left unprocessed).
    fn drain_cqes(&mut self, completed: &mut Vec<usize>) -> ANNResult<()> {
        completed.clear();
        let mut first_error: Option<ANNError> = None;

        for cqe in self.ring.completion() {
            let slot_id = cqe.user_data() as usize;
            debug_assert!(slot_id < self.max_slots);
            debug_assert_eq!(self.slot_states[slot_id], SlotState::InFlight);
            self.in_flight -= 1;

            if cqe.result() < 0 {
                self.slot_states[slot_id] = SlotState::Free;
                self.free_slots.push_back(slot_id);
                if first_error.is_none() {
                    first_error = Some(ANNError::log_io_error(
                        std::io::Error::from_raw_os_error(-cqe.result()),
                    ));
                }
                continue;
            }

            let bytes_read = cqe.result() as usize;
            if bytes_read < self.slot_size {
                self.slot_states[slot_id] = SlotState::Free;
                self.free_slots.push_back(slot_id);
                if first_error.is_none() {
                    first_error = Some(ANNError::log_io_error(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!(
                            "short read: expected {} bytes, got {}",
                            self.slot_size, bytes_read
                        ),
                    )));
                }
                continue;
            }

            self.slot_states[slot_id] = SlotState::Completed;
            completed.push(slot_id);
        }

        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    // ------------------------------------------------------------------
    // Slot access
    // ------------------------------------------------------------------

    /// Returns the read buffer for a completed slot.
    ///
    /// # Panics
    /// Panics if `slot_id` is out of range or the slot is not in `Completed`
    /// state (i.e. data is not yet ready or has already been released).
    pub fn get_slot_buf(&self, slot_id: usize) -> &[u8] {
        assert!(slot_id < self.max_slots, "slot_id out of range");
        assert_eq!(
            self.slot_states[slot_id],
            SlotState::Completed,
            "slot {slot_id} is not Completed (state: {:?})",
            self.slot_states[slot_id],
        );
        // SAFETY: The slot is Completed — the kernel has finished writing.
        // `buf_base` was derived from a valid, aligned allocation that outlives
        // `self`. The slice covers exactly `slot_size` bytes within bounds.
        unsafe { std::slice::from_raw_parts(self.buf_base.add(slot_id * self.slot_size), self.slot_size) }
    }

    /// Release a completed slot back to the free-list for reuse.
    ///
    /// # Panics
    /// Panics if the slot is not in `Completed` state.
    pub fn release_slot(&mut self, slot_id: usize) {
        assert!(slot_id < self.max_slots, "slot_id out of range");
        assert_eq!(
            self.slot_states[slot_id],
            SlotState::Completed,
            "cannot release slot {slot_id}: not Completed (state: {:?})",
            self.slot_states[slot_id],
        );
        self.slot_states[slot_id] = SlotState::Free;
        self.free_slots.push_back(slot_id);
    }

    // ------------------------------------------------------------------
    // Lifecycle helpers
    // ------------------------------------------------------------------

    /// Returns `true` if a free slot is available for [`enqueue_read`](Self::enqueue_read).
    pub fn has_free_slot(&self) -> bool {
        !self.free_slots.is_empty()
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

    /// Reset the reader for reuse: drain all in-flight IOs, release all
    /// completed slots, then restore every slot to `Free`.
    pub fn reset(&mut self) {
        self.drain_all();
    }

    /// Drain all in-flight IOs, blocking until they complete, then reset all
    /// slot states to `Free`.
    ///
    /// On transient errors (`EINTR`) retries automatically. On unrecoverable
    /// errors aborts the process — deallocating the buffer while the kernel
    /// still holds DMA references would cause memory corruption.
    fn drain_all(&mut self) {
        let mut remaining = self.in_flight;
        while remaining > 0 {
            match self.ring.submit_and_wait(remaining) {
                Ok(_) => {}
                Err(ref e) if e.raw_os_error() == Some(libc::EINTR) => continue,
                Err(_) => {
                    // Cannot safely deallocate while kernel may have DMA refs.
                    std::process::abort();
                }
            }
            for cqe in self.ring.completion() {
                let _ = cqe;
                remaining = remaining.saturating_sub(1);
            }
        }
        self.in_flight = 0;
        for state in &mut self.slot_states {
            *state = SlotState::Free;
        }
        self.free_slots.clear();
        self.free_slots.extend(0..self.max_slots);
    }
}

impl Drop for PipelinedReader {
    fn drop(&mut self) {
        // Must wait for all in-flight kernel IOs to complete before the
        // allocation backing `_slot_bufs` is freed.
        self.drain_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::io::Write;

    const SECTOR: usize = 4096;

    /// Create a temp file with `n_sectors` sectors of known data.
    /// Each sector is filled with the byte `(sector_index & 0xFF) as u8`.
    fn make_test_file(n_sectors: usize) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("create tempfile");
        for i in 0..n_sectors {
            let byte = (i & 0xFF) as u8;
            f.write_all(&vec![byte; SECTOR]).expect("write sector");
        }
        f.flush().expect("flush");
        f
    }

    /// Create a reader backed by a temp file. Returns both so the file
    /// outlives the reader.
    fn make_reader(
        n_sectors: usize,
        max_slots: usize,
    ) -> (tempfile::NamedTempFile, PipelinedReader) {
        let file = make_test_file(n_sectors);
        let reader = PipelinedReader::new(
            file.path().to_str().unwrap(),
            max_slots,
            SECTOR,
            SECTOR,
            &PipelinedReaderConfig::default(),
        )
        .unwrap();
        (file, reader)
    }

    /// Enqueue reads for `sectors`, flush, wait for all completions.
    /// Returns the slot IDs in enqueue order.
    fn enqueue_flush_wait(
        reader: &mut PipelinedReader,
        sectors: impl IntoIterator<Item = usize>,
    ) -> Vec<usize> {
        let mut slots = Vec::new();
        for s in sectors {
            slots.push(reader.enqueue_read((s * SECTOR) as u64).unwrap());
        }
        reader.flush().unwrap();
        drain_all_completions(reader);
        slots
    }

    /// Wait until all in-flight IOs complete.
    fn drain_all_completions(reader: &mut PipelinedReader) {
        let mut buf = Vec::new();
        while reader.in_flight_count() > 0 {
            reader.wait_completions(&mut buf).unwrap();
        }
    }

    /// Assert that a completed slot contains the expected fill byte for a
    /// given sector index (test files fill sector N with byte N & 0xFF).
    fn assert_sector_data(reader: &PipelinedReader, slot: usize, sector: usize) {
        let buf = reader.get_slot_buf(slot);
        let expected = (sector & 0xFF) as u8;
        assert!(
            buf.iter().all(|&b| b == expected),
            "slot {slot} (sector {sector}): expected 0x{expected:02x}, got 0x{:02x}",
            buf[0],
        );
    }

    // ===================================================================
    // Unit tests — each tests a single API behavior
    // ===================================================================

    #[test]
    fn slot_lifecycle_round_trip() {
        let (_f, mut reader) = make_reader(4, 4);

        // Enqueue → flush → wait → get_buf → release
        let slot = reader.enqueue_read(0).unwrap();
        assert_eq!(reader.slot_states[slot], SlotState::InFlight);

        reader.flush().unwrap();
        drain_all_completions(&mut reader);
        assert_eq!(reader.slot_states[slot], SlotState::Completed);

        assert_sector_data(&reader, slot, 0);
        reader.release_slot(slot);
        assert_eq!(reader.slot_states[slot], SlotState::Free);

        // Reuse the slot for a different sector
        let slots = enqueue_flush_wait(&mut reader, [1]);
        assert_sector_data(&reader, slots[0], 1);
        reader.release_slot(slots[0]);
    }

    #[test]
    fn slot_exhaustion_returns_error() {
        let (_f, mut reader) = make_reader(8, 4);
        for i in 0..4 {
            reader.enqueue_read((i * SECTOR) as u64).unwrap();
        }
        assert!(reader.enqueue_read(0).is_err());
    }

    #[test]
    #[should_panic(expected = "not Completed")]
    fn double_release_panics() {
        let (_f, mut reader) = make_reader(1, 2);
        let slots = enqueue_flush_wait(&mut reader, [0]);
        reader.release_slot(slots[0]);
        reader.release_slot(slots[0]); // should panic
    }

    #[test]
    #[should_panic(expected = "not Completed")]
    fn get_buf_on_free_slot_panics() {
        let (_f, reader) = make_reader(1, 2);
        reader.get_slot_buf(0);
    }

    #[test]
    #[should_panic(expected = "not Completed")]
    fn get_buf_on_inflight_slot_panics() {
        let (_f, mut reader) = make_reader(1, 2);
        let slot = reader.enqueue_read(0).unwrap();
        reader.flush().unwrap();
        reader.get_slot_buf(slot); // still InFlight
    }

    #[test]
    fn drop_drains_in_flight() {
        let (_f, mut reader) = make_reader(4, 4);
        for i in 0..4 {
            reader.enqueue_read((i * SECTOR) as u64).unwrap();
        }
        reader.flush().unwrap();
        drop(reader); // must not panic or leak
    }

    #[test]
    fn data_integrity_multi_slot() {
        let (_f, mut reader) = make_reader(8, 4);
        let slots = enqueue_flush_wait(&mut reader, 0..4);
        for (slot, sector) in slots.iter().zip(0..4) {
            assert_sector_data(&reader, *slot, sector);
            reader.release_slot(*slot);
        }
    }

    #[test]
    fn reset_clears_all_state() {
        let (_f, mut reader) = make_reader(4, 4);
        enqueue_flush_wait(&mut reader, [0, 1]);
        reader.enqueue_read(2 * SECTOR as u64).unwrap();
        reader.flush().unwrap();

        reader.reset();
        assert_eq!(reader.in_flight, 0);
        assert_eq!(reader.free_slots.len(), 4);
        assert!(reader.slot_states.iter().all(|&s| s == SlotState::Free));
    }

    #[test]
    fn poll_and_wait_return_empty_when_idle() {
        let (_f, mut reader) = make_reader(1, 2);
        let mut buf = Vec::new();
        reader.poll_completions(&mut buf).unwrap();
        assert!(buf.is_empty());
        reader.wait_completions(&mut buf).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    fn short_read_detected_as_error() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&vec![0xABu8; 512]).unwrap(); // < SECTOR
        f.flush().unwrap();

        let mut reader = PipelinedReader::new(
            f.path().to_str().unwrap(),
            1,
            SECTOR,
            SECTOR,
            &PipelinedReaderConfig::default(),
        )
        .unwrap();
        reader.enqueue_read(0).unwrap();
        reader.flush().unwrap();

        let mut completed = Vec::new();
        let result = reader.wait_completions(&mut completed);
        assert!(result.is_err(), "short read should be detected");
        assert!(completed.is_empty());
    }

    #[test]
    fn drop_with_unflushed_sqes() {
        let (_f, mut reader) = make_reader(8, 8);
        for i in 0..8 {
            reader.enqueue_read((i * SECTOR) as u64).unwrap();
        }
        // Enqueued but never flushed — drain_all's submit_and_wait handles it
        drop(reader);
    }

    // ===================================================================
    // Stress tests — exercise the state machine at scale
    // ===================================================================

    /// Randomized state-machine fuzzer using seeded RNG for reproducibility.
    /// Exercises random interleavings of enqueue, flush, poll, wait, release,
    /// and reset with data verification.
    #[test]
    fn stress_random_slot_lifecycle() {
        let (_f, mut reader) = make_reader(256, 16);
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
        let mut pending_completed: Vec<usize> = Vec::new();
        let mut total_verified = 0u64;

        for _ in 0..2000 {
            match rng.random_range(0u32..100) {
                0..40 => {
                    if reader.has_free_slot() {
                        let sector = rng.random_range(0usize..256);
                        reader.enqueue_read((sector * SECTOR) as u64).unwrap();
                    }
                }
                40..55 => {
                    reader.flush().unwrap();
                }
                55..70 => {
                    let mut buf = Vec::new();
                    reader.poll_completions(&mut buf).unwrap();
                    pending_completed.extend_from_slice(&buf);
                }
                70..80 => {
                    if reader.in_flight_count() > 0 {
                        reader.flush().unwrap();
                        let mut buf = Vec::new();
                        reader.wait_completions(&mut buf).unwrap();
                        pending_completed.extend_from_slice(&buf);
                    }
                }
                80..95 => {
                    if let Some(slot) = pending_completed.pop() {
                        let buf = reader.get_slot_buf(slot);
                        let first = buf[0];
                        assert!(
                            buf.iter().all(|&b| b == first),
                            "data corruption in slot {slot}"
                        );
                        reader.release_slot(slot);
                        total_verified += 1;
                    }
                }
                _ => {
                    pending_completed.clear();
                    reader.reset();
                }
            }
        }

        // Cleanup: flush + drain remaining
        reader.flush().unwrap();
        let mut buf = Vec::new();
        while reader.in_flight_count() > 0 {
            reader.wait_completions(&mut buf).unwrap();
            for &slot in &buf {
                let data = reader.get_slot_buf(slot);
                assert!(data.iter().all(|&b| b == data[0]));
                reader.release_slot(slot);
                total_verified += 1;
            }
        }
        for &slot in &pending_completed {
            let data = reader.get_slot_buf(slot);
            assert!(data.iter().all(|&b| b == data[0]));
            reader.release_slot(slot);
            total_verified += 1;
        }
        assert!(total_verified > 0, "stress test verified zero reads");
    }

    /// Saturate all slots, drain, repeat — catches off-by-one in free-list.
    #[test]
    fn stress_saturate_and_drain_cycles() {
        let max_slots = 32;
        let (_f, mut reader) = make_reader(max_slots, max_slots);

        for cycle in 0..100 {
            let sectors: Vec<usize> =
                (0..max_slots).map(|i| (cycle * max_slots + i) % max_slots).collect();
            let slots = enqueue_flush_wait(&mut reader, sectors.iter().copied());
            assert!(reader.enqueue_read(0).is_err());

            for (slot, &sector) in slots.iter().zip(sectors.iter()) {
                assert_sector_data(&reader, *slot, sector);
                reader.release_slot(*slot);
            }
        }
    }

    /// 1-slot reader: max state transitions per slot.
    #[test]
    fn stress_single_slot_rapid_reuse() {
        let n_sectors = 64;
        let (_f, mut reader) = make_reader(n_sectors, 1);

        for i in 0..500 {
            let sector = i % n_sectors;
            let slots = enqueue_flush_wait(&mut reader, [sector]);
            assert_sector_data(&reader, slots[0], sector);
            reader.release_slot(slots[0]);
        }
    }

    /// Drop with 0, 1, 2, … max_slots in-flight IOs.
    #[test]
    fn stress_drop_at_various_inflight_counts() {
        let max_slots = 16;
        for inflight in 0..=max_slots {
            let (_f, mut reader) = make_reader(max_slots, max_slots);
            for i in 0..inflight {
                reader.enqueue_read((i * SECTOR) as u64).unwrap();
            }
            if inflight > 0 {
                reader.flush().unwrap();
            }
            drop(reader);
        }
    }

    /// Read every sector in a 256-sector file through 8 slots, verify all.
    #[test]
    fn stress_full_file_sequential_scan() {
        let n_sectors = 256;
        let max_slots = 8;
        let (_f, mut reader) = make_reader(n_sectors, max_slots);

        let mut sectors_verified = vec![false; n_sectors];
        let mut slot_to_sector = [0usize; 128];
        let mut next_sector = 0usize;
        let mut buf = Vec::new();

        while next_sector < n_sectors || reader.in_flight_count() > 0 {
            while next_sector < n_sectors && reader.has_free_slot() {
                let slot = reader.enqueue_read((next_sector * SECTOR) as u64).unwrap();
                slot_to_sector[slot] = next_sector;
                next_sector += 1;
            }
            reader.flush().unwrap();

            reader.wait_completions(&mut buf).unwrap();
            for &slot in &buf {
                let sector = slot_to_sector[slot];
                assert_sector_data(&reader, slot, sector);
                sectors_verified[sector] = true;
                reader.release_slot(slot);
            }
        }

        assert!(sectors_verified.iter().all(|&v| v), "not all sectors verified");
    }
}
