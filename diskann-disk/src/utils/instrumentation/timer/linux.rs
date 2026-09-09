/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fs::File,
    io::{self, BufRead},
    path::Path,
};

pub fn get_process_cycle_time() -> Option<u64> {
    Some(0)
}

// Gets the process time in kernel and user modes.
pub fn get_process_time() -> Option<u64> {
    None
}

// Gets the system time in kernel and user modes.
pub fn get_system_time() -> Option<u64> {
    None
}

// Gets the number of processors.
pub fn get_number_of_processors() -> Option<u64> {
    None
}

/// Retrieves the peak resident set size of the current process.
///
/// This function returns the VmHWM field from the /proc/self/status file, which represents the maximum amount of memory that the process has used at any point in time. The resident set size is the portion of a process's memory that is held in RAM.
/// The VmHWM value might be inaccurate, according to https://manpages.ubuntu.com/manpages/jammy/man5/proc.5.html.
/// If there are accuracy concerns, consider switching to /proc/[pid]/smaps or /proc/[pid]/smaps_rollup instead, which are much slower but provide accurate, detailed information.
///
/// # Arguments
///
/// * `process_handle` - An optional process handle. This argument is ignored on Linux.
///
/// # Returns
///
/// An `Option<u64>` representing the peak working set size in bytes, or `None` if the operation fails or is not supported on the current platform.
pub fn get_peak_workingset_size() -> Option<u64> {
    if cfg!(unix) {
        // Open the file
        let path = Path::new("/proc/self/status");
        let file = File::open(path).ok()?;
        let reader = io::BufReader::new(file);

        // Read the file line by line
        for line in reader.lines() {
            let line = line.ok()?;
            // Look for the VmHWM field
            if line.starts_with("VmHWM:") {
                // Split the line into parts
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    // Parse the value as u64
                    if let Ok(value) = parts[1].parse::<u64>() {
                        // Return the value in bytes
                        return Some(value * 1024);
                    }
                }
            }
        }
    }

    None
}
