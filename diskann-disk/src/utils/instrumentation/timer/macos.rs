/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// Gets the process cycle time.
/// On macOS, this returns 0 as cycle time is not readily available.
pub fn get_process_cycle_time() -> Option<u64> {
    Some(0)
}

/// Gets the process time in kernel and user modes.
/// On macOS, we use rusage to get process times.
pub fn get_process_time() -> Option<u64> {
    use libc::{getrusage, rusage, RUSAGE_SELF};

    let mut usage: rusage = unsafe { std::mem::zeroed() };
    let result = unsafe { getrusage(RUSAGE_SELF, &mut usage) };

    if result == 0 {
        // Convert timeval to 100-nanosecond units (to match Windows FILETIME)
        // tv_sec is in seconds, tv_usec is in microseconds
        // 1 second = 10_000_000 * 100ns, 1 microsecond = 10 * 100ns
        let user_time =
            (usage.ru_utime.tv_sec as u64) * 10_000_000 + (usage.ru_utime.tv_usec as u64) * 10;
        let system_time =
            (usage.ru_stime.tv_sec as u64) * 10_000_000 + (usage.ru_stime.tv_usec as u64) * 10;
        return Some(user_time + system_time);
    }

    None
}

/// Gets the system time in kernel and user modes.
/// On macOS, this is not easily available, so we return None.
pub fn get_system_time() -> Option<u64> {
    None
}

/// Gets the number of processors.
pub fn get_number_of_processors() -> Option<u64> {
    use libc::{sysconf, _SC_NPROCESSORS_ONLN};

    let result = unsafe { sysconf(_SC_NPROCESSORS_ONLN) };
    if result > 0 {
        return Some(result as u64);
    }

    None
}

/// Retrieves the peak resident set size of the current process.
///
/// On macOS, this function uses getrusage to retrieve the maximum resident set size.
/// The ru_maxrss field represents the maximum resident set size used (in bytes on macOS).
///
/// # Returns
///
/// An `Option<u64>` representing the peak working set size in bytes, or `None` if the operation fails.
pub fn get_peak_workingset_size() -> Option<u64> {
    use libc::{getrusage, rusage, RUSAGE_SELF};

    let mut usage: rusage = unsafe { std::mem::zeroed() };
    let result = unsafe { getrusage(RUSAGE_SELF, &mut usage) };

    if result == 0 {
        // On macOS, ru_maxrss is in bytes (unlike Linux where it's in kilobytes)
        return Some(usage.ru_maxrss as u64);
    }

    None
}
