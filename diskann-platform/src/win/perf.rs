/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::mem::{self, zeroed};

use windows_sys::Win32::{
    Foundation::FILETIME,
    System::{
        ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
        SystemInformation::{GetSystemInfo, SYSTEM_INFO},
        Threading::{GetCurrentProcess, GetProcessTimes, GetSystemTimes},
        WindowsProgramming::QueryProcessCycleTime,
    },
};

pub fn get_process_cycle_time() -> Option<u64> {
    let mut cycle_time: u64 = 0;

    // SAFETY: Call a Win32 API.
    let handle = unsafe { GetCurrentProcess() };

    let result = unsafe { QueryProcessCycleTime(handle, &mut cycle_time as *mut u64) } != 0;
    if result {
        return Some(cycle_time);
    }
    None
}

// Gets the process time in kernel and user modes.
pub fn get_process_time() -> Option<u64> {
    let handle = unsafe { GetCurrentProcess() };

    let mut creation_time: FILETIME = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let mut exit_time: FILETIME = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let mut kernel_time: FILETIME = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let mut user_time: FILETIME = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };

    let result = unsafe {
        GetProcessTimes(
            handle,
            &mut creation_time,
            &mut exit_time,
            &mut kernel_time,
            &mut user_time,
        )
    };

    if result != 0 {
        let kernel_time = filetime_to_u64(kernel_time);
        let user_time = filetime_to_u64(user_time);
        return Some(kernel_time + user_time);
    }

    None
}

// Gets the system time in kernel and user modes.
pub fn get_system_time() -> Option<u64> {
    let mut idle_time: FILETIME = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let mut kernel_time: FILETIME = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let mut user_time: FILETIME = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };

    let result = unsafe { GetSystemTimes(&mut idle_time, &mut kernel_time, &mut user_time) };

    if result != 0 {
        let kernel_time = filetime_to_u64(kernel_time);
        let user_time = filetime_to_u64(user_time);
        return Some(kernel_time + user_time);
    }

    None
}

// Gets the number of processors.
pub fn get_number_of_processors() -> Option<u64> {
    let mut system_info: SYSTEM_INFO = unsafe { zeroed() };
    unsafe { GetSystemInfo(&mut system_info) };
    Some(system_info.dwNumberOfProcessors as u64)
}

#[inline(always)]
const fn filetime_to_u64(ft: FILETIME) -> u64 {
    ((ft.dwHighDateTime as u64) << 32) | (ft.dwLowDateTime as u64)
}

pub fn get_peak_workingset_size() -> Option<u64> {
    // SAFETY: Call a Win32 API.
    let handle = unsafe { GetCurrentProcess() };

    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { mem::zeroed() };
    counters.cb = mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;

    let result = unsafe {
        GetProcessMemoryInfo(
            handle,
            &mut counters as *mut PROCESS_MEMORY_COUNTERS,
            counters.cb,
        )
    };

    if result != 0 {
        let peak_working_set_size = counters.PeakWorkingSetSize;
        return Some(peak_working_set_size as u64);
    }
    None
}
