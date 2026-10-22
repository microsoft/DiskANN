# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Windows runner: pin one benchmark process and preserve its timing provenance."""

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import subprocess
import time


def active_workloads():
    """Read Windows process names without invoking another shell or requiring psutil."""
    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD),
            ('th32ProcessID', wintypes.DWORD), ('th32DefaultHeapID', ctypes.c_size_t),
            ('th32ModuleID', wintypes.DWORD), ('cntThreads', wintypes.DWORD),
            ('th32ParentProcessID', wintypes.DWORD), ('pcPriClassBase', wintypes.LONG),
            ('dwFlags', wintypes.DWORD), ('szExeFile', wintypes.WCHAR * 260),
        ]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = kernel.Process32FirstW.argtypes
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    processes = []
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        more = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while more:
            if entry.szExeFile.lower() in ['cargo.exe', 'rustc.exe', 'diskann-benchmark.exe']:
                processes.append({'pid': entry.th32ProcessID, 'name': entry.szExeFile,
                                  'parent_pid': entry.th32ParentProcessID})
            more = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(snapshot)
    return processes


def wait_for_idle():
    started = time.perf_counter()
    previous = None
    while True:
        workloads = active_workloads()
        if not workloads:
            return time.perf_counter() - started
        if workloads != previous:
            print(json.dumps({'waiting_for_other_workloads': workloads}), flush=True)
            previous = workloads
        time.sleep(10)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--affinity', type=lambda value: int(value, 0), default=16)
    args = parser.parse_args()
    if args.affinity <= 0:
        parser.error('--affinity must be a positive processor mask')
    executable, source, output = (path.resolve() for path in (args.exe, args.input, args.output))
    metadata_path = output.with_suffix('.metadata.json')
    log_path = output.with_suffix('.log')
    if any(path.exists() for path in [output, metadata_path, log_path]):
        parser.error('output, metadata, or log already exists; use unique names')
    output.parent.mkdir(parents=True, exist_ok=True)
    idle_wait = wait_for_idle()

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetPriorityClass.argtypes = [wintypes.HANDLE]
    kernel.GetPriorityClass.restype = wintypes.DWORD
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), args.affinity):
        raise ctypes.WinError(ctypes.get_last_error())

    command = [str(executable), 'run', '--input-file', str(source), '--output-file', str(output)]
    metadata = {'command': command, 'affinity_requested': args.affinity,
                'executable_sha256': sha256(executable), 'input_sha256': sha256(source),
                'idle_wait_seconds': idle_wait, 'overlap_observations': [],
                'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    started = time.perf_counter()
    with log_path.open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.NORMAL_PRIORITY_CLASS)
        metadata['pid'] = process.pid
        try:
            # Popen retains the process handle after exit, allowing GetProcessTimes.
            handle = int(process._handle)
            process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
            if not kernel.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
                raise ctypes.WinError(ctypes.get_last_error())
            metadata['affinity_observed'] = process_mask.value
            metadata['priority_class'] = kernel.GetPriorityClass(handle)
            if process_mask.value != args.affinity or metadata['priority_class'] != 0x20:
                raise RuntimeError('child affinity or Normal priority verification failed')
            print(json.dumps({'pid': process.pid, 'output': str(output), 'affinity': process_mask.value}), flush=True)
            while True:
                try:
                    metadata['exit_code'] = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    overlaps = [item for item in active_workloads() if item['pid'] != process.pid]
                    if overlaps:
                        metadata['overlap_observations'].append({
                            'elapsed_seconds': time.perf_counter() - started,
                            'processes': overlaps,
                        })
            created, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel_time), ctypes.byref(user_time)):
                raise ctypes.WinError(ctypes.get_last_error())
            for name, value in [('kernel_seconds', kernel_time), ('user_seconds', user_time)]:
                metadata[name] = ((value.dwHighDateTime << 32) + value.dwLowDateTime) / 1e7
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            metadata.setdefault('exit_code', process.returncode)
            metadata['wall_seconds'] = time.perf_counter() - started
            metadata['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            metadata_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metadata), flush=True)
    if metadata['exit_code'] != 0:
        raise SystemExit(metadata['exit_code'])


if __name__ == '__main__':
    run()
