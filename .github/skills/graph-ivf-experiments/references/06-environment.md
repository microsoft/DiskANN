# Environment and tooling

Machine-specific behaviour that has cost real time on this workstation. Read this first
when something misbehaves for no apparent reason.

## Python interpreters

Two interpreters, and they are **not** interchangeable:

| | `.venv\Scripts\python.exe` | conda base (`miniconda3\python.exe`, on PATH as `python`) |
|---|---|---|
| numpy | 2.5.1 | 2.2.6 |
| openpyxl | 3.1.5 | 3.1.5 |
| **matplotlib** | **absent** | 3.10.9 |

- **Plot scripts must run under conda base.** The venv fails with
  `ModuleNotFoundError: No module named 'matplotlib'`.
- Dataset prep (numpy-only) runs fine under either.
- The Pylance run-code-snippet tool uses the **venv**, so it can do numpy and openpyxl
  inspection but not plotting.

Scripts `import benchlib` by bare name, so run them from `_results/scripts/`:

```powershell
Push-Location "$(git rev-parse --show-toplevel)\_results\scripts"
python plot_online_bytes_recall.py
Pop-Location
```

## PowerShell

- **Never begin a command with `Set-Location`.** Use `Push-Location` / `Pop-Location`.
- **Redirect with `2>&1 | Tee-Object -FilePath … | Out-Null`**, not `Out-File` —
  `Out-File` writes UTF-16LE by default. (`benchlib.read_text` handles the BOM for
  historical logs, but do not create new ones.)
- **A `foreach` loop over benchmark invocations silently no-ops.** Loops are fine for short
  Python scripts; issue benchmark runs one at a time.
- **The terminal intermittently injects a `^U` control character** at the start of a
  command, corrupting the first token. Symptom: "the term 'ython' is not recognized" or a
  mangled path. Just retry; invoking the executable with a literal absolute path and no
  leading `&` is more robust.
- A sync command occasionally reports "Command produced no output" while silently killing
  the child. Re-run in async mode.

## Long-running jobs

- Use **async mode** for builds and sweeps; you are notified on completion. Do not poll and
  do not `Start-Sleep`.
- **Terminal cleanup kills running children.** Do not tidy up terminals while a build or
  sweep is in flight.
- Always end a run command with `Write-Output "DONE exit=$LASTEXITCODE"` so completion and
  status are unambiguous in the captured output.

## Resources

- 64 GB RAM; ~1.27 TB free on C:.
- An 8 GB fp16 corpus plus a 4 GB minmax8 copy plus saved indexes is ~15 GB per large
  dataset. Check free space before adding one.
- Stream large corpora with `np.memmap` in chunks (65536 rows works well). Never load one
  whole.

## Repository state

- `_results/` is **untracked** — not gitignored, just never committed. It holds all logs,
  scripts, plots and workbooks.
- `build_online_workbook.py` **at the repository root is a stale duplicate** and is marked
  as such in its docstring. The live copy is `_results/scripts/build_online_workbook.py`.
- Experiment work happens on `u/adkrishnan/graph-ivf`; the default branch is `main`.
