#Performance Tests

The bash scripts in this folder are responsible for running a suite of performance 
tests.

The timing and recall metrics reported by these tests when run periodically can then
be used to identify performance improvements or regressions as 
development continues.

## Usage

### Run the perf suite

The main entrypoint is `scripts/perf/perf_test.sh`.

Environment variables:

- `DATA_TYPE` (default: `float`)
	- Supported: `float`, `bf16`
- `PERF_MODE` (default: `memory`)
	- `memory`: run in-memory index perf only
	- `disk`: run SSD/disk index perf only
	- `both`: run memory + disk perf

Examples:

```bash
# Memory index perf with bf16
DATA_TYPE=bf16 PERF_MODE=memory ./scripts/perf/perf_test.sh

# Disk index perf (float-only)
DATA_TYPE=float PERF_MODE=disk ./scripts/perf/perf_test.sh
```

Notes:

- Disk index perf is currently **float-only**. If `PERF_MODE` includes `disk` and `DATA_TYPE!=float`, disk tests are skipped.
- Legacy `RUN_DISK=1` is still accepted and maps to `PERF_MODE=both` for backward compatibility.

`docker build` must be run with the context directory set to `scripts`, but the Dockerfile set to `scripts/perf/Dockerfile` as in:
```bash
docker build [--build-arg GIT_COMMIT_ISH=<rev>] -f scripts/perf/Dockerfile scripts
```

We prefer to install the dependencies from the commit-ish that we're building against, but as the deps were not stored 
in a known file in all commits, we will fall back to the one currently in HEAD if one is not found already.

The `--build-arg GIT_COMMIT_ISH=<rev>` is optional, with a default value of HEAD if not otherwise specified.
