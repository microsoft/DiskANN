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
	- `memory`: run **in-memory** index perf only
	- `disk`: run **SSD/disk** index perf only
	- `both`: run **both** memory + disk perf

`PERF_MODE` details:

- `PERF_MODE=memory`
	- Builds and searches **in-memory** indexes (`build_memory_index`, `search_memory_index`).
	- For `DATA_TYPE=float`, also runs `fast_l2` memory search.

- `PERF_MODE=disk`
	- Builds and searches **SSD/disk** indexes (`build_disk_index`, `search_disk_index`).
	- For `DATA_TYPE=float`, runs disk perf for `l2`, `mips`, `cosine`.
	- For `DATA_TYPE=bf16`, runs disk perf for `l2`, `cosine`:
		- full-precision disk (`PQ_disk_bytes=0`)
		- disk-PQ + reorder (`--append_reorder_data` / `--use_reorder_data`)

- `PERF_MODE=both`
	- Runs everything from both `memory` and `disk`.

Legacy compatibility:

- `RUN_DISK=1` maps to `PERF_MODE=both`
- `RUN_DISK=0` maps to `PERF_MODE=memory`

Examples:

```bash
# Memory index perf with bf16
DATA_TYPE=bf16 PERF_MODE=memory ./scripts/perf/perf_test.sh

# Memory + disk perf with bf16
DATA_TYPE=bf16 PERF_MODE=both ./scripts/perf/perf_test.sh

# Disk index perf (float)
DATA_TYPE=float PERF_MODE=disk ./scripts/perf/perf_test.sh

# Disk index perf (bf16): runs both full-precision disk and disk-PQ(+reorder)
DATA_TYPE=bf16 PERF_MODE=disk ./scripts/perf/perf_test.sh

# Disk index perf (bf16) with custom disk-PQ bytes (default: 8)
DATA_TYPE=bf16 PERF_MODE=disk DISK_BF16_PQ_DISK_BYTES=16 ./scripts/perf/perf_test.sh
```

Notes:

- For `DATA_TYPE=bf16` and `PERF_MODE` includes `disk`, the script runs:
	- full-precision disk (`PQ_disk_bytes=0`) and
	- disk-PQ with reorder (`--append_reorder_data` / `--use_reorder_data`).
- `DISK_BF16_PQ_DISK_BYTES` controls the bf16 disk-PQ compression level for the disk-PQ runs (default: 8).
- For backward compatibility, `RUN_DISK` is still accepted and overrides `PERF_MODE`.

`docker build` must be run with the context directory set to `scripts`, but the Dockerfile set to `scripts/perf/Dockerfile` as in:
```bash
docker build [--build-arg GIT_COMMIT_ISH=<rev>] -f scripts/perf/Dockerfile scripts
```

We prefer to install the dependencies from the commit-ish that we're building against, but as the deps were not stored 
in a known file in all commits, we will fall back to the one currently in HEAD if one is not found already.

The `--build-arg GIT_COMMIT_ISH=<rev>` is optional, with a default value of HEAD if not otherwise specified.
