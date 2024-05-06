#Performance Tests

The bash scripts in this folder are responsible for running a suite of performance 
tests.

The timing and recall metrics reported by these tests when run periodically can then
be used to identify performance improvements or regressions as 
development continues.

## Usage

`docker build` must be run with the context directory set to `scripts`, but the Dockerfile set to `scripts/perf/Dockerfile` as in:
```bash
docker build [--build-arg GIT_COMMIT_ISH=<rev>] -f scripts/perf/Dockerfile scripts
```

We prefer to install the dependencies from the commit-ish that we're building against, but as the deps were not stored 
in a known file in all commits, we will fall back to the one currently in HEAD if one is not found already.

The `--build-arg GIT_COMMIT_ISH=<rev>` is optional, with a default value of HEAD if not otherwise specified.
