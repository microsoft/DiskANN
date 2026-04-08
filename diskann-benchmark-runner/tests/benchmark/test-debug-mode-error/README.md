# Debug Mode Error Test

This test verifies that running benchmarks in debug mode without the `--allow-debug` flag produces an error.

The test intentionally omits the `--allow-debug` flag to ensure the debug mode check fires and blocks execution with an appropriate error message.

- **In debug builds**: The debug mode check fires and blocks execution with an error message (validated against `stdout.txt`)
- **In release builds**: No debug assertions, so the benchmark runs normally with empty jobs (validated against `stdout_release.txt`)
