# Debug Mode Error Test

This test verifies that running benchmarks in debug mode without the `--allow-debug` flag produces an error.

The test intentionally omits the `--allow-debug` flag to ensure the debug mode check fires and blocks execution with an appropriate error message.

This test only runs when `cfg!(debug_assertions)` is true (i.e., in debug builds).
