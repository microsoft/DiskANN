This test covers before/after schema drift with matching entry counts.

The setup output is generated from a `dim` benchmark, producing integer results. The
regression check is then run against a `test-input-types` benchmark, which expects string
results. The check should report a structured deserialization error and still write
`checks.json`.
