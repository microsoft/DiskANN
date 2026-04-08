In this test - we verify that if a check fails with an error, we

- Only print the errors for the failing tests (to avoid spamming stdout)
- Record which of the entries failed (there can be multiple)
- Verify that errors get recorded in the output `checks.json` file.
