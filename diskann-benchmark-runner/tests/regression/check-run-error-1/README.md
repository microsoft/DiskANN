In this test is a little complicated. What we do is use `input.json` to generate an `output.json`.
Then during the check run, we use a different `regression_input.json`.
This second file as a different number of entries.

The test then verifies that we properly reject this situation.
