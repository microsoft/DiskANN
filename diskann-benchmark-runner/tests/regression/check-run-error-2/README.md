This test covers input drift after output generation.

The tolerance/input tags are still compatible, but the regression input has changed to
`float64`, which no registered regression benchmark supports. `check run` should fail with
an explicit "no matching regression benchmark" diagnostic instead of panicking.
