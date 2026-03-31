When two benchmarks match the same input, the one with the better MatchScore wins.

`ExactTypeBench<f32, 1000>` matches float32 + dim=1000 with MatchScore(0).
`TypeBench<f32>` matches float32 (any dim) with MatchScore(10).

The runner should pick `ExactTypeBench`.
