# Exact tile quick verification evidence (2026-10-23)

Frozen final binary: 7beea60c66cd4128bb58ee571d98cc6a02568d28212a8dda55b1a99abe842dbd.
Source parent: 4f84618561e0208199a963cb8cb0a99ddfe6f67b; source hashes in provenance.

- summary.json retains every build/search result and the per-block A/B samples.
  Twelve complete builds; YFCC 100 queries, Enron/FBV8 first 32 queries.
- fit.json/fit-summary.json retain 288 measured complete fits across 18 shapes,
  four ABBA blocks per shape, plus explicit warmup order and exact diagnostics.
- regression.json records six layouts of the three earlier synthetic divergence
  configurations; all per-step, center, sum, assignment and RNG checks pass.
- audit-summary.json records the four local real-data counterexamples and 56,400
  parent point-step decisions. Coordinates/parent files are not published.
- development/ retains the initial exact-but-slow closure version and generated
  assembly evidence. It has a different frozen binary and is not pooled.
- raw-manifest.json hashes local invocation input/output/log/metadata files.
  Public JSON normalizes local paths; normalized file hashes differ intentionally.
- coverage/ contains local selected-source reports, not GitHub Codecov results.

The final full-build experiment completed at 01:58:07 UTC before coverage/tests.
Later instrumented benchmark tests took 4,138 s; none of that time is a sample.
No source compilation overlapped the timed experiment. The machine is a shared
Windows desktop with OS-managed clocks; no exclusive-host guarantee exists.

One ABBA block per real dataset is a quick observation, not a stable speedup
claim. Search timings varied substantially even with identical search results
and logical work; all timings are retained. Logical scans are not physical I/O.
