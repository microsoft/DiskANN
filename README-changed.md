# PR: Remove RaBitQ reorder-prefilter + README cleanup

## What changed

This PR removes the previously added **RaBitQ reorder prefilter** feature end-to-end, while keeping **RaBitQ main-search approximate scoring** support.

### DiskANN core (SSD search)

- Removed all reorder-prefilter state and logic from `PQFlashIndex`.
  - No longer loads/uses `<index>_rabitq_reorder.bin`.
  - No longer supports `DISKANN_USE_RABITQ_REORDER_PREFILTER` / related tuning env vars.
- Kept RaBitQ **main traversal** approximate scorer support (runtime-gated).

### Build / CLI integration

- Removed build-time generation options and plumbing for reorder-prefilter codes.
- Removed the reorder-code generator app from the build targets.

### Documentation

- Removed the entire “RaBitQ multi-bit reorder prefilter” section from the top-level README.
- Updated the RaBitQ main-search README section to remove the old fallback wording that referenced reorder-code sidecars.

## Why

The reorder-prefilter mode is no longer desired; maintaining only the RaBitQ main-search mode simplifies the feature surface and avoids documenting/maintaining a removed behavior.

## Notes

- RaBitQ main-search mode remains runtime-gated (disabled by default).
- If you have existing indexes/code sidecars produced for reorder-prefilter, they are no longer consumed by the library.
