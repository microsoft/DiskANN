---
mode: 'agent'
description: 'Generate tests for the current file following DiskANN conventions'
---

Generate tests for the code in #{file:${file}}.

Follow these DiskANN testing conventions:
- Do not add tests for derived traits (Clone, Debug, PartialEq)
- Do not add tests for enums unless they have explicit functionality
- Look for existing test infrastructure before creating new patterns
- Use `VirtualStorageProvider::new_overlay()` for storage in tests (never `new_physical()`)
- Use functions from `random.rs` instead of `rand::thread_rng`
- Check if the crate has a `test` module with shared helpers
