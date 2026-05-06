---
applyTo: '.github/workflows/*.yml'
---
# CI Workflow Conventions
**When to read:** Modifying GitHub Actions workflow files.

- Do not modify CI workflows unless explicitly asked
- The `RUST_CONFIG` env var in ci.yml sets `-Dwarnings` — do not remove or weaken this
- Test jobs run on both Linux and Windows — ensure changes are cross-platform
