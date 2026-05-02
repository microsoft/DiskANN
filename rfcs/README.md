# DiskANN RFCs

This directory contains **Requests for Comments (RFCs)** — design documents for substantial changes to DiskANN.

## When to Write an RFC

An RFC is recommended for changes that are **cross-cutting, architecturally significant, or affect widely-used APIs** — especially when the design has multiple viable approaches that benefit from broader input. Examples include:

- Adding a new crate to the workspace
- Introducing or modifying cross-crate traits or abstractions
- New distance functions, storage layouts, or index formats
- Changes with backward-compatibility implications for downstream consumers

An RFC is **not** required for routine API additions scoped to a single crate (e.g., adding a variant to an internal enum), bug fixes, internal refactors, or documentation improvements.

## RFC Lifecycle

The pull request is the RFC. **Merging an RFC PR is acceptance.**

- **Open PR** — the RFC is under discussion. Authors iterate on feedback.
- **Merged PR** — the RFC is accepted. Only accepted RFCs appear in this directory.
- **Closed PR** — the RFC is rejected or withdrawn. The full text and discussion remain available in the closed PR.

Tag every RFC pull request with the **`RFC`** label so they are easy to find.

## How to Submit an RFC

1. Copy [00000-template.md](00000-template.md) to `00000-short-title.md`.
2. Fill in all sections. Remove instructional comments.
3. Open a pull request with the RFC file and apply the **`RFC`** label. The PR description should summarize the proposal.
4. Once the PR is created, rename the file from `00000-short-title.md` to `NNNNN-short-title.md`, where `NNNNN` is the pull request number zero-padded to 5 digits.
5. Discuss in the PR. Update the RFC based on feedback.
6. A maintainer merges the PR to accept the RFC, or closes it to reject/withdraw.
