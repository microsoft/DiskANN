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

| Status | Meaning |
|--------|---------|
| **Draft** | Initial proposal, open for discussion |
| **InReview** | Published and open for comments from maintainers and stakeholders |
| **Accepted** | Approved by maintainers and merged |
| **Rejected** | Declined — the RFC documents why |

## How to Submit an RFC

1. Copy [0000-template.md](0000-template.md) to `NNNN-short-title.md` (use the next available number).
2. Fill in all sections. Remove instructional comments.
3. Open a pull request with the RFC file. The PR description should summarize the proposal.
4. Discuss in the PR. Update the RFC based on feedback.
5. Once accepted, the RFC is merged and the status is updated to **Accepted**.
