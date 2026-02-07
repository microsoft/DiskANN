# DiskANN RFCs

This directory contains **Requests for Comments (RFCs)** — design documents for substantial changes to DiskANN.

## When to Write an RFC

An RFC is required for changes that affect the public API or architecture of the project, or whenever an important design decision needs review from maintainers and stakeholders. Examples include:

- Adding a new crate to the workspace
- Introducing or modifying cross-crate APIs or traits
- New distance functions, storage layouts, or index formats
- Changes with backward-compatibility implications
- Design decisions with multiple viable approaches that benefit from broader input

An RFC is **not** required for bug fixes, internal refactors that don't change public APIs, documentation improvements, or small feature additions scoped to a single crate.

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

## Index

| # | Title | Status | Author | Resolved Date |
|---|-------|--------|--------|---------------|
| [0000](0000-template.md) | RFC Template | - | - | - |
| [0001](0001-multi-vector-distance-functions.md) | Multi-Vector Distance Functions | InReview | Suryansh Gupta | - |
