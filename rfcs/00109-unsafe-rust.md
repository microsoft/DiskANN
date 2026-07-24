# Unsafe code in DiskANN

At DiskANN, **security is our top priority**.

We strive to use **safe Rust** wherever possible. Introducing `unsafe` code can open the door to potential security vulnerabilities, so it should be avoided unless absolutely necessary.

That said, there are cases where `unsafe` code is justified:

* **Performance-critical paths** where there is **strong evidence** that the performance benefits clearly outweigh the risks. Recommended threshold: 10%+ performance improvement.
*  **Non-performance-related cases** usually involving communication or interactions beyond Rust’s safety boundaries:
    * Interfacing with **FFI** (foreign function interfaces)
    * Calling **external commands or libraries**
    * Performing **low-level pointer or memory allocation manipulation**, when there are no safe alternatives
    * Handling **platform-specific behavior** that cannot be safely abstracted

To ensure transparency and maintain trust in the codebase, any `unsafe` code introduced for **performance reasons** must:

*   Be wrapped in a safe API that enforces normal Rust guarantees at the point of use.
*   Be backed by an **automated, reproducible benchmark** integrated into the CI pipeline. This benchmark serves as a **quantitative justification** for the `unsafe` block, demonstrating the performance gains. It also prevents accidental or uninformed reversion to a safe (but slower) implementation.

For non-performance-related uses of `unsafe`, the rationale must be **clearly documented in the code**, helping reviewers and future maintainers understand the necessity and ensuring the continued trustworthiness of the DiskANN library.