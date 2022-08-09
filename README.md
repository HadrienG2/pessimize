# pessimize - Compiler optimization barriers for stable Rust

Microbenchmarking is a subtle exercise to begin with, and the lack of stable and
lightweight optimization barriers on stable Rust makes it even more difficult.
This crate aims to improve upon the statu quo by providing an alternative to
`std::hint::black_box` which is...

- Available on stable Rust
- As lightweight as a library-based approach can be
- Fine-grained, enabling precise intent statement and thus lower overhead
- Reliable (no more "this might do nothing on some compiler targets", if it
  compiles then it should work as advertized).

In exchange, the price to pay is...

- Reduced microbenchmark portability (currently 32-bit and 64-bit flavors of
  ARM, x86 and RISC-V, more can be envisioned on nightly / future stable Rust)
- Lots of unsafe code in the implementation (we often need to convert an `std`
  type with safety invariants to a simpler type with less safety invariants,
  pass it through an optimization barrier that does nothing, then convert it
  back to the original type).
