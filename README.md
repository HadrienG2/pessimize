# pessimize - More efficient Rust compiler optimization barriers

[![MPL licensed](https://img.shields.io/badge/license-MPL-blue.svg)](./LICENSE)
[![Package on crates.io](https://img.shields.io/crates/v/pessimize.svg)](https://crates.io/crates/pessimize)
[![Documentation](https://docs.rs/pessimize/badge.svg)](https://docs.rs/pessimize/)
[![Continuous
Integration](https://img.shields.io/github/actions/workflow/status/HadrienG2/pessimize/ci.yml?branch=master)](https://github.com/HadrienG2/pessimize/actions?query=workflow%3A%22Continuous+Integration%22)
![Requires rustc
1.79.0+](https://img.shields.io/badge/rustc-1.79.0+-lightgray.svg)

Microbenchmarking is a subtle exercise to begin with, and the lack of
lightweight optimization barriers on stable Rust makes it even more difficult.
This crate aims to improve upon the statu quo by providing an alternative to
`std::hint::black_box` which is...

- As lightweight as a library-based approach can be
- Fine-grained, enabling precise intent statement and thus lower overhead
- Reliable (no more "this might do nothing on some compiler targets", if it
  compiles then it should work as advertized).

In exchange, the price to pay is...

- Reduced microbenchmark portability (currently 32-bit and 64-bit flavors of
  ARM, x86 and RISC-V, more can be envisioned on nightly / future stable Rust)
- Lots of unsafe code in the implementation (we often need to convert an `std`
  type with safety invariants to a simpler data structure with less safety
  invariants, pass the members of this data structure through an optimization
  barrier that does nothing, then convert it back to the original type).
