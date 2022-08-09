# pessimize - Compiler optimization barriers for stable Rust

Microbenchmarking is a subtle exercise to begin with, and the lack of stable and
lightweight optimization barriers on stable Rust makes it even more difficult.
This crate aims to improve upon the statu quo by providing an alternative to
`std::hint::black_box` which is...

- Available on stable Rust
- As lightweight as a library-based approach can be
- Fine-grained, enabling precise intent statement and thus lower overhead
- Reliable (no more "this might do nothing on some compiler targets", if it
  compiles then it should work as advertiszed).
