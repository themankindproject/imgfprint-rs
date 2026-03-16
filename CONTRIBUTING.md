# Contributing to imgfprint-rs

## Versioning & Release Policy

This crate follows [Semantic Versioning](https://semver.org/) (SemVer).

### Version Number Format: MAJOR.MINOR.PATCH

- **MAJOR** (x.0.0): Breaking changes to the public API
- **MINOR** (0.x.0): New functionality (backward compatible)
- **PATCH** (0.0.x): Bug fixes (backward compatible)

### API Stability

- All public APIs are considered stable unless marked with `#[unstable]`
- Re-exports from `pub use` statements are part of the public API
- Struct fields marked with `pub` are part of the public API

### Deprecation Policy

1. Deprecated features will remain for at least **one minor version** before removal
2. Deprecation warnings will be emitted via `#[deprecated]` attribute
3. Migration instructions will be provided in the deprecation message

### What Constitutes a Breaking Change

- Removing or renaming public functions, types, or constants
- Changing function signatures
- Changing return types (even to more general types)
- Changing enum variant names or adding non-exhaustive enums
- Changing behavior of existing functions (unless documented as undefined)
- Removing `#[must_use]` attributes

### What is NOT a Breaking Change

- Adding new public functions or types
- Adding new enum variants to non-`#[non_exhaustive]` enums
- Adding new optional parameters with defaults
- Documentation changes
- Internal implementation changes
- Adding new error variants to `ImgFprintError` (it's `#[non_exhaustive]`)

### MSRV (Minimum Supported Rust Version)

- Current MSRV: **1.70**
- MSRV may be bumped in minor versions with advance notice
- New features that require newer Rust will be feature-gated

## Development

```bash
# Run tests
cargo test

# Run with all features
cargo test --all-features

# Run clippy
cargo clippy --all-targets

# Run benchmarks
cargo bench
```

## Pull Requests

1. Ensure all tests pass
2. Run `cargo clippy` and fix warnings
3. Update documentation if needed
4. Add tests for new functionality
