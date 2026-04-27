# Roadmap to Production-Grade

This document inventories everything that would need to be done to take
`imgfprint` from "0.4.0, useful and fully tunable on every weight axis"
to "production-grade SDK that you'd happily depend on at scale". Items
are grouped by area and tagged with rough effort and priority.

Legend:

- **Priority:** P0 = blocks "production-ready" claim, P1 = important but
  not blocking, P2 = nice-to-have, P3 = research-grade
- **Effort:** S = hours, M = days, L = weeks, XL = months

---

## 1. Configurability Completeness

The 0.4.0 release tuned every weight and decode-time guard. What's left
are the structural knobs that need buffer surgery.

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 1.1 | **Per-algorithm hash dimensions** — `PHashConfig { dct_size, hash_bits }`, `AHashConfig { block_grid, hash_bits }`, `DHashConfig { direction, hash_bits }` | P1 | L | Requires turning the static `[f32; 32*32]` and `[[f32; 64*64]; 16]` buffers into runtime-sized vectors through every hash function and the preprocessor. The cleanest path is `const N: usize` generics with a small set of supported sizes (32, 48, 64) plus a runtime fallback. |
| 1.2 | **Algorithm subsetting at compute time** — skip AHash/DHash entirely when only PHash is wanted | P2 | M | Today setting `ahash_weight = 0.0` in `MultiHashConfig` zeros it out at scoring time, but the hash is still computed. Restructure `MultiHashFingerprint` to use `Option<ImageFingerprint>` per slot. Breaking change → 0.5.0. |
| 1.3 | **Custom block grid layout** — `4x4`, `8x8`, `1x1+overlap`, etc. | P2 | L | Tied to 1.1; once the grid size isn't fixed, expose layout as a config field. Useful for very tall or very wide images where 4×4 is a poor partition. |
| 1.4 | **Crop-detection threshold tuning** — exposed already as `block_distance_threshold`, but lacks a guide for picking it | P2 | S | Documentation + a calibration helper that sweeps thresholds against a labelled corpus. |
| 1.5 | **Pluggable preprocess pipeline** — let callers swap the resize filter, the colour-conversion path, the EXIF handler | P3 | L | Currently hardcoded: `fast_image_resize` Lanczos3 + standard luminance conversion. A trait-based pipeline lets specialists (medical imaging, satellite, screenshots) inject domain-specific normalisation. |

---

## 2. Major Features

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 2.1 | **First-class semantic alignment models** — built-in `EmbeddingProvider` impls for ImageBind, OpenCLIP, SigLIP behind feature flags | P0 | M | Today only the generic `LocalProvider` (ONNX) exists. UCFP needs aligned models for cross-modal (image↔text↔audio) retrieval; making them first-class lowers the integration tax. |
| 2.2 | **Watermark detection** (mirroring `audiofp::watermark`) — content-provenance via embedded marks (e.g., C2PA, Stable Signature) | P1 | M | Image watermark detection is increasingly required for AI-generated content compliance. Similar tract-onnx wrapper pattern. |
| 2.3 | **Animated formats first-frame extraction** — GIF, animated WebP, APNG | P1 | S | Today the `image` crate decodes the first frame implicitly; surface it as an explicit option and add a "fingerprint per frame" mode for animation deduplication. |
| 2.4 | **Streaming / partial fingerprinting** — fingerprint very large images (gigapixel, satellite) without loading the full bitmap into RAM | P2 | L | Region-based incremental hash computation; mirrors `audiofp`'s `StreamingFingerprinter` pattern. |
| 2.5 | **Tile / region fingerprinting** — return per-tile fingerprints for large images, enabling sub-image matching ("does this crop appear inside that picture?") | P2 | M | Generalisation of the existing 4×4 block hashes to a configurable tile grid + per-tile lookup index. |
| 2.6 | **Manipulation / forgery detection** — splice / copy-move / inpainting detection | P3 | XL | Adjacent problem to fingerprinting; would justify a separate crate but shares the preprocessor. |
| 2.7 | **Color fingerprint** — palette / colour-distribution hash for "find images with similar colour scheme" workflows | P2 | M | Complements the structure-only PHash/DHash. |
| 2.8 | **Wavelet hash (wHash)** as a fourth algorithm | P2 | M | Catches frequency-domain similarity that DCT misses. Slot into the existing `MultiHashFingerprint` once 1.2 lands. |

---

## 3. Performance & Scale

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 3.1 | **Hand-tuned SIMD for DCT** — `realfft` covers FFT but the 32×32 DCT for PHash is scalar | P1 | M | AVX2/NEON DCT-II would close a measurable gap on `compute_phash`. |
| 3.2 | **GPU batch fingerprinting** via `wgpu` for catalog-scale enrollment | P2 | L | A 100M-image ingestion run benefits from GPU offload of resize → grayscale → DCT. |
| 3.3 | **Async API surface** for non-blocking decode + fingerprint | P2 | M | Currently every API is sync; async would integrate cleanly with `tokio` users (and most ingest pipelines are tokio-based). |
| 3.4 | **Memory-mapped fingerprint storage** helpers | P2 | S | `ImageFingerprint` and `MultiHashFingerprint` are already POD-shaped; expose `bytemuck::Pod` impls + `Vec<Fingerprint> ↔ &[u8]` helpers for zero-copy persistence. |
| 3.5 | **Streaming decode for huge JPEG/PNG** without full pixel-buffer materialisation | P2 | M | Today `image::load_from_memory` allocates the entire decoded bitmap; on-demand row decoding would reduce peak RSS for gigapixel inputs. |
| 3.6 | **`fingerprint_batch_chunked` parallelism** — confirm rayon worker count, optionally let the caller pin a thread pool | P1 | S | Today rayon picks its own pool; high-throughput services may want to share one. |
| 3.7 | **Per-platform tuning profiles** for Apple Silicon, x86_64 AVX2/AVX-512, ARMv8 | P2 | M | Conditional compilation for the resize / DCT kernels. |
| 3.8 | **Block-hash parallelism granularity** — measure whether the rayon overhead on 16 small blocks dominates the work | P2 | S | A simple SIMD loop might beat rayon for the per-block hash phase. |

---

## 4. Robustness & Testing

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 4.1 | **Quantitative robustness corpus** with held-out CC0 images | P0 | M + corpus | Synthetic transforms (crop, rotate, blur, resize, JPEG quality, watermark, colour shift) over a labelled corpus, replacing the unit-style tests with measured "≥ 90 % similarity at JPEG q=50" claims. Mirrors `audiofp`'s deferred 4.1. |
| 4.2 | **Adversarial-input stress** — pathological PNGs (tiny, all-zero, all-NaN-after-decode), zip-bomb GIFs, malformed EXIF, polyglot JPEGs | P0 | S–M | Verify no panics, no quadratic slowdowns, correct bounded memory. The existing `fuzz/` harness covers the decode boundary; expand to the full pipeline. |
| 4.3 | **`cargo-fuzz` coverage expansion** — beyond decode, fuzz `compare`, `compare_with_config`, `is_similar` for any panics on adversarial fingerprint data | P1 | S | Would catch e.g. NaN propagation through user-supplied weights. |
| 4.4 | **Cross-platform CI matrix** — Linux + macOS + Windows | P0 | S | Confirm whether CI runs only on `ubuntu-latest` and expand. |
| 4.5 | **MSRV CI job** alongside stable | P1 | S | Run the test suite on the pinned MSRV (1.70) to catch accidental usage of newer-stable features. |
| 4.6 | **`miri` interpreter run** | P1 | M | Catches `unsafe` bugs in `bytemuck` / SIMD paths. Image preprocessor likely uses unsafe internally via `fast_image_resize`. |
| 4.7 | **Mutation testing** (`cargo-mutants`) | P2 | M | Measures whether the 211 lib tests actually catch behavioural regressions. |
| 4.8 | **Code coverage tracked over time** | P1 | S | A coverage badge + per-PR coverage diff via `cargo-llvm-cov` or Codecov. |
| 4.9 | **Snapshot / regression goldens** for fingerprints over a small image corpus | P0 | S | Frozen `(image_bytes, expected_fingerprint)` pairs in a binary blob; any algorithm tweak that changes them is a deliberate breaking change. Already in spirit via the determinism tests, but corpus-backed goldens are stronger. |
| 4.10 | **Big-endian + 32-bit target tests** | P2 | S | Probably "just works" but worth verifying — fingerprint serialisation is the highest-risk path. |
| 4.11 | **Concurrency safety static-assertions** — `Send + Sync` boundaries we promise | P1 | S | Mirror what `audiofp` did with `static_assertions`. |
| 4.12 | **Property tests for `MultiHashConfig` weight invariants** | P1 | S | E.g., score is always in `[0.0, 1.0]` regardless of weights; setting all weights to zero on different fingerprints returns score 0. Currently covered by 4 spot tests; proptest would generalise. |

---

## 5. Documentation & Developer Experience

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 5.1 | **mdBook documentation site** with tutorials | P1 | M | Per-algorithm deep-dives, end-to-end deduplication tutorial, UCFP integration recipe. Hosted on GitHub Pages. |
| 5.2 | **`MultiHashConfig` tuning guide** — concrete recipes for "screenshot dedup", "stock photo dedup", "logo matching", "satellite imagery" with the weights and thresholds that work | P1 | S–M | The new config is powerful but sets users up for cargo-culting defaults. A recipe book is the single highest-leverage doc work right now. |
| 5.3 | **Interactive WASM playground** | P2 | M | Drag-and-drop two images, see fingerprints + similarity computed live. Best discovery tool. |
| 5.4 | **Comparison with `imghash` / `image-hasher` / Python `imagehash`** with reproducible benchmarks | P1 | M | Quantitative head-to-head on the same corpus and same hash dimension. |
| 5.5 | **Migration guide 0.3 → 0.4** explaining the new config knobs | P1 | S | Explicit "what changed and what defaults reproduce 0.3 behaviour" — already implicit in the CHANGELOG, but a dedicated MIGRATION.md is more discoverable. |
| 5.6 | **`docs.rs` examples module** that compiles | P1 | S | Per-algorithm example modules visible in the docs.rs sidebar. |
| 5.7 | **Algorithm whitepapers** linked to readable summaries | P2 | S | Pointers to original PHash/DHash/AHash papers + plain-English summaries of what each captures and where they fail. |
| 5.8 | **API stability policy** in CONTRIBUTING | P0 | S | What "0.x → 1.0" means, deprecation policy, MSRV bumps. |
| 5.9 | **Plain-English error messages** for common misuses | P2 | S | Already good; one review pass. |
| 5.10 | **`cargo-deadlinks` in CI** | P2 | S | Catches broken intra-doc links before they ship. |

---

## 6. Format & Codec Support

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 6.1 | **HEIC / HEIF support** (Apple's default) | P1 | M | Requires a libheif binding (e.g., `libheif-rs`) behind a feature flag because the C library is non-trivial to ship. |
| 6.2 | **AVIF support** | P1 | S | The `image` crate has experimental AVIF decode; surface it behind a feature flag. |
| 6.3 | **TIFF support** | P2 | S | `image` decodes TIFF; just enable the feature flag and the format detection list. Adds reach into archival / scientific pipelines. |
| 6.4 | **PSD / PSB support** | P3 | M | Useful for DAM (digital asset management) integrations. Likely a third-party crate dependency. |
| 6.5 | **Raw camera formats** (CR2, NEF, ARW, DNG) | P3 | L | Via `rawler` or similar. Niche but high-value for photo workflow tools. |
| 6.6 | **Format-detection mismatch resilience** — handle files with wrong extensions / spoofed magic bytes | P1 | S | Currently `with_guessed_format` handles most cases; add explicit tests for adversarial format spoofing. |

---

## 7. UCFP Integration

These items only matter if `ucfp` (or another integrator crate) consumes
`imgfprint` directly.

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 7.1 | **YAML → config mapping** in UCFP — surface `MultiHashConfig` and `PreprocessConfig` fields in `server.yaml` | P0 | S | Trivial wiring on the UCFP side; documented schema needs to be added there. Lives in the UCFP repo, not here. |
| 7.2 | **`imgfprint::Perceptual` adapter** — implement whatever trait UCFP's `perceptual` crate expects, so UCFP imports `imgfprint` directly without a wrapper crate | P0 | S–M | Confirms the per-modality SDK / integrator split actually works in practice. |
| 7.3 | **Cross-modal embedding registry** — agreed `model_id` scheme across `imgfprint` / `txtfp` / `audiofp` / `vidfp` so UCFP can verify alignment before computing cosine similarity | P1 | M | The `Embedding { model_id }` field exists; needs a documented registry of names (e.g., `"imagebind-v1"`) and a sanity-check helper. |
| 7.4 | **Batch enroll / query helpers** that match UCFP's ingestion shape | P1 | S | Mostly already exists via `fingerprint_batch_chunked`; confirm the callback signature and error semantics line up with what UCFP needs. |
| 7.5 | **Quantization helpers** for fingerprint persistence in UCFP's index — fixed-bit packing of `MultiHashFingerprint` | P2 | M | UCFP's `index` crate already supports compression / quantization; document or expose imgfprint-side serializers that fit cleanly. |
| 7.6 | **Schema versioning** — fingerprint format version field so UCFP can refuse to compare across incompatible schemas | P1 | S | Currently the structural shape is implicit. A version constant + serde tag makes future format changes safe. |

---

## 8. Release Engineering

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 8.1 | **Tag and publish 0.4.0** | P0 | S | Code is ready; awaiting commit + tag + `cargo publish`. |
| 8.2 | **Cross-platform release matrix in CI** — build artifacts on Linux / macOS / Windows | P1 | S | Confirms publish-readiness for the next release. |
| 8.3 | **`cargo-msrv check` in CI** | P1 | S | Verifies the declared MSRV (1.70) actually compiles. |
| 8.4 | **`cargo-deny` in CI** — license + advisory checks | P1 | S | Particularly relevant before publishing; catches incompatible dependency licenses. |
| 8.5 | **Reproducible-build verification** — same input → same crate hash | P2 | S | Useful for supply-chain auditability. |
| 8.6 | **Release-notes automation** from CHANGELOG | P2 | S | A small workflow that opens the GitHub release with the relevant CHANGELOG section. |

---

## 9. Things That Are Already Done in 0.4.0

For reference, so future contributors don't re-litigate them:

- ✅ Tunable inter-algorithm weights (`MultiHashConfig.{ahash,phash,dhash}_weight`)
- ✅ Tunable intra-algorithm global/block split (`MultiHashConfig.{global,block}_weight`)
- ✅ Tunable block-distance threshold (`MultiHashConfig.block_distance_threshold`)
- ✅ Tunable decode-time guards (`PreprocessConfig.{max_input_bytes,max_dimension,min_dimension}`)
- ✅ Path-API and bytes-API gated by the same `PreprocessConfig` (no silent bypass)
- ✅ All previous APIs preserved as default-config wrappers (zero behaviour change)
- ✅ `compute_similarity_with_weights` exposed for raw `ImageFingerprint` callers
- ✅ `Hash` derive on fingerprint types (HashSet/HashMap usable)
- ✅ `fingerprint_path` / `fingerprint_path_with_*` convenience methods
- ✅ `ImgFprintError::IoError` variant for file-read failures
- ✅ POPCNT-backed Hamming distance
- ✅ Cached ONNX `RunnableModel` (eliminates per-inference clone overhead)
- ✅ rayon `map_init` per-worker context caching in batch APIs
- ✅ NaN/infinity threshold clamping in `is_similar`
- ✅ Single decode pass (no double-decode for dimension check)
- ✅ Post-EXIF dimension re-validation
- ✅ Bilinear resample identity-case fast path
- ✅ Generic `EmbeddingProvider` trait + `Embedding { vector, model_id }` for cross-modal alignment

---

## How to use this list

- **P0 items** are what gates a "production-grade" claim. Pick these first.
- **P1 items** make the SDK pleasant; ship them before declaring 1.0.
- **P2 / P3 items** are good ideas to keep visible but not committed.
- Effort tags are rough — treat them as "order of magnitude" not estimates.
- When you start an item, open a tracking issue and link it here, so this
  file stays the source of truth for "what's left".
