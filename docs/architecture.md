# architecture.md

# data-says-lot — Architecture

This document describes the high-level architecture, module responsibilities, dataflow, deployment recommendations, and extension points for the `data_says_lot` single-file forensic toolkit. It is written to help maintainers, contributors, and reviewers understand design decisions and where to extend the project into a production-ready package.

---

## 1. Overview

`data_says_lot` is a compact offline forensic triage tool that performs metadata extraction, static heuristics, ML-based steganography anomaly detection, optional sandbox orchestration helpers, and automated PDF reporting with embedded charts. The primary design goals are:

* **Offline-first:** No external API calls required. All analyses run locally.
* **Auditable outputs:** Human-readable PDF and machine-readable JSON appendices.
* **Extensible:** Clear places to add YARA, pHash, DCT/RS stego detectors, or remote indexing.
* **Portable single-file delivery:** Useful to run quickly on an investigator's host.

---

## 2. Logical components

The code is organized conceptually into the following logical components (implemented as classes/functions in the single file):

* **Ingest** — Recursively discover files, normalize paths, and classify by type (image, video, pdf, unknown). Handles symlink avoidance and deterministic ordering.

* **MetadataExtractor** — Extracts metadata per file type:

  * Images: EXIF tags, dimensions, generate thumbnails (Pillow + exifread).
  * PDFs: metadata, page count, raw stream scanning for `/JavaScript`, `/EmbeddedFile`, and `%%EOF` anomalies (PyPDF2 + raw scan).
  * Videos: invokes `ffprobe` (if installed) to obtain container/stream metadata.

* **Analyzer** — Static heuristics and ML features:

  * Appended-data detection (PK, MZ, ELF, TAR markers), embedded-executable detection.
  * Image metrics: LSB ratio, grayscale entropy, noise score (median-filter diff).
  * ML: `IsolationForest` model training (`train_stego_model`) and prediction (`predict_stego`).

* **Report** — PDF generator using ReportLab:

  * Builds a Times-Roman styled PDF with a cover, per-file pages, embedded thumbnails, charts (matplotlib), and a JSON appendix.
  * Chart generation is separated: Matplotlib produces PNGs which the report embeds.

* **Sandbox helper** — `run_in_sandbox` wrapper that uses `firejail` (if available) for basic dynamic execution; otherwise, it returns guidance to use VM/QEMU.

* **Orchestration & CLI** — command-line parsing, workflows for training, analyzing, testing, benchmarking, and report production.

---

## 3. Dataflow

1. User invokes CLI with input path and options.
2. `Ingest` discovers files → list of absolute paths.
3. For each path, `MetadataExtractor` extracts metadata and creates thumbnails as appropriate.
4. `Analyzer` computes heuristics and features; if model loaded, it runs ML prediction.
5. Aggregation: orchestration collects per-file outputs into a top-level results structure.
6. Charting: aggregation step can compute distributions (e.g., LSB histogram) via Matplotlib.
7. `Report` generates the PDF (cover → per-file pages → charts → JSON appendix) and writes optional JSON to disk.

All intermediary outputs (thumbnails, chart PNGs, JSON) are written to the working directory by default so they can be packaged into an evidence bundle.

---

## 4. Persistence & Evidence bundle

A recommended evidence bundle contains:

* Original file(s) (or a copy) with computed `sha256` and `md5` in `manifest.json`.
* `report.pdf` (human report).
* `analysis.json` (machine-readable per-file output).
* `charts/` (all PNG charts).
* `model/` (optional trained `*.joblib` model).
* `README.metadata` (short, human-readable manifest of commands run and environment).

Sign the bundle with GPG for chain-of-custody where required.

---

## 5. Extension points

* **YARA integration:** Add a `yara_scan()` method in `Analyzer` and a CLI flag to point to a local rules directory.
* **Reverse-image matching / pHash:** Add a `phash_index` module and a background indexer; integrate results into the `Report` per-file section.
* **DCT/RDWT-based stego checks:** Replace or augment `feature_vector()` with DCT coefficients and RS-statistics for JPEGs.
* **Dynamic analysis pipeline:** Create a `dynamic/` submodule that orchestrates VM snapshots, network capture, and evidence extraction.
* **Parallelization:** Add a `worker_pool` that parallelizes `extractor.image_meta()` and `analyzer.feature_vector()` across CPU cores.

---

## 6. Security and privacy considerations

* Treat generated JSON and PDF as sensitive evidence — they may contain GPS or personal data.
* The tool does not execute arbitrary untrusted code; the sandbox helper is only a convenience. Use a dedicated VM for real dynamic analysis.
* Limit file permissions on generated artifacts; ensure secure storage and transport.

---

## 7. Runtime requirements & environment

* **Python:** 3.8+ recommended.
* **Pip packages:** Pillow, exifread, PyPDF2, reportlab, numpy, scikit-learn, matplotlib, joblib.
* **Optional:** `ffmpeg` (ffprobe) for video metadata; `firejail` for sandbox helper.
* **OS:** Linux recommended for sandboxing and typical forensic workflows, but metadata extraction and report generation work cross-platform.

---

## 8. Testing & CI suggestions

* Add unit tests for: EXIF parsing, LSB calculations (with fixtures), JSON serialization, and basic CLI flows.
* Include integration tests that run the CLI on a small `samples/` set and validate output JSON fields.
* Use CI to run linting, unit tests, and coverage; keep heavy tests as optional matrix entries or artifacts.

---

## 9. Performance & scaling notes

* Image feature extraction is CPU- and memory-bound for large images; consider streaming processing for very large files.
* For very large corpora, use multiprocessing or chunk the corpus to avoid memory pressure when building ML feature arrays.
* Limit concurrent `ffprobe` calls to avoid IO saturation.

---

---

