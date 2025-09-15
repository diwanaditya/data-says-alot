# data-says-lot

[![GitHub Repo](https://img.shields.io/badge/github-diwanaditya/data-says-lot-blue)](https://github.com/diwanaditya/data-says-alot)  
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)  
[![Last Commit](https://img.shields.io/github/last-commit/diwanaditya/data-says-lot)](https://github.com/diwanaditya/data-says-alot/commits/main)  
[![Stars](https://img.shields.io/github/stars/diwanaditya/data-says-lot)](https://github.com/diwanaditya/data-says-alot/stargazers)  
[![Issues](https://img.shields.io/github/issues/diwanaditya/data-says-lot)](https://github.com/diwanaditya/data-says-alot/issues)  

---

## 🔹 Project Overview

`data-says-lot` is a **compact forensic triage toolkit** designed for analyzing files, detecting anomalies, and generating professional reports. It works with **images, videos, and PDFs**, performing **metadata extraction, static heuristics, ML-based steganography detection**, and **optional sandboxed dynamic analysis**.

**Highlights:**

- Offline-first, no external API calls required  
- Human-readable PDF reports + machine-readable JSON outputs  
- ML-based anomaly detection for hidden/stego data  
- Charts and thumbnails for visual reporting  
- Optional sandbox helper (`firejail`) for safe dynamic analysis  

---

## 🔹 Supported File Types

| Type    | Extensions           | Features |
|---------|--------------------|----------|
| Image   | .jpg, .jpeg, .png, .heic, .webp, .tiff | EXIF metadata, LSB ratio, entropy, noise, stego detection |
| Video   | .mp4, .mov, .mkv, .avi, .webm | Metadata via ffprobe |
| PDF     | .pdf               | Metadata, page count, embedded scripts/files |
| Other   | Any                 | File size, hash, appended data checks |

---

## 🔹 How It Works

1. **Ingest:** Scans files/folders, classifies by type  
2. **Metadata Extraction:** Extracts EXIF, PDF properties, video info  
3. **Analysis:** Runs static heuristics + ML anomaly detection  
4. **Aggregation:** Combines per-file results  
5. **Reporting:** Generates PDF reports with charts and thumbnails, JSON for automation  

---

## 🔹 Installation

### Prerequisites

- Python 3.8+  
- Linux / Windows / macOS supported (sandbox helper works best on Linux)

### Steps

1. Clone the repo:

```
git clone https://github.com/diwanaditya/data-says-lot.git
cd data-says-lot

```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Optional tools:

```
sudo apt install ffmpeg firejail
```

---

## 🔹 Usage

### Analyze folder & generate PDF + JSON

```
python3 data_says_lot.py -i /path/to/files -o report.pdf --json out.json
```

### Train ML model on clean images

```
python3 data_says_lot.py --train-clean /path/to/clean/images
```

### Use trained ML model for detection

```
python3 data_says_lot.py -i /path/to/files -o report.pdf --model clean_images.joblib --json out.json
```

### Run tests

```
python3 data_says_lot.py --run-tests
```

### Benchmark analysis performance

```
python3 data_says_lot.py -i /path/to/files --benchmark
```

---

## 🔹 CLI Options

| Option        | Description                          |
| ------------- | ------------------------------------ |
| -i, --input   | Input file/folder path               |
| -o, --output  | Output PDF report path               |
| --json        | Save JSON analysis                   |
| --train-clean | Train stego ML model on clean images |
| --model       | Load existing ML model (`.joblib`)   |
| --run-tests   | Run unit/integration tests           |
| --benchmark   | Run performance benchmark            |

---

## 🔹 Output

* **PDF Report:** Human-readable report with charts, thumbnails, and per-file analysis
* **JSON Report:** Machine-readable per-file analysis
* **Charts folder:** PNG charts generated for reports
* **Optional ML Model:** `.joblib` trained model file

---

## 🔹 Architecture

* **Ingest:** File discovery, classification, path normalization
* **MetadataExtractor:** Per-type metadata extraction (images, PDFs, videos)
* **Analyzer:** Static heuristics + ML steganography detection
* **Report Generator:** PDF using ReportLab + Matplotlib charts
* **Sandbox Helper:** Optional firejail wrapper for dynamic analysis
* **Orchestration & CLI:** Command-line workflows for training, analyzing, testing, benchmarking, report generation

---

## 🔹 Security & Privacy

* Handle generated JSON/PDF carefully; may contain GPS/EXIF or personal info
* Sandbox helper is optional; use VM/QEMU for full security
* Limit permissions on generated artifacts

---

## 🔹 References

* [ExifRead Documentation](https://pypi.org/project/ExifRead/)
* [FFmpeg / FFprobe Documentation](https://ffmpeg.org/)
* [ReportLab User Guide](https://www.reportlab.com/docs/reportlab-userguide.pdf)
* [scikit-learn IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
* Steganalysis literature

---

[![GitHub Repo](https://img.shields.io/badge/github-diwanaditya/data-says-lot-blue)](https://github.com/diwanaditya/data-says-lot)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

```

 **This is fully complete**—no missing sections. You can copy-paste this directly as your `README.md`.  

---

If you want, I can **also add a “PRO version” README** with **diagrams for architecture, visual workflow, and output example screenshots**—this makes it look *GitHub-repo-level elite*.  

Do you want me to make that visual version too?
```
