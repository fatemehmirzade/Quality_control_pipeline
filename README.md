# Quality_control_pipeline

Downloads public proteomics datasets from PRIDE, converts raw files to mzML and computes 61 ID free QC metrics exported as mzQC v1.0.0 files.

List of the ID free metrics can be found here:

https://docs.google.com/spreadsheets/d/1F8sB_X-BrMjBR4qKguJSscKVq-VgFH1NGMXitptHvMg/edit?usp=sharing

---

## Installation

```bash
pip install scipy numpy pandas pyteomics lxml pymzml joblib tqdm requests pyopenms
```

For RAW file conversion, you also need one of:
- **ThermoRawFileParser + Mono** — for Thermo `.raw` files
- **Docker + msconvert** — for Waters/Bruker files

---

## Usage

Run the four steps in order:

```bash
python3 data.py          # Search PRIDE and save dataset list
python3 download.py      # Download files from PRIDE
python3 convert.py       # Convert RAW -> mzML (skip if already mzML)
python3 QC_pipeline.py   # Compute QC metrics
```

---

## What Each Script Does

| Script | What it does |
|---|---|
| `data.py` | Queries the PRIDE REST API and saves up to 50 diverse datasets to `pride_datasets.json` |
| `download.py` | Downloads mzML/mzXML/MGF files to `data/`, and RAW files to `data_raw/` |
| `convert.py` | Converts RAW files to mzML using ThermoRawFileParser (Mono) or msconvert (Docker) |
| `QC_pipeline.py` | Reads all MS files, computes 61 QC metrics, writes `.mzQC` and `.tsv` outputs |

---

## Outputs

```
results/
├── tsv/
│   ├── all_metrics.tsv          # 61 metrics per file
│   └── summary_by_dataset.tsv   # mean/std/min/max per dataset
└── mzqc/
    └── <dataset>/<file>.mzQC    # mzQC v1.0.0 per file
```

---

## Supported File Formats

| Format | Handled by |
|---|---|
| `.mzML`, `.mzXML`, `.mgf` | Read directly — no conversion needed |
| Thermo `.raw` | ThermoRawFileParser + Mono |
| Waters/Bruker | msconvert via Docker |
