# Quality_control_pipeline
Downloads public proteomics datasets from PRIDE, converts raw files to mzML and computes 81 ID free QC metrics exported as mzQC v1.0.0 files.

List of the metrics can be found here:

https://docs.google.com/spreadsheets/d/1F8sB_X-BrMjBR4qKguJSscKVq-VgFH1NGMXitptHvMg/edit?usp=sharing

More detailed list of New metrics can be found here:


---

## Installation
```bash
pip install scipy numpy pandas pyteomics lxml pymzml joblib tqdm requests pyopenms
```
For RAW file conversion you also need:
- **ThermoRawFileParser + Mono** for Thermo `.raw` files

---

## Usage
Run the four steps in order:
```bash
python3 convert.py       #Convert RAW -> mzML (skip if already mzML)
python3 QC_pipeline.py   #Compute QC metrics
```

---

## What Each Script Does
| Script | What it does |
|---|---|
| `convert.py` | Converts Thermo RAW files to mzML using ThermoRawFileParser + Mono |
| `QC_pipeline.py` | Reads all MS files, computes 81 QC metrics, writes `.mzQC` and `.tsv` outputs |

---

## Outputs
```
results/
├── tsv/
│   ├── all_metrics.tsv          #61 metrics per file
│   └── summary_by_dataset.tsv   #mean/std/min/max per dataset
└── mzqc/
    └── <dataset>/<file>.mzQC    #mzQC v1.0.0 per file
```

---

## Supported File Formats
| Format | Handled by |
|---|---|
| `.mzML`, `.mzXML`, `.mgf` | Read directly |
| Thermo `.raw` | ThermoRawFileParser + Mono |
