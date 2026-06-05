import json
import csv
import time
import datetime
import warnings
import logging
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.stats import entropy as scipy_entropy
from pyteomics import mzml, mzxml

warnings.filterwarnings("ignore")
logging.getLogger("pyteomics").setLevel(logging.ERROR)
logging.getLogger("lxml").setLevel(logging.ERROR)

for _noisy in ("pyteomics.mzml", "pyteomics.mzxml", "pyteomics.auxiliary",
               "pyteomics.xml"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

try:
    import pyopenms as _oms
    PYOPENMS_AVAILABLE = True
except ImportError:
    PYOPENMS_AVAILABLE = False

DATA_DIR    = Path("./data")
RESULTS_DIR = Path("./results")
TSV_DIR     = RESULTS_DIR / "tsv"
MZQC_DIR    = RESULTS_DIR / "mzqc"

for _d in [TSV_DIR, MZQC_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

MS_EXTENSIONS = {".mzml", ".mzxml", ".mgf"}


#mzQC document builder

def build_mzqc(metrics, file_path, dataset):
    now   = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    fname = Path(file_path).name
    ext   = Path(file_path).suffix.lower()

    fmt_map = {
        ".mzml":  {"accession": "MS:1000564", "name": "PSI mzML file"},
        ".mzxml": {"accession": "MS:1000566", "name": "ISB mzXML file"},
        ".mgf":   {"accession": "MS:1001062", "name": "Mascot MGF file"},
    }
    fmt_cv = fmt_map.get(ext, {"accession": "MS:1000564", "name": "mzML file"})

    controlled_vocabularies = [
        {
            "name":    "Proteomics Standards Initiative Mass Spectrometry Ontology",
            "uri":     "https://github.com/HUPO-PSI/psi-ms-CV/releases/download/v4.1.130/psi-ms.obo",
            "version": "4.1.130",
        },
        {
            "name":    "Unit Ontology",
            "uri":     "https://raw.githubusercontent.com/bio-ontology-research-group/unit-ontology/master/unit.obo",
            "version": "09:04:2023",
        },
    ]

    input_file = {
        "name":     fname,
        "location": str(Path(file_path).resolve()),
        "fileFormat": {
            "accession": fmt_cv["accession"],
            "name":      fmt_cv["name"],
        },
        "fileProperties": [
            {"accession": "MS:1000031", "name": "instrument model", "value": "unknown"}
        ],
    }

    analysis_software = [
        {"cvParameter": {"accession": "MS:1003282", "name": "QC pipeline"}}
    ]

    #Scalar metrics
    METRIC_MAP = {
        "MS:4000059": ("MS:4000059", "number of MS1 spectra",
                       "Total number of MS1 survey scans", None, None),
        "MS:4000060": ("MS:4000060", "number of MS2 spectra",
                       "Total number of MS2 fragment scans", None, None),
        "MS:4000067": ("MS:4000067", "MS1 RT range",
                       "Total retention time span of the run", "UO:0000031", "minute"),
        "MS:4000053": ("MS:4000053", "chromatography duration",
                       "RT span of the active elution window in MS1", "UO:0000031", "minute"),
        "MS:4000029": ("MS:4000029", "area under TIC (MS1)",
                       "Trapezoidal area under the MS1 TIC curve", None, None),
        "MS:4000030": ("MS:4000030", "area under TIC (MS2)",
                       "Trapezoidal area under the MS2 TIC curve", None, None),
        "MS:4000031": ("MS:4000031", "MS1 to MS2 signal ratio",
                       "Ratio of MS1 AUC to MS2 AUC", None, None),
        "MS:4000065": ("MS:4000065", "MS1 fastest frequency",
                       "Maximum MS1 scan acquisition frequency", "UO:0000106", "hertz"),
        "MS:4000066": ("MS:4000066", "MS2 fastest frequency",
                       "Maximum MS2 scan acquisition frequency", "UO:0000106", "hertz"),
        "MS:4000095": ("MS:4000095", "MS1 slowest frequency",
                       "Minimum MS1 scan acquisition frequency", "UO:0000106", "hertz"),
        "MS:4000096": ("MS:4000096", "MS2 slowest frequency",
                       "Minimum MS2 scan acquisition frequency", "UO:0000106", "hertz"),
        "MS:4000097": ("MS:4000097", "MS1 signal jump count",
                       "Number of 10x intensity jumps in consecutive MS1 TICs", None, None),
        "MS:4000098": ("MS:4000098", "MS1 signal fall count",
                       "Number of 10x intensity falls in consecutive MS1 TICs", None, None),
        "MS:4000099": ("MS:4000099", "number of empty MS1 spectra",
                       "MS1 scans with zero total intensity", None, None),
        "MS:4000100": ("MS:4000100", "number of empty MS2 spectra",
                       "MS2 scans with zero total intensity", None, None),
        "MS:4000193": ("MS:4000193", "DIA isolation window median cycle time",
                       "Median time between successive scans of the same window",
                       "UO:0000031", "minute"),
        "MS:4000194": ("MS:4000194", "DIA isolation window count",
                       "Number of distinct isolation windows", None, None),
        "NEW-001": ("NEW:0000001", "acquisition mode",
                    "DDA / DIA / MS1-only / MS2-only / DDA-MS2only / DIA-MS2only",
                    None, None),
        "NEW-002": ("NEW:0000002", "MS level distribution",
                    "Count of scans at each MS level", None, None),
        "NEW-003": ("NEW:0000003", "scan polarity distribution",
                    "Count of positive, negative, and unknown polarity scans", None, None),
        "NEW-004": ("NEW:0000004", "median spectral entropy MS1",
                    "Median Shannon entropy of MS1 intensity distributions", None, None),
        "NEW-005": ("NEW:0000005", "median spectral entropy MS2",
                    "Median Shannon entropy of MS2 intensity distributions", None, None),
        "NEW-006": ("NEW:0000006", "fraction sparse MS2 spectra",
                    "Fraction of MS2 scans with fewer than 10 peaks", None, None),
        "NEW-007": ("NEW:0000007", "MS2 fragment dynamic range",
                    "Log10 ratio of 95th to 5th percentile MS2 TIC", None, None),
        "NEW-008": ("NEW:0000008", "fraction zero-intensity peaks",
                    "Fraction of all reported peaks with zero intensity", None, None),
        "NEW-009": ("NEW:0000009", "TIC coefficient of variation",
                    "CV of MS1 TIC values across the run", None, None),
        "NEW-010": ("NEW:0000010", "MS1 TIC Gini coefficient",
                    "Gini inequality coefficient of MS1 TIC distribution", None, None),
        "NEW-011": ("NEW:0000011", "baseline noise level",
                    "5th percentile of non-zero MS1 TIC values", None, None),
        "NEW-012": ("NEW:0000012", "signal-to-baseline ratio",
                    "Median MS1 TIC divided by 5th-percentile baseline", None, None),
        "NEW-013": ("NEW:0000013", "DDA isolation window width distribution",
                    "Q1/Q2/Q3 of precursor isolation window widths", "UO:0000221", "dalton"),
        "NEW-014": ("NEW:0000014", "precursor redundancy rate",
                    "Fraction of MS2 scans whose precursor m/z matches another scan within 0.01 Da",
                    None, None),
        "NEW-016": ("NEW:0000016", "DIA isolation window overlap fraction",
                    "Fraction of adjacent DIA windows that overlap in m/z", None, None),
        "NEW-017": ("NEW:0000017", "DIA m/z coverage fraction",
                    "Fraction of the total m/z range covered by DIA windows", None, None),
        "NEW-018": ("NEW:0000018", "DIA per-window TIC CV",
                    "Median CV of TIC across cycles for each DIA window", None, None),
        "NEW-019": ("NEW:0000019", "DIA cycle regularity",
                    "CV of inter-cycle times across DIA windows", None, None),
        "NEW-020": ("NEW:0000020", "DIA fragmentation coverage uniformity",
                    "min/max ratio of median peak counts across DIA windows", None, None),
        "NEW-021": ("NEW:0000021", "scan metadata completeness",
                    "Fraction of MS2 scans with all required metadata fields present",
                    None, None),
        "NEW-022": ("NEW:0000022", "centroid vs profile mode",
                    "Detected spectral mode: centroid, profile, mixed, or unknown",
                    None, None),
        "NEW-023": ("NEW:0000023", "precursor charge annotation rate",
                    "Fraction of MS2 scans with charge state annotated", None, None),
        "NEW-024": ("NEW:0000024", "injection time annotation rate",
                    "Fraction of MS2 scans with ion injection time recorded", None, None),
    }

    quality_metrics = []
    for mkey, val in metrics.items():
        if mkey.startswith("_") or val is None:
            continue
        if val == "NOT_COMPUTED":
            continue
        if mkey not in METRIC_MAP:
            continue
        acc, name, desc, unit_acc, unit_name = METRIC_MAP[mkey]
        qm = {"cvParameter": {"accession": acc, "name": name,
                               "description": desc, "value": val}}
        if unit_acc:
            qm["unit"] = {"accession": unit_acc, "name": unit_name}
        quality_metrics.append(qm)

    # List-valued metrics
    LIST_METRICS = {
        "MS:4000061": ("MS:4000061", "MS1 density quantiles",
                       "Q1/Q2/Q3 of MS1 peak counts per scan", None, None),
        "MS:4000062": ("MS:4000062", "MS2 density quantiles",
                       "Q1/Q2/Q3 of MS2 peak counts per scan", None, None),
        "MS:4000069": ("MS:4000069", "m/z acquisition range",
                       "[min, max] m/z across all scans", "UO:0000221", "dalton"),
        "MS:4000070": ("MS:4000070", "RT acquisition range",
                       "[min, max] retention time", "UO:0000031", "minute"),
        "MS:4000183": ("MS:4000183", "TIC quantile RT fractions",
                       "RT fractions at which Q1/Q2/Q3/Q4 of cumulative TIC are reached",
                       None, None),
        "MS:4000184": ("MS:4000184", "MS1 quantile RT fractions",
                       "RT fractions at which Q1/Q2/Q3/Q4 of MS1 scan count are reached",
                       None, None),
        "MS:4000185": ("MS:4000185", "MS2 quantile RT fractions",
                       "RT fractions at which Q1/Q2/Q3/Q4 of MS2 scan count are reached",
                       None, None),
        "MS:4000186": ("MS:4000186", "MS1 TIC-change quantile ratios",
                       "Log ratios of successive TIC-change quantiles", None, None),
        "MS:4000187": ("MS:4000187", "MS1 TIC quantile ratios",
                       "Log ratios of successive TIC quantiles", None, None),
        "MS:4000106": ("MS:4000106", "MS1 frequency per RT quarter",
                       "Average MS1 scan frequency in each of 4 equal RT windows",
                       "UO:0000106", "hertz"),
        "MS:4000107": ("MS:4000107", "MS2 frequency per RT quarter",
                       "Average MS2 scan frequency in each of 4 equal RT windows",
                       "UO:0000106", "hertz"),
        "MS:4000116": ("MS:4000116", "MS2 precursor intensity distribution",
                       "Q1/Q2/Q3 of MS2 precursor ion intensities", None, None),
        "MS:4000195": ("MS:4000195", "DIA isolation window m/z widths",
                       "[min, max] of DIA isolation window widths", "UO:0000221", "dalton"),
        "MS:4000196": ("MS:4000196", "DIA isolation window sampling counts",
                       "[min, max] number of cycles per DIA window", None, None),
        "MS:4000197": ("MS:4000197", "DIA isolation window half-TIC RT",
                       "[min, max] RT at which each window reaches 50% cumulative TIC",
                       "UO:0000031", "minute"),
        "MS:4000198": ("MS:4000198", "DIA isolation window TIC",
                       "[min, max] total TIC per DIA window", None, None),
        "MS:4000199": ("MS:4000199", "DIA isolation window peak counts",
                       "[min, max] median peak count per DIA window", None, None),
        "MS:4000051": ("MS:4000051", "XIC-FWHM quantiles",
                       "Q1/Q2/Q3 of chromatographic peak widths (FWHM) of "
                       "MS1 features detected by FeatureFindingMetabo",
                       "UO:0000031", "minute"),
        "MS:4000050": ("MS:4000050", "XIC50 fraction",
                       "Fraction of MS1 features whose FWHM window contains "
                       ">= 50% of the total MS1 TIC", None, None),
        "NEW-015":    ("NEW:0000015", "MS1-to-MS2 ratio per RT quarter",
                       "MS1/MS2 scan count ratio in each of 4 equal RT windows",
                       None, None),
    }

    for mkey, (acc, name, desc, unit_acc, unit_name) in LIST_METRICS.items():
        val = metrics.get(mkey)
        if val is None:
            continue
        if isinstance(val, list) and all(v is None for v in val):
            continue
        qm = {"cvParameter": {"accession": acc, "name": name,
                               "description": desc, "value": val}}
        if unit_acc:
            qm["unit"] = {"accession": unit_acc, "name": unit_name}
        quality_metrics.append(qm)

    #Dict valued metrics
    DICT_METRICS = {
        "MS:4000063": ("MS:4000063", "MS2 known precursor charge fractions",
                       "Fraction of MS2 scans at each observed charge state", None, None),
        "NEW-002":    ("NEW:0000002", "MS level distribution",
                       "Count of scans at each MS level", None, None),
        "NEW-003":    ("NEW:0000003", "scan polarity distribution",
                       "Count by polarity", None, None),
    }
    for mkey, (acc, name, desc, unit_acc, unit_name) in DICT_METRICS.items():
        val = metrics.get(mkey)
        if val is None or not isinstance(val, dict):
            continue
        qm = {"cvParameter": {"accession": acc, "name": name,
                               "description": desc, "value": val}}
        quality_metrics.append(qm)

    mzqc_doc = {
        "mzQC": {
            "version":     "1.0.0",
            "description": f"ID-free QC metrics for {fname} (dataset {dataset})",
            "runQualities": [
                {
                    "metadata": {
                        "label":            f"{dataset}/{fname}",
                        "inputFiles":       [input_file],
                        "analysisSoftware": analysis_software,
                    },
                    "qualityMetrics": quality_metrics,
                }
            ],
            "controlledVocabularies": controlled_vocabularies,
        }
    }
    return mzqc_doc


def open_reader(path_str):
    p = path_str.lower()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if p.endswith(".mzxml"):
            return mzxml.MzXML(path_str)
        if p.endswith(".mgf"):
            from pyteomics import mgf
            return mgf.MGF(path_str)
        return mzml.MzML(path_str, use_index=False)


def is_mgf_path(path_str):
    return path_str.lower().endswith(".mgf")


def is_mzxml_path(path_str):
    return path_str.lower().endswith(".mzxml")


def _rt_to_minutes(val):
    unit = getattr(val, "unit_info", None)
    v = float(val)
    if unit == "second":
        return v / 60.0
    return v


def get_rt(spec, is_mgf=False, is_mzxml=False):
    if is_mgf:
        params = spec.get("params", {})
        rt_sec = params.get("rtinseconds", params.get("rt", 0))
        return float(rt_sec) / 60.0

    #mzXML uses camelCase keys, try them first then fall through to mzML keys
    if is_mzxml:
        for key in ["retentionTime", "retention time", "scanStartTime",
                    "scan start time"]:
            val = spec.get(key)
            if val is not None:
                return _rt_to_minutes(val)
        return 0.0

    #mzML: top level or nested inside scanList
    for key in ["scan start time", "retentionTime", "retention time", "scanStartTime"]:
        val = spec.get(key)
        if val is not None:
            return _rt_to_minutes(val)
    try:
        scans = spec.get("scanList", {}).get("scan", [{}])
        for s in scans:
            for key in ["scan start time", "retentionTime"]:
                val = s.get(key)
                if val is not None:
                    return _rt_to_minutes(val)
    except Exception:
        pass
    return 0.0


def get_arrays(spec, is_mgf=False):
    if is_mgf:
        ints  = np.asarray(spec.get("intensity array",
                           spec.get("intensity", [])), dtype=float)
        mzarr = np.asarray(spec.get("m/z array",
                           spec.get("mass array", [])), dtype=float)
        return ints, mzarr
    ints  = np.asarray(spec.get("intensity array", []), dtype=float)
    mzarr = np.asarray(spec.get("m/z array", []),       dtype=float)
    return ints, mzarr


def get_tic(spec, ints, is_mgf=False):
    if not is_mgf:
        for key in ["total ion current", "totalIonCurrent", "TIC"]:
            val = spec.get(key)
            if val is not None:
                return float(val)
    return float(ints.sum()) if len(ints) > 0 else 0.0


def get_ms_level(spec, is_mgf, is_mzxml):
    if is_mgf:
        return 2
    if is_mzxml:
        #pyteomics mzXML stores msLevel as an integer directly
        lvl = spec.get("msLevel", spec.get("ms level"))
        return int(lvl) if lvl is not None else 1
    return int(spec.get("ms level", 1))


def get_polarity(spec, is_mgf, is_mzxml):
    """Return 'positive', 'negative' or 'unknown' MGF files are always reported as positive (convention)"""
    if is_mgf:
        return "positive"

    if is_mzxml:
        pol = spec.get("polarity", "")
        if pol == "+":
            return "positive"
        if pol == "-":
            return "negative"
        return "unknown"

    #mzML CV term flags
    if spec.get("positive scan") is not None:
        return "positive"
    if spec.get("negative scan") is not None:
        return "negative"
    return "unknown"

#used by different vendor converters

def _extract_prec_mz(sel):
    """Extract precursor m/z from a selectedIon dict, trying common key names"""
    for key in ("selected ion m/z", "selectedIonMz", "m/z", "mz"):
        v = sel.get(key)
        if v is not None:
            return float(v)
    return None


def _extract_prec_charge(sel):
    """Extract charge state from a selectedIon dict"""
    for key in ("charge state", "chargeState", "charge"):
        v = sel.get(key)
        if v is not None:
            try:
                return int(v)
            except (ValueError, TypeError):
                pass
    return None


def _extract_prec_intensity(sel):
    """Extract precursor intensity from a selectedIon dict"""
    for key in ("peak intensity", "selected ion intensity",
                "peakIntensity", "intensity"):
        v = sel.get(key)
        if v is not None:
            return float(v)
    return None


def _extract_isolation_window(iwin):
    lo = iwin.get("isolation window lower offset",
                  iwin.get("lowerOffset", iwin.get("lower offset")))
    hi = iwin.get("isolation window upper offset",
                  iwin.get("upperOffset", iwin.get("upper offset")))
    return lo, hi


#acquisition & centroid detection

def detect_mode(window_pairs, n_ms2, ms1_count=0):
    """classify acquisition mode from window pairs and scan counts"""
    if n_ms2 == 0 and ms1_count == 0:
        return "unknown"
    if n_ms2 == 0 and ms1_count > 0:
        return "MS1-only"
    if ms1_count == 0:
        if not window_pairs:
            return "MS2-only"
        unique = len(set(window_pairs))
        return "DIA-MS2only" if unique <= 100 else "DDA-MS2only"
    if not window_pairs:
        return "DDA"

    unique = len(set(window_pairs))

    #primary criterion: <=100 unique windows and enough ms2 scans
    if unique <= 100 and n_ms2 > 50:
        return "DIA"

    if n_ms2 > 50:
        widths = [round(hi - lo, 2) for lo, hi in window_pairs]
        if len(widths) > 0:
            w_arr = np.array(widths)
            width_cv = (w_arr.std() / w_arr.mean()) if w_arr.mean() > 0 else 1.0
            #uniform width (CV < 5%) with a plausible window count => DIA
            if width_cv < 0.05 and unique <= 500:
                return "DIA"

    return "DDA"


def detect_centroid(path_str, n_sample=30):
    """Estimate whether spectra are centroid, profile, or mixed"""
    densities = []
    mgf_file  = is_mgf_path(path_str)
    try:
        with open_reader(path_str) as reader:
            for i, spec in enumerate(reader):
                if i >= n_sample:
                    break
                _, mzarr = get_arrays(spec, mgf_file)
                if len(mzarr) > 1:
                    rng = float(mzarr[-1]) - float(mzarr[0])
                    if rng > 0:
                        densities.append(len(mzarr) / rng)
    except Exception:
        return "unknown"
    if not densities:
        return "unknown"
    d = float(np.median(densities))
    if d > 5:
        return "profile"
    if d < 2:
        return "centroid"
    return "mixed"


def _spectral_entropy(arr):
    arr = arr[arr > 0]
    if len(arr) == 0:
        return 0.0
    p = arr / arr.sum()
    return float(scipy_entropy(p, base=2))


def safe_quantiles(arr, qs=(0.25, 0.5, 0.75)):
    if len(arr) == 0:
        return [None] * len(qs)
    return [float(np.quantile(arr, q)) for q in qs]


def auc_trapz(tics, rts):
    return float(np.trapz(tics, rts)) if len(rts) >= 2 else 0.0


def freq_max(intervals):
    vals = [x for x in intervals if x > 0]
    return float(1.0 / min(vals)) if vals else None


def freq_min(intervals):
    vals = [x for x in intervals if x > 0]
    return float(1.0 / max(vals)) if vals else None


def tic_quantile_rt_frac(tics, rts, qs=(0.25, 0.5, 0.75, 1.0)):
    if len(tics) == 0 or len(rts) == 0:
        return [None] * len(qs)
    cum   = np.cumsum(tics)
    total = cum[-1]
    dur   = rts[-1] - rts[0]
    if dur == 0 or total == 0:
        return [None] * len(qs)
    out = []
    for q in qs:
        idx = min(int(np.searchsorted(cum, q * total)), len(rts) - 1)
        out.append(float(min(max((rts[idx] - rts[0]) / dur, 0.0), 1.0)))
    return out


def freq_per_quarter(rts, n=4):
    if len(rts) < 2:
        return [None] * n
    t0, t1 = float(rts[0]), float(rts[-1])
    dur = (t1 - t0) / n
    if dur == 0:
        return [None] * n
    return [
        float(int(((rts >= t0 + i * dur) & (rts < t0 + (i + 1) * dur)).sum()) / dur)
        for i in range(n)
    ]


def signal_jumps(tics, factor=10):
    j = f = 0
    for i in range(len(tics) - 1):
        if tics[i] > 0:
            r = tics[i + 1] / tics[i]
            if r > factor:
                j += 1
            elif r < 1 / factor:
                f += 1
    return j, f


def coeff_variation(arr):
    arr = np.asarray(arr, dtype=float)
    mu  = arr.mean()
    return float(arr.std() / mu) if len(arr) > 0 and mu != 0 else None


def gini_coeff(arr):
    arr = np.sort(np.abs(np.asarray(arr, dtype=float)))
    n   = len(arr)
    if n == 0 or arr.sum() == 0:
        return None
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr) / (n * arr.sum())) - (n + 1) / n)


def tic_change_quantile_ratios(tics):
    if len(tics) < 2:
        return [None, None, None]
    d = np.abs(np.diff(tics))
    d = d[d > 0]
    if len(d) == 0:
        return [None, None, None]
    q1, q2, q3 = np.percentile(d, [25, 50, 75])
    mx = d.max()
    lr = lambda a, b: float(np.log(a / b)) if a > 0 and b > 0 else None
    return [lr(q2, q1), lr(q3, q2), lr(mx, q3)]


def tic_quantile_ratios(tics):
    t = tics[tics > 0]
    if len(t) == 0:
        return [None, None, None]
    q1, q2, q3 = np.percentile(t, [25, 50, 75])
    mx = t.max()
    lr = lambda a, b: float(np.log(a / b)) if a > 0 and b > 0 else None
    return [lr(q2, q1), lr(q3, q2), lr(mx, q3)]


def chromatography_duration(tics, rts, thr_frac=0.01):
    if len(tics) == 0:
        return None
    mx = tics.max()
    if mx == 0:
        return None
    active = rts[tics > mx * thr_frac]
    return float(active[-1] - active[0]) if len(active) >= 2 else None

#DIA window metrics

def compute_dia_metrics(dia_windows):
    null = {k: None for k in [
        "MS:4000193", "MS:4000194", "MS:4000195", "MS:4000196",
        "MS:4000197", "MS:4000198", "MS:4000199",
        "NEW-016", "NEW-017", "NEW-018", "NEW-019", "NEW-020",
    ]}
    if not dia_windows:
        return null

    keys = list(dia_windows.keys())
    n_windows  = len(keys)
    widths     = [hi - lo for lo, hi in keys]
    mz_widths  = [float(min(widths)), float(max(widths))]
    sampling   = [len(dia_windows[k]["tics"]) for k in keys]
    samp_range = [int(min(sampling)), int(max(sampling))]

    all_deltas = []
    for k in keys:
        rts_s = sorted(dia_windows[k]["rts"])
        if len(rts_s) >= 2:
            all_deltas.extend(np.diff(rts_s).tolist())
    median_cycle = float(np.median(all_deltas)) if all_deltas else None

    half_tic_rts = []
    for k in keys:
        w = dia_windows[k]
        if not w["rts"]:
            continue
        order  = np.argsort(w["rts"])
        rts_s  = np.array(w["rts"])[order]
        tics_s = np.array(w["tics"])[order]
        cum    = np.cumsum(tics_s)
        total  = cum[-1]
        if total > 0:
            idx = min(int(np.searchsorted(cum, 0.5 * total)), len(rts_s) - 1)
            half_tic_rts.append(float(rts_s[idx]))
    ht_range = ([float(min(half_tic_rts)), float(max(half_tic_rts))]
                if half_tic_rts else None)

    win_tics   = [sum(dia_windows[k]["tics"]) for k in keys]
    tic_range  = [float(min(win_tics)), float(max(win_tics))]
    med_peaks  = [float(np.median(dia_windows[k]["peaks"])) for k in keys]
    peak_range = [float(min(med_peaks)), float(max(med_peaks))]

    sorted_wins  = sorted(keys, key=lambda x: x[0])
    overlaps     = sum(1 for i in range(len(sorted_wins) - 1)
                       if sorted_wins[i][1] > sorted_wins[i + 1][0])
    overlap_frac = (float(overlaps / (len(sorted_wins) - 1))
                    if len(sorted_wins) > 1 else 0.0)

    total_range = max(k[1] for k in keys) - min(k[0] for k in keys)
    merged, covered = [], 0.0
    for s, e in sorted_wins:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append([s, e])
    for s, e in merged:
        covered += e - s
    coverage = float(covered / total_range) if total_range > 0 else None

    cvs = []
    for k in keys:
        t = np.array(dia_windows[k]["tics"])
        if t.mean() > 0:
            cvs.append(float(t.std() / t.mean()))
    median_cv = float(np.median(cvs)) if cvs else None

    ad = np.array(all_deltas)
    cycle_cv = (float(ad.std() / ad.mean())
                if len(ad) > 0 and ad.mean() > 0 else None)

    frag_uni = (float(min(med_peaks) / max(med_peaks))
                if max(med_peaks) > 0 else None)

    return {
        "MS:4000193": median_cycle,
        "MS:4000194": n_windows,
        "MS:4000195": mz_widths,
        "MS:4000196": samp_range,
        "MS:4000197": ht_range,
        "MS:4000198": tic_range,
        "MS:4000199": peak_range,
        "NEW-016":    overlap_frac,
        "NEW-017":    coverage,
        "NEW-018":    median_cv,
        "NEW-019":    cycle_cv,
        "NEW-020":    frag_uni,
    }

#XIC feature detection 

def compute_xic_metrics(path_str, ms2_count=0):
    """chromatographic feature detection on MS1 scans via pyopenms"""
    null = {"MS:4000051": None, "MS:4000050": None}

    if not PYOPENMS_AVAILABLE:
        return null
    if ms2_count == 0:
        return null
    if Path(path_str).suffix.lower() != ".mzml":
        return null

    try:
        oms = _oms

        exp_full = oms.MSExperiment()
        oms.MzMLFile().load(path_str, exp_full)

        exp = oms.MSExperiment()
        for spec in exp_full:
            if spec.getMSLevel() == 1 and spec.size() > 0:
                exp.addSpectrum(spec)

        if exp.getNrSpectra() < 10:
            return null

        exp.sortSpectra()

        rts_s  = np.array([s.getRT() for s in exp])
        rt_dur = rts_s[-1] - rts_s[0]
        if rt_dur <= 0:
            return null

        #mass trace detection
        mtd = oms.MassTraceDetection()
        p   = mtd.getDefaults()
        p.setValue("mass_error_ppm",      10.0)
        p.setValue("noise_threshold_int", 500.0)
        p.setValue("min_trace_length",    5.0)
        p.setValue("max_trace_length",   -1.0)
        mtd.setParameters(p)
        traces = []
        mtd.run(exp, traces, 0)
        if not traces:
            return null

        #Elution peak detection
        epd = oms.ElutionPeakDetection()
        p2  = epd.getDefaults()
        p2.setValue("width_filtering", "auto")
        p2.setValue("min_fwhm",  1.0)
        p2.setValue("max_fwhm",  rt_dur * 0.8)
        epd.setParameters(p2)
        split = []
        epd.detectPeaks(traces, split)
        if not split:
            return null

        #Feature finding
        ffm = oms.FeatureFindingMetabo()
        p3  = ffm.getDefaults()
        p3.setValue("isotope_filtering_model", "none")
        p3.setValue("remove_single_traces",    "false")
        p3.setValue("mz_scoring_13C",          "false")
        ffm.setParameters(p3)
        feat_map  = oms.FeatureMap()
        chrom_out = []
        ffm.run(split, feat_map, chrom_out)

        if feat_map.size() == 0:
            return null

        #FWHM values
        fwhm_s = []
        for i in range(feat_map.size()):
            f = feat_map[i]
            if f.metaValueExists("FWHM"):
                v = float(f.getMetaValue("FWHM"))
                if v > 0:
                    fwhm_s.append(v)

        if len(fwhm_s) < 3:
            return null

        fwhm_min = np.array(fwhm_s) / 60.0
        xic_fwhm = [round(float(np.quantile(fwhm_min, q)), 4)
                    for q in (0.25, 0.50, 0.75)]

        feat_intensities = []
        for i in range(feat_map.size()):
            intensity = float(feat_map[i].getIntensity())
            if intensity > 0:
                feat_intensities.append(intensity)

        xic50 = None
        if feat_intensities:
            feat_intensities_sorted = sorted(feat_intensities, reverse=True)
            total_feat_intensity    = sum(feat_intensities_sorted)
            if total_feat_intensity > 0:
                cumsum = 0.0
                count_needed = 0
                for intensity in feat_intensities_sorted:
                    cumsum += intensity
                    count_needed += 1
                    if cumsum / total_feat_intensity >= 0.50:
                        break
                xic50 = round(count_needed / len(feat_intensities), 4)

        return {"MS:4000051": xic_fwhm, "MS:4000050": xic50}

    except Exception:
        return null


def compute_all_metrics(mzml_path):
    path_str  = str(mzml_path)
    mgf_file  = is_mgf_path(path_str)
    mzxml_file = is_mzxml_path(path_str)

    ms1_tics, ms1_rts, ms1_peaks = [], [], []
    ms2_tics, ms2_rts, ms2_peaks = [], [], []
    ms1_entropy_vals             = []
    ms2_entropy_vals             = []
    ms2_prec_mz                  = []
    ms2_prec_int                 = []
    ms2_charge_list              = []
    polarities                   = []
    ms_levels                    = []
    window_pairs                 = []
    dia_windows                  = {}

    ms1_intervals, ms2_intervals = [], []
    prev_ms1_rt = prev_ms2_rt    = None

    ms1_empty = ms2_empty        = 0
    ms2_total                    = 0
    charge_annotated             = 0
    it_annotated                 = 0
    meta_complete                = 0
    zero_peaks = total_peaks     = 0
    all_mz_min = float("inf")
    all_mz_max = float("-inf")

    import io, contextlib
    try:
        _devnull = io.StringIO()
        with warnings.catch_warnings(), contextlib.redirect_stderr(_devnull):
            warnings.simplefilter("ignore")
        with open_reader(path_str) as reader:
            for spec in reader:
                level = get_ms_level(spec, mgf_file, mzxml_file)
                rt    = get_rt(spec, mgf_file, mzxml_file)
                ints, mzarr = get_arrays(spec, mgf_file)
                tic   = get_tic(spec, ints, mgf_file)
                npk   = len(ints)

                ms_levels.append(level)

                polarities.append(get_polarity(spec, mgf_file, mzxml_file))

                if npk > 0:
                    zero_peaks  += int((ints == 0).sum())
                    total_peaks += npk

                if len(mzarr) > 0:
                    all_mz_min = min(all_mz_min, float(mzarr.min()))
                    all_mz_max = max(all_mz_max, float(mzarr.max()))

                if level == 1:
                    ms1_tics.append(tic)
                    ms1_rts.append(rt)
                    ms1_peaks.append(npk)
                    ms1_entropy_vals.append(_spectral_entropy(ints))
                    if tic == 0:
                        ms1_empty += 1
                    if prev_ms1_rt is not None:
                        ms1_intervals.append(rt - prev_ms1_rt)
                    prev_ms1_rt = rt

                elif level == 2:
                    ms2_tics.append(tic)
                    ms2_rts.append(rt)
                    ms2_peaks.append(npk)
                    ms2_entropy_vals.append(_spectral_entropy(ints))
                    if tic == 0:
                        ms2_empty += 1
                    ms2_total += 1
                    if prev_ms2_rt is not None:
                        ms2_intervals.append(rt - prev_ms2_rt)
                    prev_ms2_rt = rt

                    if not mgf_file:
                        try:
                            prec = spec["precursorList"]["precursor"][0]
                            sel  = prec["selectedIonList"]["selectedIon"][0]
                            iwin = prec.get("isolationWindow", {})
                            act  = prec.get("activation", {})

                            pmz = _extract_prec_mz(sel)
                            pch = _extract_prec_charge(sel)
                            pi  = _extract_prec_intensity(sel)
                            lo, hi = _extract_isolation_window(iwin)

                            if pmz is not None:
                                ms2_prec_mz.append(pmz)
                            if pch is not None:
                                ms2_charge_list.append(pch)
                                charge_annotated += 1
                            if pi is not None:
                                ms2_prec_int.append(pi)
                            if lo is not None and hi is not None and pmz is not None:
                                lo_f, hi_f = float(lo), float(hi)
                                wp   = (round(pmz - lo_f, 3), round(pmz + hi_f, 3))
                                wkey = (round(pmz - lo_f, 2), round(pmz + hi_f, 2))
                                window_pairs.append(wp)
                                if wkey not in dia_windows:
                                    dia_windows[wkey] = {"tics": [], "rts": [], "peaks": []}
                                dia_windows[wkey]["tics"].append(tic)
                                dia_windows[wkey]["rts"].append(rt)
                                dia_windows[wkey]["peaks"].append(npk)
                            pol_ok = polarities[-1] != "unknown"
                            if (pmz is not None and lo is not None
                                    and hi is not None and pol_ok and len(act) > 0):
                                meta_complete += 1
                            it_val = (spec.get("ion injection time") or
                                      spec.get("ionInjectionTime"))
                            if it_val is not None:
                                it_annotated += 1
                        except (KeyError, IndexError):
                            pass
                    else:
                        params = spec.get("params", {})
                        pm = params.get("pepmass")
                        if pm is not None:
                            pmz_val = pm[0] if isinstance(pm, (list, tuple)) else float(pm)
                            ms2_prec_mz.append(float(pmz_val))
                        ch = str(params.get("charge", "")).replace("+", "").replace("-", "")
                        if ch.isdigit():
                            ms2_charge_list.append(int(ch))
                            charge_annotated += 1

    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}

    ms1_tics = np.array(ms1_tics, dtype=float)
    ms2_tics = np.array(ms2_tics, dtype=float)
    ms1_rts  = np.array(ms1_rts,  dtype=float)
    ms2_rts  = np.array(ms2_rts,  dtype=float)
    all_tics = (np.concatenate([ms1_tics, ms2_tics])
                if len(ms1_tics) + len(ms2_tics) > 0 else np.array([]))
    all_rts  = (np.concatenate([ms1_rts, ms2_rts])
                if len(ms1_rts) + len(ms2_rts) > 0 else np.array([]))

    if len(ms1_tics) >= 2:
        jumps, falls = signal_jumps(ms1_tics)
    else:
        jumps, falls = None, None

    active_ms1 = ms1_tics[ms1_tics > 0]
    baseline   = float(np.percentile(active_ms1, 5)) if len(active_ms1) > 0 else None
    median_ms1 = float(np.median(active_ms1))        if len(active_ms1) > 0 else 0.0
    sbr        = float(median_ms1 / baseline) if baseline and baseline > 0 else None

    auc_ms1 = auc_trapz(ms1_tics, ms1_rts) if len(ms1_tics) >= 2 else None
    auc_ms2 = auc_trapz(ms2_tics, ms2_rts) if len(ms2_tics) >= 2 else None
    mode    = detect_mode(window_pairs, ms2_total, ms1_count=int(len(ms1_tics)))

    run_duration = (float(all_rts.max() - all_rts.min())
                    if len(all_rts) > 1 and all_rts.max() > 0 else None)
    rt_range     = ([float(all_rts.min()), float(all_rts.max())]
                    if len(all_rts) > 0 and all_rts.max() > 0 else None)

    charge_fracs = None
    if ms2_charge_list:
        cc    = Counter(ms2_charge_list)
        total = sum(cc.values())
        charge_fracs = {str(k): round(v / total, 6) for k, v in sorted(cc.items())}

    iso_widths = [hi - lo for lo, hi in window_pairs] if window_pairs else []

    ms2_pos   = ms2_tics[ms2_tics > 0]
    dyn_range = None
    if len(ms2_pos) >= 10:
        p5, p95 = np.percentile(ms2_pos, 5), np.percentile(ms2_pos, 95)
        if p5 > 0:
            dyn_range = float(np.log10(p95 / p5))

    ms1_ms2_ratio = [None] * 4
    if len(ms1_rts) > 0 and len(ms2_rts) > 0:
        t0  = min(float(ms1_rts[0]), float(ms2_rts[0]))
        t1  = max(float(ms1_rts[-1]), float(ms2_rts[-1]))
        dur = (t1 - t0) / 4 if t1 > t0 else 0
        if dur > 0:
            ms1_ms2_ratio = []
            for i in range(4):
                lo_b, hi_b = t0 + i * dur, t0 + (i + 1) * dur
                n1 = int(((ms1_rts >= lo_b) & (ms1_rts < hi_b)).sum())
                n2 = int(((ms2_rts >= lo_b) & (ms2_rts < hi_b)).sum())
                ms1_ms2_ratio.append(float(n1 / n2) if n2 > 0 else None)

    redundancy = None
    if mode == "DDA" and ms2_prec_mz and ms2_total > 0:
        mz_arr    = np.array(ms2_prec_mz)
        redundant = 0
        for i in range(len(mz_arr)):
            diffs    = np.abs(mz_arr - mz_arr[i])
            diffs[i] = 999
            if diffs.min() < 0.01:
                redundant += 1
        redundancy = float(redundant / ms2_total)

    m = {}
    m["MS:4000059"] = int(len(ms1_tics))
    m["MS:4000060"] = int(len(ms2_tics))
    m["MS:4000067"] = run_duration
    m["MS:4000053"] = chromatography_duration(ms1_tics, ms1_rts)
    m["MS:4000069"] = ([all_mz_min, all_mz_max]
                       if all_mz_min != float("inf") else None)
    m["MS:4000070"] = rt_range
    m["MS:4000065"] = freq_max(ms1_intervals)
    m["MS:4000066"] = freq_max(ms2_intervals)
    m["MS:4000095"] = freq_min(ms1_intervals)
    m["MS:4000096"] = freq_min(ms2_intervals)
    m["MS:4000106"] = freq_per_quarter(ms1_rts)
    m["MS:4000107"] = freq_per_quarter(ms2_rts)
    m["MS:4000029"] = auc_ms1
    m["MS:4000030"] = auc_ms2
    m["MS:4000031"] = (float(auc_ms1 / auc_ms2)
                       if auc_ms1 is not None and auc_ms2 and auc_ms2 > 0
                       else None)
    m["MS:4000186"] = tic_change_quantile_ratios(ms1_tics)
    m["MS:4000187"] = tic_quantile_ratios(ms1_tics)
    m["MS:4000183"] = tic_quantile_rt_frac(all_tics, all_rts)
    m["MS:4000184"] = tic_quantile_rt_frac(np.ones(len(ms1_rts)), ms1_rts)
    m["MS:4000185"] = tic_quantile_rt_frac(np.ones(len(ms2_rts)), ms2_rts)
    m["MS:4000061"] = safe_quantiles(ms1_peaks)
    m["MS:4000062"] = safe_quantiles(ms2_peaks)
    m["MS:4000097"] = int(jumps) if jumps is not None else None
    m["MS:4000098"] = int(falls) if falls is not None else None
    m["MS:4000099"] = int(ms1_empty)
    m["MS:4000100"] = int(ms2_empty)
    m["MS:4000063"] = charge_fracs
    m["MS:4000116"] = safe_quantiles(ms2_prec_int)

    xic = compute_xic_metrics(path_str, ms2_count=ms2_total)
    m["MS:4000051"] = xic["MS:4000051"]   #XIC-FWHM Q1/Q2/Q3(min)
    m["MS:4000050"] = xic["MS:4000050"]   #XIC50 fraction

    m["NEW-001"] = mode
    m["NEW-002"] = dict(Counter(ms_levels))
    m["NEW-003"] = dict(Counter(polarities))
    m["NEW-004"] = (float(np.median(ms1_entropy_vals))
                    if ms1_entropy_vals else None)
    m["NEW-005"] = (float(np.median(ms2_entropy_vals))
                    if ms2_entropy_vals else None)
    m["NEW-006"] = (float(sum(1 for p in ms2_peaks if p < 10) / len(ms2_peaks))
                    if ms2_peaks else None)
    m["NEW-007"] = dyn_range
    m["NEW-008"] = float(zero_peaks / total_peaks) if total_peaks > 0 else None
    m["NEW-009"] = coeff_variation(ms1_tics)
    m["NEW-010"] = gini_coeff(ms1_tics)
    m["NEW-011"] = baseline
    m["NEW-012"] = sbr
    m["NEW-013"] = safe_quantiles(iso_widths) if iso_widths else None
    m["NEW-014"] = redundancy
    m["NEW-015"] = ms1_ms2_ratio
    m["NEW-021"] = float(meta_complete / ms2_total) if ms2_total > 0 else None
    m["NEW-022"] = detect_centroid(path_str)
    m["NEW-023"] = float(charge_annotated / ms2_total) if ms2_total > 0 else None
    m["NEW-024"] = (float(it_annotated / ms2_total)
                    if ms2_total > 0 and not mgf_file else None)

    if mode in ("DIA", "DIA-MS2only"):
        m.update(compute_dia_metrics(dia_windows))
    else:
        for k in ["MS:4000193", "MS:4000194", "MS:4000195", "MS:4000196",
                  "MS:4000197", "MS:4000198", "MS:4000199",
                  "NEW-016", "NEW-017", "NEW-018", "NEW-019", "NEW-020"]:
            m[k] = None

    return m


def flatten(v, prefix=""):
    out = {}
    if isinstance(v, dict):
        if prefix:
            out[prefix] = str(v)
        for k, val in v.items():
            out.update(flatten(val, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(v, list):
        if prefix:
            out[prefix] = str(v)
        for i, val in enumerate(v):
            out.update(flatten(val, f"{prefix}[{i}]" if prefix else f"[{i}]"))
    else:
        out[prefix] = v
    return out


def sfmt(v, decimals=3):
    if v is None or str(v) in ("None", "", "NOT_COMPUTED"):
        return "N/A"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)[:12]



def main():
    raw_files = []
    for ext in ["*.mzML", "*.mzml", "*.mzXML", "*.mzxml", "*.mgf", "*.MGF"]:
        raw_files.extend(DATA_DIR.rglob(ext))
    raw_files = sorted(set(raw_files))

    EXT_RANK = {".mgf": 0, ".mzml": 1, ".mzxml": 2}
    best = {}
    for f in raw_files:
        key  = (str(f.parent), f.stem.lower())
        rank = EXT_RANK.get(f.suffix.lower(), 9)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, f)
    ms_files = sorted(v for _, v in best.values())

    active_datasets = sorted(set(f.parent.name for f in ms_files))
    all_folders     = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    skipped_folders = [d for d in all_folders if d not in active_datasets]

    print(f"{'='*65}")
    print(f"QC Pipeline  {len(ms_files)} files across "
          f"{len(active_datasets)} datasets")
    if skipped_folders:
        print(f"Skipped (no MS files): {', '.join(skipped_folders)}")
    print(f"{'='*65}\n")

    all_results, errors = [], []

    for i, f in enumerate(ms_files):
        dataset = f.parent.name
        fname   = f.name
        size_mb = f.stat().st_size / 1_000_000

        print(f"[{i+1:02d}/{len(ms_files)}] {dataset}/{fname} ({size_mb:.0f} MB)")

        t0      = time.time()
        metrics = compute_all_metrics(f)
        elapsed = time.time() - t0

        if "_error" in metrics:
            print(f"  ERROR: {metrics['_error']}\n")
            errors.append({"file": fname, "dataset": dataset,
                           "error": metrics["_error"]})
            continue

        metrics["_file"]    = fname
        metrics["_dataset"] = dataset
        metrics["_size_mb"] = round(size_mb, 1)
        metrics["_mode"]    = metrics.get("NEW-001", "unknown")
        metrics["_time_s"]  = round(elapsed, 1)
        all_results.append(metrics)

        mode = metrics.get("NEW-001", "?")
        print(f"  Mode          : {mode}")
        print(f"  Centroid      : {metrics.get('NEW-022', '?')}")
        print(f"  MS1 scans     : {metrics.get('MS:4000059', '?')}")
        print(f"  MS2 scans     : {metrics.get('MS:4000060', '?')}")
        print(f"  Run duration  : {sfmt(metrics.get('MS:4000067'), 1)} min")
        print(f"  Chrom dur     : {sfmt(metrics.get('MS:4000053'), 1)} min")
        print(f"  AUC MS1       : {sfmt(metrics.get('MS:4000029'), 0)}")
        print(f"  AUC MS2       : {sfmt(metrics.get('MS:4000030'), 0)}")
        print(f"  MS1/MS2 AUC   : {sfmt(metrics.get('MS:4000031'))}")
        print(f"  TIC CV        : {sfmt(metrics.get('NEW-009'))}")
        print(f"  TIC Gini      : {sfmt(metrics.get('NEW-010'))}")
        print(f"  Entropy MS1   : {sfmt(metrics.get('NEW-004'))}")
        print(f"  Entropy MS2   : {sfmt(metrics.get('NEW-005'))}")
        print(f"  Sparse MS2%   : {sfmt(metrics.get('NEW-006'))}")
        print(f"  Dyn range     : {sfmt(metrics.get('NEW-007'))}")
        print(f"  Zero peaks%   : {sfmt(metrics.get('NEW-008'))}")
        print(f"  Baseline      : {sfmt(metrics.get('NEW-011'), 0)}")
        print(f"  SBR           : {sfmt(metrics.get('NEW-012'), 1)}")
        print(f"  Metadata ok   : {sfmt(metrics.get('NEW-021'))}")
        print(f"  Charge annot  : {sfmt(metrics.get('NEW-023'))}")
        print(f"  IT annot      : {sfmt(metrics.get('NEW-024'))}")
        print(f"  Redundancy    : {sfmt(metrics.get('NEW-014'))}")
        j_up   = metrics.get("MS:4000097")
        j_down = metrics.get("MS:4000098")
        j_str  = f"{j_up} up  {j_down} down" if j_up is not None else "N/A"
        print(f"  Signal jumps  : {j_str}")
        if mode in ("DIA", "DIA-MS2only"):
            print(f"  DIA windows   : {metrics.get('MS:4000194', '?')}")
            print(f"  DIA overlap%  : {sfmt(metrics.get('NEW-016'))}")
            print(f"  DIA coverage  : {sfmt(metrics.get('NEW-017'))}")
            print(f"  DIA TIC CV    : {sfmt(metrics.get('NEW-018'))}")
            print(f"  DIA cycle CV  : {sfmt(metrics.get('NEW-019'))}")
            print(f"  DIA frag uni  : {sfmt(metrics.get('NEW-020'))}")
        xic_fwhm_q2 = None
        v51 = metrics.get("MS:4000051")
        if isinstance(v51, list) and len(v51) >= 2:
            xic_fwhm_q2 = v51[1]
        print(f"  XIC-FWHM Q2   : {sfmt(xic_fwhm_q2, 3)} min")
        print(f"  XIC50         : {sfmt(metrics.get('MS:4000050'))}")
        print(f"  Time          : {elapsed:.1f}s\n")

        #Write mzQC
        mzqc_doc  = build_mzqc(metrics, str(f), dataset)
        mzqc_path = MZQC_DIR / dataset / f"{f.stem}.mzQC"
        mzqc_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mzqc_path, "w") as jf:
            json.dump(mzqc_doc, jf, indent=2, default=str)

    print(f"\n{'='*65}")
    print(f"Processed {len(all_results)}/{len(ms_files)} files  "
          f"({len(errors)} errors)")
    print(f"{'='*65}\n")

    if not all_results:
        print("No results to save.")
        return

    flat_rows = [flatten(r) for r in all_results]
    all_keys  = set()
    for row in flat_rows:
        all_keys.update(row.keys())
    meta_cols   = sorted(k for k in all_keys if k.startswith("_"))
    metric_cols = sorted(k for k in all_keys if not k.startswith("_"))
    all_cols    = meta_cols + metric_cols

    tsv_out = TSV_DIR / "all_metrics.tsv"
    with open(tsv_out, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=all_cols,
                                delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in flat_rows:
            writer.writerow({k: row.get(k, "") for k in all_cols})

    print(f"Flat TSV      : {tsv_out}")
    print(f"Rows          : {len(flat_rows)}")
    print(f"Columns       : {len(all_cols)}")

    numeric_metrics = [
        #acquisition coverage
        "MS:4000059", "MS:4000060", "MS:4000067", "MS:4000053",
        "MS:4000065", "MS:4000066", "MS:4000095", "MS:4000096",
        "MS:4000097", "MS:4000098", "MS:4000099", "MS:4000100",
        #intensity / signal
        "MS:4000029", "MS:4000030", "MS:4000031",
        #spectral quality
        "NEW-004", "NEW-005", "NEW-006", "NEW-007", "NEW-008",
        "NEW-009", "NEW-010", "NEW-011", "NEW-012", "NEW-014",
        #metadata / annotation
        "NEW-021", "NEW-023", "NEW-024",
        #XIC feature detection
        "MS:4000050",
        #DIA metrics
        "MS:4000193", "MS:4000194",
        "NEW-016", "NEW-017", "NEW-018", "NEW-019", "NEW-020",
    ]

    datasets_dict = {}
    for r in all_results:
        datasets_dict.setdefault(r.get("_dataset", "unknown"), []).append(r)

    summary_rows = []
    for ds, rows in sorted(datasets_dict.items()):
        sr = {
            "dataset":  ds,
            "n_files":  len(rows),
            "mode":     rows[0].get("NEW-001", "?"),
            "centroid": rows[0].get("NEW-022", "?"),
        }
        for mk in numeric_metrics:
            vals = [r.get(mk) for r in rows
                    if r.get(mk) is not None
                    and isinstance(r.get(mk), (int, float))]
            if vals:
                sr[f"{mk}_mean"] = round(float(np.mean(vals)),  4)
                sr[f"{mk}_std"]  = round(float(np.std(vals)),   4)
                sr[f"{mk}_min"]  = round(float(np.min(vals)),   4)
                sr[f"{mk}_max"]  = round(float(np.max(vals)),   4)
            else:
                sr[f"{mk}_mean"] = sr[f"{mk}_std"] = \
                sr[f"{mk}_min"]  = sr[f"{mk}_max"] = ""
        summary_rows.append(sr)

    sum_out  = TSV_DIR / "summary_by_dataset.tsv"
    sum_keys = list(summary_rows[0].keys())
    with open(sum_out, "w", newline="") as sf:
        writer = csv.DictWriter(sf, fieldnames=sum_keys,
                                delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Dataset TSV   : {sum_out}")
    print(f"mzQC files    : {MZQC_DIR}/<dataset>/<file>.mzQC")

    print(f"\n{'─'*78}")
    print(f"{'File':<38} {'Mode':<12} {'MS1':>6} {'MS2':>6} "
          f"{'TIC_CV':>8} {'H_MS2':>7} {'Dur':>6}")
    print(f"{'─'*78}")
    for r in flat_rows:
        print(f"  {str(r.get('_file', ''))[:36]:<36} "
              f"{str(r.get('_mode', ''))[:10]:<12} "
              f"{str(r.get('MS:4000059', ''))[:6]:>6} "
              f"{str(r.get('MS:4000060', ''))[:6]:>6} "
              f"{sfmt(r.get('NEW-009')):>8} "
              f"{sfmt(r.get('NEW-005')):>7} "
              f"{sfmt(r.get('MS:4000067'), 1):>6}")

    if errors:
        print(f"\n{'─'*78}")
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e['dataset']}/{e['file']}: {e['error'][:70]}")

    print(f"\n{'─'*78}")
    print("N/A guide:")
    print("  MS1 metrics N/A  = 0 MS1 scans (MS2-only or targeted acquisition)")
    print("  MS2 metrics N/A  = 0 MS2 scans (MS1-only survey or metabolomics)")
    print("  Metadata ok = 0  = isolation window / activation absent from file")
    print("  Redundancy N/A   = precursor m/z not present in file metadata")
    print("  Mode=MS2-only    = only MS2 exported; DDA/DIA not determinable")
    print("  Mode=MS1-only    = survey or native MS; no fragmentation")
    print(f"{'─'*78}")
    print(f"\nDone. Results: {RESULTS_DIR.absolute()}")


if __name__ == "__main__":
    main()
