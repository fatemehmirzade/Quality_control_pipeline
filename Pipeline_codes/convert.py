import argparse
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR  = Path("/Users/fateme/Desktop/test_metadata/QC/QC_pipeline")
RAW_DIR   = BASE_DIR / "data_raw"
OUT_DIR   = BASE_DIR / "data"
TRFP_EXE  = BASE_DIR / "ThermoRawFileParser" / "ThermoRawFileParser.exe"

RAW_EXTENSIONS = {".raw", ".RAW"}  


def check_mono() -> bool:
    try:
        r = subprocess.run(
            ["mono", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            version_line = r.stdout.splitlines()[0] if r.stdout else "unknown"
            print(f"  mono   : {version_line}")
            return True
    except FileNotFoundError:
        pass
    return False


def check_trfp(exe: Path) -> bool:
    if exe.exists():
        print(f"  parser : {exe}")
        return True
    return False


def find_raw_files(raw_dir: Path) -> list[Path]:
    files = []
    for ext in RAW_EXTENSIONS:
        files.extend(raw_dir.rglob(f"*{ext}"))
    return sorted(files)


def already_converted(raw_file: Path, out_dir: Path) -> bool:
    accession = raw_file.parent.name
    mzml = out_dir / accession / f"{raw_file.stem}.mzML"
    return mzml.exists() and mzml.stat().st_size > 0


def convert_file(raw_file: Path, out_dir: Path, trfp_exe: Path) -> bool:
    accession  = raw_file.parent.name
    target_dir = out_dir / accession
    target_dir.mkdir(parents=True, exist_ok=True)
    mzml = target_dir / f"{raw_file.stem}.mzML"

    cmd = [
        "mono", str(trfp_exe),
        "-i", str(raw_file),        
        "-o", str(target_dir),     
        "-f", "1",                  
    ]

    print(f"    cmd : {' '.join(cmd)}")
    t0 = time.time()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        import threading

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def _drain(pipe, store, label):
            for ln in pipe:
                s = ln.rstrip()
                if s:
                    store.append(s)
                    print(f"      [{label}] {s}", flush=True)

        t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_lines, "out"), daemon=True)
        t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_lines, "err"), daemon=True)
        t_out.start()
        t_err.start()

        while proc.poll() is None:
            elapsed = int(time.time() - t0)
            print(f"\r    ... {elapsed}s elapsed", end="", flush=True)
            time.sleep(5)
        print() 

        t_out.join(); t_err.join()
        elapsed = int(time.time() - t0)

        if proc.returncode == 0 and mzml.exists() and mzml.stat().st_size > 0:
            size_mb = mzml.stat().st_size / 1e6
            print(f"     {mzml.name}  ({size_mb:.0f} MB, {elapsed}s)")
            return True
        else:
            print(f"    ✗ Failed (exit {proc.returncode}, {elapsed}s)")
            all_output = stdout_lines + stderr_lines
            for ln in all_output[-10:]:
                print(f"      {ln}")
            return False

    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"\n Timeout (>1 hour) — skipping {raw_file.name}")
        return False
    except Exception as exc:
        print(f" Error: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert Thermo RAW files to mzML using ThermoRawFileParser + Mono"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR,
        help=f"Root folder containing RAW files (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_DIR,
        help=f"Root folder for mzML output (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--trfp", type=Path, default=TRFP_EXE,
        help=f"Path to ThermoRawFileParser.exe (default: {TRFP_EXE})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-convert files that already have an mzML output",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Thermo RAW → mzML Conversion (ThermoRawFileParser + Mono)")
    print("=" * 60)

    print("\nChecking prerequisites...")
    ok = True

    if not check_mono():
        print(" mono not found on PATH")
        print("    Install from: https://www.mono-project.com/download/stable/")
        print("    Or direct pkg: https://download.mono-project.com/archive/"
              "6.12.0/macos-10-universal/"
              "MonoFramework-MDK-6.12.0.206.macos10.xamarin.universal.pkg")
        ok = False
    else:
        print("mono found")

    if not check_trfp(args.trfp):
        print(f"  ThermoRawFileParser.exe not found at: {args.trfp}")
        print("    Download and unzip from:")
        print("    https://github.com/compomics/ThermoRawFileParser/releases/tag/v1.4.5")
        print(f"    Then place ThermoRawFileParser.exe at: {args.trfp}")
        ok = False
    else:
        print(" ThermoRawFileParser.exe found")

    if not args.raw_dir.is_dir():
        print(f" RAW directory not found: {args.raw_dir}")
        ok = False
    else:
        print(f" RAW directory: {args.raw_dir}")

    if not ok:
        print("\nFix the issues above, then re-run this script.")
        sys.exit(1)

    print(f"\nScanning {args.raw_dir} ...")
    raw_files = find_raw_files(args.raw_dir)

    if not raw_files:
        print("  No .raw / .RAW files found.")
        print("  Make sure your files are under data_raw/<ACCESSION>/*.raw")
        sys.exit(0)

    print(f"  Found {len(raw_files)} RAW file(s):\n")
    for f in raw_files:
        status = ""
        if not args.force and already_converted(f, args.out_dir):
            status = "  [already converted — will skip]"
        print(f"    {f.parent.name}/{f.name}{status}")

    print()
    ok_count   = 0
    skip_count = 0
    fail_count = 0

    for raw_file in raw_files:
        accession = raw_file.parent.name
        print(f"\n {accession}/{raw_file.name}")

        if not args.force and already_converted(raw_file, args.out_dir):
            print(f"    Already converted — skipping (use --force to reconvert)")
            skip_count += 1
            continue

        success = convert_file(raw_file, args.out_dir, args.trfp)
        if success:
            ok_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 60)
    print("Conversion summary")
    print("=" * 60)
    print(f"  Converted : {ok_count}")
    print(f"  Skipped   : {skip_count}  (already existed)")
    print(f"  Failed    : {fail_count}")

    all_mzml = sorted(args.out_dir.rglob("*.mzML"))
    print(f"\nTotal mzML files now available: {len(all_mzml)}")
    for f in all_mzml:
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.parent.name}/{f.name}  ({size_mb:.0f} MB)")

    if fail_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
