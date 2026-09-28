"""Download EPA AQS daily-summary files for the Atmos study scope.

Downloads (or resumes) national daily CSVs from the AQS airdata site using
only the Python standard library, so it can run before any project
dependencies are installed. Files are large national datasets; California
subset selection happens in the ETL stage, not here.

Usage (from the project root):
    py scripts/download_data.py            # download missing files
    py scripts/download_data.py --check    # verify existing files only
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    AQS_BASE_URL,
    DATA_RAW,
    PARAMETER_CODES,
    YEARS,
)

FILE_PATTERN = "daily_{code}_{year}.zip"


def remote_names() -> list[str]:
    names = []
    for name, code in PARAMETER_CODES.items():
        for year in YEARS:
            names.append(FILE_PATTERN.format(code=code, year=year))
    return names


def existing_files() -> dict[str, int]:
    out: dict[str, int] = {}
    for f in sorted(DATA_RAW.glob("daily_*.csv")):
        out[f.name] = f.stat().st_size
    return out


def extract(zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".csv")]
        if len(members) != 1:
            raise RuntimeError(f"expected one CSV inside {zip_path.name}, got {members}")
        dest = DATA_RAW / members[0]
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        zf.extract(members[0], DATA_RAW)
        return dest


def download(name: str, retries: int = 3) -> Path:
    url = f"{AQS_BASE_URL}/{name}"
    dest = DATA_RAW / name
    tmp = dest.with_suffix(".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"  downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "atmos-project/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                with open(tmp, "wb") as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
            tmp.replace(dest)
            csv = extract(dest)
            print(f"  saved -> {csv.name} ({csv.stat().st_size / 1e6:.1f} MB)")
            return dest
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  attempt {attempt} failed: {exc}")
    raise RuntimeError(f"could not download {name}: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download EPA AQS daily files")
    ap.add_argument("--check", action="store_true", help="verify existing files only")
    args = ap.parse_args()

    present = existing_files()
    expected = [n.replace(".zip", ".csv") for n in remote_names()]
    missing = [n for n in expected if n not in present]

    print(f"Expected {len(expected)} files; {len(expected) - len(missing)} present, {len(missing)} missing.")
    for name, size in present.items():
        print(f"  [present] {name:28s} {size / 1e6:8.1f} MB")

    if args.check:
        return 0

    failures = []
    for name in missing:
        try:
            download(name)
        except RuntimeError as exc:
            print(exc)
            failures.append(name)
    if failures:
        print(f"FAILED downloads: {failures}")
        return 1
    print("All required files are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
