"""Fetch the freely, programmatically downloadable test datasets.

Downloads land in ``datasets/`` (gitignored — never committed). Every URL
here was verified reachable without an account. Datasets that DO require
an account or an access request are intentionally not fetched; they are
listed at the bottom and left to the user.

Usage:
    python download_datasets.py           # standard set (~2.3 GB)
    python download_datasets.py --full    # + remaining Ko-PER sequences (~3 GB more)
    python download_datasets.py --list    # show what would be downloaded
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

KOPER_BASE = ("https://www.uni-ulm.de/fileadmin/website_uni_ulm/iui.inst.110/"
              "Bilder/Forschung/Datensaetze")

#: (relative target path, url, approx size, include-in-standard-set)
DOWNLOADS = [
    # Ko-PER: instrumented German intersection (8 cameras + laserscanners),
    # pedestrians/cyclists/vehicles with reference labels. Sequence 1a has
    # object labels; sequences 2/3 are turn and straight-crossing maneuvers.
    ("koper/ExampleOneCamSeq1a.avi", f"{KOPER_BASE}/ExampleOneCamSeq1a.avi", "12 MB", True),
    ("koper/DatasetDocumentation.pdf", f"{KOPER_BASE}/20141010_DatasetDocumentation.pdf", "2 MB", True),
    ("koper/Sequence1a.zip", f"{KOPER_BASE}/20140618_Sequence1a.zip", "1.3 GB", True),
    ("koper/Sequence3.zip", f"{KOPER_BASE}/20140527_Sequence3.zip", "565 MB", True),
    ("koper/Sequence2.zip", f"{KOPER_BASE}/20140527_Sequence2.zip", "418 MB", False),
    ("koper/Sequence1b.zip", f"{KOPER_BASE}/20140527_Sequence1b.zip", "1.3 GB", False),
    # DAWN: ~1,000 annotated vehicle images in fog/rain/snow/sandstorm
    # (detector benchmarking; Mendeley Data public zip endpoint).
    ("dawn/DAWN.zip", "https://data.mendeley.com/api/datasets/766ygrbt8y/versions/3/zip",
     "unknown (hundreds of MB)", True),
    # MIT Traffic: only the ground truth survives online — the 20 video-clip
    # links (people.csail.mit.edu/xgwang/mv2_0XX.zip) are dead (403,
    # no mirror, not in the Wayback Machine).
    ("mit_traffic/ground_truth.tar.gz",
     "http://www.ee.cuhk.edu.hk/~xgwang/MIT_traffic_ground_truth_data.tar.gz",
     "small", True),
]

ACCOUNT_GATED = """
Left to you (account or access request required):
  - AAU RainSnow (rain/snow/night intersections, RGB+thermal, COCO GT):
      https://www.kaggle.com/datasets/aalborguniversity/aau-rainsnow  (free Kaggle account)
  - UA-DETRAC (night+rain surveillance, 1.21M boxes): Kaggle/IEEE DataPort mirrors
  - inD (11,500+ trajectories at German intersections): academic request at
      https://levelxdata.com/ind-dataset/
"""


def fetch(url: str, target: Path) -> bool:
    if target.exists() and target.stat().st_size > 0:
        print(f"  [skip] {target.name} already present "
              f"({target.stat().st_size // 1048576} MB)")
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for verify in (True, False):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=180,
                              verify=verify) as response:
                response.raise_for_status()
                received = 0
                with open(partial, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        handle.write(chunk)
                        received += len(chunk)
                        if received % (200 << 20) < (1 << 20):
                            print(f"    ... {received >> 20} MB", flush=True)
            partial.replace(target)
            note = "" if verify else "  (server certificate not verifiable!)"
            print(f"  [ok]   {target.name}: {received >> 20} MB{note}")
            return True
        except requests.exceptions.SSLError:
            if verify:
                print(f"  [warn] SSL verification failed for {url}; retrying unverified")
                continue
            break
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            print(f"  [fail] {url}: {exc}")
            break
        finally:
            if partial.exists():
                partial.unlink(missing_ok=True)
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true",
                        help="also fetch the optional large Ko-PER sequences")
    parser.add_argument("--list", action="store_true", help="list without downloading")
    args = parser.parse_args(argv)

    selected = [d for d in DOWNLOADS if args.full or d[3]]
    print(f"Target directory: {DATASETS_DIR}")
    for rel, url, size, _ in selected:
        print(f"  {rel}  ({size})")
    if args.list:
        print(ACCOUNT_GATED)
        return 0

    failures = 0
    for rel, url, size, _ in selected:
        print(f"\nFetching {rel} ({size}) ...")
        if not fetch(url, DATASETS_DIR / rel):
            failures += 1

    print(ACCOUNT_GATED)
    if failures:
        print(f"{failures} download(s) failed — re-run to retry (existing files are kept).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
