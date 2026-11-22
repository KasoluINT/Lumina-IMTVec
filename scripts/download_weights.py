#!/usr/bin/env python3
"""Download the model weights (about 390 MB) into ./weights.

Cross-platform twin of download_weights.sh, so a native Windows install (no Docker)
has the same one-liner:

    python scripts/download_weights.py [--force]

Uses whatever proxy `HTTP_PROXY` / `HTTPS_PROXY` point at, if set.
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "weights"

FILES = [
    # Real-ESRGAN anime video model (BSD-3-Clause, xinntao/Real-ESRGAN): AI de-grain
    ("realesr-animevideov3.pth",
     "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"),
    # ControlNet lineart_anime annotator (lllyasviel): line layer, "anime" model
    ("netG.pth", "https://huggingface.co/lllyasviel/Annotators/resolve/main/netG.pth"),
    # MangaLineExtraction (MIT): line layer v3 / --fine, "manga" model
    ("erika.pth", "https://huggingface.co/lllyasviel/Annotators/resolve/main/erika.pth"),
]


def _download(url, dst):
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "lumina-vec-weights"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                sys.stdout.write(f"\r    {pct:3d}%  {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
            else:
                sys.stdout.write(f"\r    {done / 1e6:7.1f} MB")
            sys.stdout.flush()
    print()
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser(description="Download Lumina-Vec model weights")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    ap.add_argument("--dir", default=str(DEST), help=f"target directory (default: {DEST})")
    a = ap.parse_args()
    dest = Path(a.dir)
    dest.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, url in FILES:
        dst = dest / name
        if dst.is_file() and dst.stat().st_size > 1024 * 1024 and not a.force:
            print(f"{name}: already present ({dst.stat().st_size / 1e6:.1f} MB), skipping")
            continue
        print(f"{name}: downloading")
        try:
            _download(url, dst)
            print(f"{name}: ok ({dst.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:  # noqa: BLE001
            print(f"{name}: FAILED - {e}")
            failed.append(name)
    if failed:
        print("\nSome weights could not be downloaded: " + ", ".join(failed))
        print("If the network blocks GitHub/HuggingFace, set HTTPS_PROXY and retry.")
        return 1
    print(f"\nAll weights ready in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
