#src/energy/data/download.py

"""
Scarica l'archivio UCI household power consumption in data/raw/.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
import requests

URL=("https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip")
RAW_DIR=Path("data/raw")
TARGET=RAW_DIR/"household_power_consumption.txt"

def main()->None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(f"{TARGET} già presente, skip del download")
        return

    print("Download in corso (~20 MB)...")
    response=requests.get(URL, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extract(TARGET.name, RAW_DIR)

    print(f"Scritto {TARGET} ({TARGET.stat().st_size/1e6:.1f} MB)")

if __name__=="__main__":
    main()