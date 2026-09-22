#src/energy/config.py

"""
Accesso unico ai parametri progettuali
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

DEFAULT_PATH=Path("configs/params.yaml")

def load_params(path:Path | str=DEFAULT_PATH)->dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)