"""Product catalogue loading utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .models import Product


def load_catalog(path: str | Path) -> List[Product]:
    """Load the JSON product catalogue.

    Parameters
    ----------
    path:
        Path to a JSON file containing product records.

    Returns
    -------
    list[Product]
        Catalogue entries converted to dataclasses.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    return [Product(**record) for record in records]
