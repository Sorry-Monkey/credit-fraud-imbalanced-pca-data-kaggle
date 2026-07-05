# -*- coding: utf-8 -*-
"""Rebuild FE-8 notebook with BASE baseline and MT-4 reference."""
import json, uuid
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "credit-fraud-feature-engineering-8.ipynb"

def cell_md(text):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": [line + "
" for line in text.split("
")]}

def cell_code(text):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "outputs": [], "execution_count": None, "source": [line + "
" for line in text.split("
")]}

