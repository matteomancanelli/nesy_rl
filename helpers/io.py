# core/io.py
import json
import os
from typing import Any, Dict

import numpy as np
import torch


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_npz(path: str, **arrays) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # ensure CPU numpy
    out = {}
    for k, v in arrays.items():
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        out[k] = v
    np.savez_compressed(path, **out)