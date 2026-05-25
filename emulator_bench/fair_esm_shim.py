import importlib
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENDOR_PATH = (
    REPO_ROOT.parent
    / "EMULaToR"
    / "data"
    / "processed"
    / "baselines"
    / "CatPred"
    / "embeddings"
    / "fair_esm"
)


def _vendor_path() -> Path:
    explicit = os.getenv("CATPRED_FAIR_ESM_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()

    cache_dir = os.getenv("CATPRED_BENCH_CACHE_DIR") or os.getenv("CATPRED_CACHE_PATH")
    if cache_dir:
        return (Path(cache_dir).expanduser().resolve() / "fair_esm").resolve()

    return DEFAULT_VENDOR_PATH.resolve()


def _is_fair_esm(module) -> bool:
    pretrained = getattr(module, "pretrained", None)
    return pretrained is not None and hasattr(pretrained, "esm2_t33_650M_UR50D")


def ensure_fair_esm():
    existing = sys.modules.get("esm")
    if existing is not None and _is_fair_esm(existing):
        return existing

    vendor = _vendor_path()
    if not (vendor / "esm").exists():
        raise ImportError(
            "CatPred requires Meta fair-esm for esm2_t33_650M_UR50D, but the active "
            "`esm` module is not fair-esm. Install fair-esm into the isolated CatPred "
            f"vendor path with:\npython -m pip install fair-esm -t {vendor}"
        )

    for name in list(sys.modules):
        if name == "esm" or name.startswith("esm."):
            del sys.modules[name]

    vendor_text = str(vendor)
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)

    module = importlib.import_module("esm")
    if not _is_fair_esm(module):
        raise ImportError(
            f"The isolated esm package at {vendor} does not expose "
            "esm.pretrained.esm2_t33_650M_UR50D."
        )

    return module
