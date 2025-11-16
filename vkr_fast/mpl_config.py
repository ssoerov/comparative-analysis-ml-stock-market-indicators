import os
from pathlib import Path


def configure_matplotlib():
    """Force headless backend and writable config dir."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    cfg = os.environ.get("MPLCONFIGDIR")
    if not cfg:
        cfg = os.path.join(os.getcwd(), ".mpl-cache")
        os.environ["MPLCONFIGDIR"] = cfg
    Path(cfg).mkdir(parents=True, exist_ok=True)


__all__ = ["configure_matplotlib"]
