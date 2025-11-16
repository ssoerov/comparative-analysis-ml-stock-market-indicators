import os
import random
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
try:
    import tensorflow as tf  # optional; may be unavailable when --no-tf
except Exception:  # pragma: no cover
    tf = None


@dataclass(frozen=True)
class Paths:
    data_dir: str = "data_input"
    cache_dir: str = "data_cache"
    out_dir: str = "outputs"
    model_dir: str = "saved_models"


@dataclass(frozen=True)
class TimeParams:
    interval_minutes: int = 60
    window: int = 60
    start_raw: pd.Timestamp = pd.Timestamp("2023-01-03", tz="UTC")
    end_raw: pd.Timestamp = pd.Timestamp("2025-04-01", tz="UTC")
    period_start: pd.Timestamp = pd.Timestamp("2023-04-03", tz="UTC")
    period_end: pd.Timestamp = pd.Timestamp("2025-04-03", tz="UTC")


@dataclass(frozen=True)
class CVParams:
    outer_folds: int = 5
    test_horizon: int = 120


@dataclass(frozen=True)
class TradingParams:
    fee: float = 0.002
    slippage: float = 0.0
    threshold: float = 0.0  # no-trade band around zero


@dataclass(frozen=True)
class LSTMParams:
    epochs: int = 12
    batch_size: int = 128


TICKERS: Dict[str, Tuple[str, str, str]] = {
    "IMOEX": ("index", "MOEX", "IMOEX"),
}


def ensure_dirs(paths: Paths) -> None:
    for d in (paths.data_dir, paths.cache_dir, paths.out_dir, paths.model_dir):
        os.makedirs(d, exist_ok=True)


def setup_environment(seed: int = 42, omp_threads: int = 4, use_single_gpu: bool = True) -> None:
    # Determinism and seeds
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)

    # matplotlib default DPI is set externally by consumer if needed

    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.keras.utils.set_random_seed(seed)
        tf.random.set_seed(seed)

        # TensorFlow threading
        try:
            tf.config.threading.set_inter_op_parallelism_threads(omp_threads)
            tf.config.threading.set_intra_op_parallelism_threads(omp_threads)
            tf.get_logger().setLevel("ERROR")
        except Exception:
            pass

        # GPU setup
        try:
            gpus = tf.config.list_physical_devices("GPU")
        except Exception:
            gpus = []
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:  # pragma: no cover — TF runtime dependent
                    pass
            if use_single_gpu:
                try:
                    logical_gpus = tf.config.list_logical_devices("GPU")
                    tf.config.set_visible_devices(logical_gpus[:1], "GPU")
                except Exception:  # pragma: no cover
                    pass


def cpu_count_limited(max_cpus: int = 4) -> int:
    return min(max_cpus, os.cpu_count() or 1)


def default_logger(name: str = "vkr_fast") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
