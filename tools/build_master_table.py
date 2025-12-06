import argparse
import os
import sys
from typing import Dict

sys.path.append(os.getcwd())

import pandas as pd

from vkr_fast.config import Paths, TimeParams, TICKERS
from vkr_fast.data import fetch_moex, yahoo_csv
from vkr_fast.features import add_indicators, make_lags


def _load_raw(paths: Paths, timep: TimeParams) -> pd.DataFrame:
    """Загружаем базовый ряд IMOEX + экзогенные (Brent, USD/RUB) без ключевой ставки."""
    base = fetch_moex(
        paths.cache_dir,
        "IMOEX",
        *TICKERS["IMOEX"],
        timep.start_raw,
        timep.end_raw,
        timep.interval_minutes,
        use_cache_only=True,
    )
    brent = yahoo_csv(paths.data_dir, "Brent_yahoo_1h_2023-01-01_2025-04-01 (1).csv")
    usdr = yahoo_csv(paths.data_dir, "USD_RUB_yahoo_1h_2023-01-01_2025-04-01.csv")
    ext = pd.DataFrame(index=base.index)
    ext["Brent"] = brent["Close"].reindex(base.index).ffill()
    ext["USD"] = usdr["Close"].reindex(base.index).ffill()
    ext.bfill(limit=1, inplace=True)
    joined = base.join(ext, how="left")
    period_slice = slice(timep.period_start, timep.period_end - pd.Timedelta(seconds=1))
    return joined.loc[period_slice]


def _timeline_from_full_preds(*dirs: str) -> pd.DatetimeIndex:
    """Собираем объединённую временную ось по всем full_preds из заданных каталогов."""
    idx_list = []
    for out_dir in dirs:
        full_dir = os.path.join(out_dir, "full_preds")
        if not os.path.isdir(full_dir):
            continue
        for fname in os.listdir(full_dir):
            if not fname.endswith(".csv") or "_f" not in fname:
                continue
            df = pd.read_csv(os.path.join(full_dir, fname), parse_dates=["Datetime"])
            idx_list.append(pd.to_datetime(df["Datetime"]))
    if not idx_list:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(sorted(pd.unique(pd.concat(idx_list))))


def _load_predictions(out_dir: str, suffix: str) -> Dict[str, pd.Series]:
    """Читает full_preds и разворачивает в словарь колонок с суффиксом (_dclose_f{n} / _logret_f{n})."""
    pred_dir = os.path.join(out_dir, "full_preds")
    cols: Dict[str, pd.Series] = {}
    if not os.path.isdir(pred_dir):
        return cols
    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith(".csv") or "_f" not in fname:
            continue
        fold = fname.split("_f")[1].split(".")[0]
        df = pd.read_csv(os.path.join(pred_dir, fname), parse_dates=["Datetime"])
        df.set_index("Datetime", inplace=True)
        for col in df.columns:
            if col in {"y_true", "Datetime"}:
                continue
            cols[f"{col}{suffix}_f{fold}"] = df[col]
    return cols


def build_master(out_dir: str, dest: str) -> None:
    paths = Paths(out_dir=out_dir)
    timep = TimeParams()

    raw = _load_raw(paths, timep)
    base = add_indicators(raw)

    # Формируем supervised выборки для двух целевых переменных
    feat_d = make_lags(base.copy(), window=timep.window, exog_lags=24, target="dclose")
    feat_l = make_lags(base.copy(), window=timep.window, exog_lags=24, target="logret")

    feat_d["Datetime"] = pd.to_datetime(feat_d["Datetime"], utc=True)
    feat_l["Datetime"] = pd.to_datetime(feat_l["Datetime"], utc=True)
    feat_d.set_index("Datetime", inplace=True)
    feat_l.set_index("Datetime", inplace=True)

    # объединяем временные оси dclose/logret full_preds и исходных фич
    tl = _timeline_from_full_preds(out_dir, "outputs_logret", out_dir.replace("outputs", "outputs_logret"))
    if tl.empty:
        tl = feat_d.index
    timeline = pd.DatetimeIndex(sorted(pd.unique(tl.union(feat_d.index).union(feat_l.index))))

    master = pd.DataFrame(index=timeline)

    # Базовые уровни и факторы (по сути feat_d содержит индикаторы и лаги)
    common_cols = [c for c in feat_d.columns if c != "y"]
    master[common_cols] = feat_d.reindex(timeline)[common_cols]
    master["y_dclose"] = feat_d.reindex(timeline)["y"]
    master["y_logret"] = feat_l.reindex(timeline)["y"]
    master[common_cols] = master[common_cols].ffill()
    master[["y_dclose", "y_logret"]] = master[["y_dclose", "y_logret"]].ffill()

    # Цепные и базисные показатели по Close
    close = master["Close"]
    master["chain_increment"] = close.diff()
    master["chain_growth_coef"] = close / close.shift(1)
    master["chain_growth_pct"] = (master["chain_growth_coef"] - 1.0) * 100
    base_level = close.iloc[0]
    master["base_abs"] = close - base_level
    master["base_growth_coef"] = close / base_level
    master["base_growth_pct"] = (master["base_growth_coef"] - 1.0) * 100

    # Кластеры режимов
    cl_path = os.path.join(out_dir, "clustering", "IMOEX_clusters.csv")
    if os.path.exists(cl_path):
        cl = pd.read_csv(cl_path, parse_dates=["Datetime"])
        cl.set_index("Datetime", inplace=True)
        master["cluster"] = cl.reindex(timeline)["cluster"]
        name_map = {}
        stats_path = os.path.join(out_dir, "clustering", "IMOEX_cluster_stats.csv")
        if os.path.exists(stats_path):
            stats = pd.read_csv(stats_path)
            name_map = {int(r.Cluster): r.ClusterName for r in stats.itertuples()}
        master["cluster_name"] = master["cluster"].map(name_map) if name_map else master["cluster"]

    # Прогнозы всех моделей и всех фолдов
    pred_d = _load_predictions(out_dir, suffix="_dclose")
    for col, series in pred_d.items():
        master[col] = series.reindex(timeline)
    pred_l = _load_predictions(out_dir.replace("outputs", "outputs_logret") if out_dir == "outputs" else out_dir, suffix="_logret")
    # Если логарифмические прогнозы лежат в отдельной папке outputs_logret, подтянем их явно
    if out_dir != "outputs_logret" and os.path.isdir("outputs_logret/full_preds"):
        pred_l = _load_predictions("outputs_logret", suffix="_logret")
    for col, series in pred_l.items():
        master[col] = series.reindex(timeline)

    master.reset_index(inplace=True)
    master.rename(columns={"index": "Datetime"}, inplace=True)
    master.to_csv(dest, index=False)
    print(f"Master table saved to {dest} (rows={len(master)}, cols={len(master.columns)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs", help="Каталог с результатами (по умолчанию outputs)")
    ap.add_argument("--dest", default="outputs/master_table.csv", help="Путь для сохранения финальной таблицы")
    args = ap.parse_args()
    build_master(args.out_dir, args.dest)


if __name__ == "__main__":
    main()
