import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.getcwd())
from vkr_fast.mpl_config import configure_matplotlib
from vkr_fast.plotting import apply_gost_style
from vkr_fast.config import Paths, TimeParams, TICKERS
from vkr_fast.data import fetch_moex
from vkr_fast.features import add_indicators

configure_matplotlib()

import matplotlib
matplotlib.use("Agg")


def md_h1(s):
    return f"# {s}\n\n"


def md_h2(s):
    return f"## {s}\n\n"


def md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False) + "\n\n"
    except Exception:
        # Fallback when tabulate is unavailable
        return "```\n" + df.to_string(index=False) + "\n```\n\n"


def append_indicator_formulas(rep: list) -> None:
    rep.append(md_h2("Формулы индикаторов"))
    # Краткие формулы/определения. Неформатированный блок для удобства чтения в GitHub.
    formulas = r"""
Y (цель): dClose_t = Close_{t+1} − Close_t

Value: денежный оборот за бар (MOEX Value).

SMA_n: SMA_n(t) = mean(Close_{t−n+1..t})  (n ∈ {5,10,20})
EMA_n: EMA_n(t) = α·Close_t + (1−α)·EMA_n(t−1),  α = 2/(n+1)  (n ∈ {5,10,20})

BB (Bollinger, n=20, k=2):
  BBH = SMA_20 + 2·σ_20,   BBL = SMA_20 − 2·σ_20

RSI_50: RSI = 100 − 100/(1 + RS),  RS = EMA(Gain,50)/EMA(Loss,50)

Stochastics (окно n):
  %K = 100·(Close − L_n)/(H_n − L_n),  %D = SMA(%K, 3)
  H_n = max(High_{t−n+1..t}),  L_n = min(Low_{t−n+1..t})

ATR_50: ATR = RMA(TR,50),  TR_t = max(High−Low, |High−Close_{t−1}|, |Low−Close_{t−1}|)

OBV: OBV_t = OBV_{t−1} + sign(Close_t − Close_{t−1})·Volume_t

MACD: MACD = EMA_12(Close) − EMA_26(Close)
MACD_SIGNAL = EMA_9(MACD),  MACD_DIFF = MACD − MACD_SIGNAL

ADX_14: ADX = RMA(DX,14),  DX = 100·|+DI − −DI|/(+DI + −DI)
  +DI, −DI получаются из +DM, −DM и TR по схеме Уайлдера

CCI_20: CCI = (TP − SMA(TP,20)) / (0.015·MD_20),  TP=(H+L+C)/3,
  MD_20 = mean(|TP − SMA(TP,20)|) за 20 баров

ROC_10: ROC = 100·(Close_t/Close_{t−10} − 1)
WILLR_14: −100·(H_14 − Close)/(H_14 − L_14)

Лаги цены: lag_k = Close_{t−k},  k = 1..60
Лаги экзогенных факторов: Brent_lag_k, USD_lag_k, KeyRate_lag_k = соответствующий уровень на t−k,  k = 1..24
"""
    rep.append("```\n" + formulas.strip() + "\n```\n\n")


def _regen_all_models_from_preds(pred_dir: str, out_dir: str, paths: Paths, timep: TimeParams) -> None:
    """Перестроить all_models графики по сохранённым предсказаниям (без повторного обучения)."""
    apply_gost_style()
    sns.set_theme(style="whitegrid")
    allowed = {"LSTM_att", "CatBoost", "Hybrid", "SARIMAX"}
    for pf in sorted(f for f in os.listdir(pred_dir) if f.endswith(".csv")):
        if "_f" not in pf:
            continue
        tk = pf.split("_f")[0]
        if tk not in TICKERS:
            continue
        dfp = pd.read_csv(os.path.join(pred_dir, pf))
        dt_fc = pd.to_datetime(dfp["Datetime"], utc=True)
        start_fc = dt_fc.min()
        # Загружаем исходный ряд_CLOSE для истории
        moex = fetch_moex(
            paths.cache_dir,
            tk,
            *TICKERS[tk],
            timep.start_raw,
            timep.end_raw,
            timep.interval_minutes,
            use_cache_only=True,
        )
        dclose = moex["Close"].astype(float).diff().dropna()
        hist = dclose[dclose.index < start_fc]
        # ограничим разумным хвостом для читаемости
        hist = hist.iloc[-400:]

        fig, ax = plt.subplots(figsize=(16, 5))
        # Reserve space on the right for an external legend
        plt.subplots_adjust(right=0.78)
        # Fact series with distinct, fixed colors
        ax.plot(
            hist.index,
            hist.values,
            label="Факт (до прогноза)",
            color="black",
            linewidth=2.4,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        fact_fc_color = "#1f77b4"  # matplotlib default blue
        ax.plot(
            dt_fc,
            dfp["y_true"],
            label="Факт (прогнозный отрезок)",
            color=fact_fc_color,
            linewidth=2.6,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        # Model overlays — ensure palette avoids black/blue to prevent confusion
        base_cols = {"Datetime", "y_true", "Close", "Close_prev", "Sigma"}
        cols = [c for c in dfp.columns if c not in base_cols and c in allowed]
        def _close(c1, c2, tol=0.03):
            return all(abs(a - b) <= tol for a, b in zip(c1, c2))
        reserved = [(0.0, 0.0, 0.0), (0.1216, 0.4667, 0.7059)]  # black, default blue
        pal_raw = sns.color_palette("tab20", n_colors=max(len(cols) + 4, 12))
        palette = [c for c in pal_raw if not any(_close(c, r) for r in reserved)]
        if len(palette) < len(cols):
            palette.extend(sns.color_palette("hls", n_colors=(len(cols) - len(palette) + 3)))
        for idx, c in enumerate(cols):
            ax.plot(
                dt_fc,
                dfp[c],
                label=f"Прогноз {c}",
                linewidth=2.2,
                alpha=0.95,
                solid_capstyle="round",
                solid_joinstyle="round",
                color=palette[idx % len(palette)],
            )
        ax.axvline(start_fc, color="#666666", linestyle="--", linewidth=1.3, label="Старт прогноза")
        # без заголовка
        ax.set_xlabel("Дата и время (UTC)")
        ax.set_ylabel("Значение")
        ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        # Smaller legend outside the axes to avoid overlap
        lg = ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0.0, frameon=True)
        fig.autofmt_xdate()
        fig.tight_layout()
        # Save under current name and a compatibility name used earlier
        out_path = os.path.join(out_dir, f"{pf.replace('.csv','')}_all_models.png")
        compat_name = f"{tk}_{pf.replace('.csv','')}_all_models.png"
        out_path_compat = os.path.join(out_dir, compat_name)
        fig.savefig(out_path, bbox_inches="tight")
        # also write the legacy/duplicate name for users expecting it
        fig.savefig(out_path_compat, bbox_inches="tight")
        plt.close(fig)


def _full_period_plots(pred_dir: str, tk: str, out_dir: str) -> list:
    """Build full-period fact vs model plots using all folds."""
    apply_gost_style()
    allowed = {"LSTM_att", "CatBoost", "Hybrid", "SARIMAX"}
    files = sorted([f for f in os.listdir(pred_dir) if f.startswith(f"{tk}_f") and f.endswith(".csv")])
    if not files:
        return []
    model_series = {}
    for f in files:
        df = pd.read_csv(os.path.join(pred_dir, f))
        dt = pd.to_datetime(df["Datetime"], utc=True)
        base_cols = {"Datetime", "y_true", "Close", "Close_prev", "Sigma"}
        for col in df.columns:
            if col in base_cols or col not in allowed:
                continue
            part = pd.DataFrame({"Datetime": dt, "y_true": df["y_true"], "y_hat": df[col]})
            model_series.setdefault(col, []).append(part)
    saved = []
    for mdl, parts in model_series.items():
        full = pd.concat(parts).sort_values("Datetime")
        full = full.drop_duplicates(subset="Datetime")
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(full["Datetime"], full["y_true"], label="Факт (ΔЦена)")
        ax.plot(full["Datetime"], full["y_hat"], label=f"Прогноз ({mdl})")
        # без заголовка
        ax.set_xlabel("Дата и время (UTC)")
        ax.set_ylabel("Значение")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{tk}_full_{mdl}.png")
        fig.savefig(out_path)
        plt.close(fig)
        saved.append(out_path)
    return saved


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='outputs')
    ap.add_argument('--report-path', default='REPORT.md', help='Путь для сохранения markdown-отчёта')
    args = ap.parse_args()
    out_dir = args.out_dir
    report_path = args.report_path
    cons_dir = os.path.join(out_dir, "consolidated")
    paths = Paths()
    timep = TimeParams()
    # Перестроим all_models графики из предсказаний (толще, шире, без переобучения)
    pred_dir = os.path.join(out_dir, "preds")
    rep_dir = os.path.join(out_dir, "reports")
    if os.path.isdir(pred_dir):
        os.makedirs(rep_dir, exist_ok=True)
        _regen_all_models_from_preds(pred_dir, rep_dir, paths, timep)
    rep = []
    rep.append(md_h1("Отчёт по моделям прогнозирования (почасовые данные)"))
    rep.append("Версия: автогенерация из артефактов в папке outputs.\n\n")

    rep.append(md_h2("Методология без утечек для IMOEX"))
    rep.append(
        "Прогнозируем почасовой ряд индекса IMOEX, добавляя экзогенные факторы только в виде лагов. "
        "Используемые формулы:\n\n"
        "$$r_t = \\mu + \\beta^\\top X_{t-1} + \\varepsilon_t, \\qquad "
        "\\sigma_t^2 = \\omega + \\alpha \\varepsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2 + \\gamma^\\top X_{t-1}.$$\n\n"
        "Вектор $X_{t-1}$ содержит лаги Brent, USD/RUB, ключевой ставки, RSI(50), ATR(50) и лаги цены. "
        "Параметры CatBoost/RF и LSTM обучаются только на стандартизованных лаговых матрицах; "
        "гибридная LSTM использует прогноз SARIMAX и σ_t из GARCH как признаки риска. "
        "Ни один признак не знает будущего, эмбарго и walk-forward разбиения исключают утечки.\n\n"
    )
    feature_spec = os.path.join(out_dir, "feature_space.json")
    if os.path.exists(feature_spec):
        with open(feature_spec, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
        feats = spec.get("features", [])
        parts = ["Y=" + spec.get("target", "dClose")]
        parts += [f"X{i+1}={name}" for i, name in enumerate(feats)]
        rep.append(md_h2("Матрица признаков (Y и Xi)"))
        rep.append("```\n" + " | ".join(parts) + "\n```\n\n")

    # Summary metrics
    m_path = os.path.join(cons_dir, "metrics_all.csv")
    e_path = os.path.join(cons_dir, "economics_all.csv")
    if not (os.path.exists(m_path) and os.path.exists(e_path)):
        rep.append("Не найдены консолидированные файлы metrics_all.csv и economics_all.csv. Сначала запустите конвейер и консолидацию.\n")
    else:
        m = pd.read_csv(m_path)
        e = pd.read_csv(e_path)
        if "Tk" in m.columns:
            m = m[m["Tk"] == "IMOEX"].reset_index(drop=True)
        if "Tk" in e.columns:
            e = e[e["Tk"] == "IMOEX"].reset_index(drop=True)
        cols = {"MAE": "mean", "RMSE": "mean"}
        if "MASE" in m.columns:
            cols["MASE"] = "mean"
        m_agg = m.groupby(["Tk", "Model"]).agg(**{k: (k, v) for k, v in cols.items()}).reset_index()
        e_agg = e.groupby(["Tk", "Model"]).agg(CumRet=("CumRet", "mean"), MaxDD=("MaxDD", "mean")).reset_index()
        res = m_agg.merge(e_agg, on=["Tk", "Model"], how="left")
        rep.append(md_h2("Сводные метрики по тикерам и моделям"))
        rep.append(md_table(res.sort_values(["Tk", "MAE"])) )

        rep.append(md_h2("Лучшие модели по MAE на тикер"))
        best_mae = res.loc[res.groupby("Tk")["MAE"].idxmin()].reset_index(drop=True)
        rep.append(md_table(best_mae))

        if "MASE" in res.columns:
            rep.append(md_h2("Лучшие модели по MASE на тикер"))
            best_mase = res.loc[res.groupby("Tk")["MASE"].idxmin()].reset_index(drop=True)
            rep.append(md_table(best_mase))

    # DM tests (по метрикам)
    dm_path = os.path.join(cons_dir, "dm_test_pairs_metrics.csv")
    if os.path.exists(dm_path):
        dm = pd.read_csv(dm_path)
        if "Tk" in dm.columns:
            dm = dm[dm["Tk"] == "IMOEX"].reset_index(drop=True)
        rep.append(md_h2("Попарные DM-тесты по метрикам"))
        # показываем только MAE/RMSE/MAPE/WAPE/sMAPE/MdAPE, если много — отсортировать по p
        keep = ["MAE","RMSE","MAPE","WAPE","SMAPE","MDAPE"]
        if "Metric" in dm.columns:
            dm = dm[dm["Metric"].isin(keep)]
            dm = dm.sort_values(["Metric", "p_val"])
        rep.append(md_table(dm))

    feat_imp_path = os.path.join(out_dir, "feature_importance.csv")
    if os.path.exists(feat_imp_path):
        fi = pd.read_csv(feat_imp_path)
        if "Tk" in fi.columns:
            fi = fi[fi["Tk"] == "IMOEX"]
        agg = fi.groupby("Feature")["Importance"].mean().reset_index().sort_values("Importance", ascending=False)
        rep.append(md_h2("Анализ важности факторов"))
        rep.append(
            "Средняя значимость признаков усреднена по фолдам и моделям (RF, CatBoost). "
            "Список ограничен десятью ключевыми факторами.\n\n"
        )
        rep.append(md_table(agg.head(10)))

        # Визуализации топ-5 важнейших признаков (overall и по моделям)
        apply_gost_style()
        os.makedirs(rep_dir := os.path.join(out_dir, "reports"), exist_ok=True)
        import matplotlib.pyplot as _plt
        import seaborn as _sns
        _sns.set_theme(style="whitegrid")

        # Overall top-5
        top_overall = agg.head(5)
        fig, ax = _plt.subplots(figsize=(8, 4))
        _sns.barplot(
            data=top_overall,
            y="Feature",
            x="Importance",
            palette="Blues_r",
            ax=ax,
        )
        ax.set_title("Топ‑5 факторов (средняя важность)")
        ax.set_xlabel("Mean importance")
        ax.set_ylabel("")
        fig.tight_layout()
        fi_overall_path = os.path.join(rep_dir, "feature_importance_top5_overall.png")
        fig.savefig(fi_overall_path, bbox_inches="tight")
        _plt.close(fig)
        rep.append(f"![feature_importance_top5_overall](outputs/reports/{os.path.basename(fi_overall_path)})\n\n")

        # Per-model top-5 (RF, CatBoost if присутствуют)
        if "Model" in fi.columns:
            for mdl in sorted(fi["Model"].unique()):
                mdi = fi[fi["Model"] == mdl].groupby("Feature")["Importance"].mean().reset_index().sort_values("Importance", ascending=False).head(5)
                if mdi.empty:
                    continue
                fig, ax = _plt.subplots(figsize=(8, 4))
                _sns.barplot(data=mdi, y="Feature", x="Importance", palette="Greens_r", ax=ax)
                ax.set_title(f"Топ‑5 факторов ({mdl})")
                ax.set_xlabel("Mean importance")
                ax.set_ylabel("")
                fig.tight_layout()
                path_m = os.path.join(rep_dir, f"feature_importance_top5_{mdl}.png")
                fig.savefig(path_m, bbox_inches="tight")
                _plt.close(fig)
                rep.append(f"![feature_importance_top5_{mdl}](outputs/reports/{os.path.basename(path_m)})\n\n")

            # Построим отдельные временные ряды топ‑5 индикаторов (по важности среди индикаторов)
            indicator_names = [
                "SMA_5","SMA_10","SMA_20","EMA_5","EMA_10","EMA_20","BBH","BBL","RSI_50",
                "StochK","StochD","ATR_50","OBV","MACD","MACD_SIGNAL","MACD_DIFF","ADX_14","CCI_20","ROC_10","WILLR_14"
            ]
            agg_ind = agg[agg["Feature"].isin(indicator_names)].head(5)
            if not agg_ind.empty:
                rep.append(md_h2("Временные ряды топ‑5 индикаторов"))
                # загрузим историю IMOEX и рассчитаем индикаторы без сети (из кэша)
                tk = "IMOEX"
                moex = fetch_moex(
                    paths.cache_dir, tk, *TICKERS[tk], timep.start_raw, timep.end_raw, timep.interval_minutes, use_cache_only=True
                )
                ind_df = add_indicators(moex)
                ts_slice = slice(timep.period_start, timep.period_end)
                os.makedirs(rep_dir, exist_ok=True)
                import matplotlib.pyplot as _plt
                apply_gost_style()
                # Отдельный график Value
                if "Value" in ind_df.columns:
                    v = ind_df["Value"].loc[ts_slice].dropna()
                    if not v.empty:
                        fig, ax = _plt.subplots(figsize=(11, 4))
                        ax.plot(v.index, v.values, label="Value", color="#ff7f0e", linewidth=1.8, alpha=0.8, solid_capstyle="round", solid_joinstyle="round")
                        ax.set_xlabel("Дата и время (UTC)")
                        ax.set_ylabel("Value")
                        ax.legend(loc="upper left", fontsize=9)
                        fig.autofmt_xdate()
                        fig.tight_layout()
                        fname_v = "indicator_series_Value.png"
                        fpath_v = os.path.join(rep_dir, fname_v)
                        fig.savefig(fpath_v, bbox_inches="tight")
                        _plt.close(fig)
                        rep.append(f"![Value](outputs/reports/{fname_v})\n\n")

                for feat in agg_ind["Feature"].tolist():
                    s = ind_df[feat].loc[ts_slice].dropna()
                    if s.empty:
                        continue
                    fig, ax = _plt.subplots(figsize=(11, 4))
                    # Основной ряд: индикатор
                    ax.plot(
                        s.index,
                        s.values,
                        label=feat,
                        color="#1f77b4",
                        linewidth=2.2,
                        solid_capstyle="round",
                        solid_joinstyle="round",
                    )
                    ax.legend(loc="upper left", fontsize=9)

                    # Подпись метрик индикатора (среднее и σ)
                    mu = float(s.mean())
                    sd = float(s.std())
                    ax.text(
                        0.99,
                        0.98,
                        f"μ = {mu:.3g}\nσ = {sd:.3g}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=9,
                        bbox=dict(facecolor="white", edgecolor="#666666", alpha=0.8, boxstyle="round,pad=0.3"),
                    )

                    ax.set_xlabel("Дата и время (UTC)")
                    ax.set_ylabel(feat)
                    fig.autofmt_xdate()
                    fig.tight_layout()
                    fname = f"indicator_series_{feat}.png"
                    fpath = os.path.join(rep_dir, fname)
                    fig.savefig(fpath, bbox_inches="tight")
                    _plt.close(fig)
                    rep.append(f"![{feat}](outputs/reports/{fname})\n\n")

        # Добавим формулы индикаторов ниже этого раздела
        append_indicator_formulas(rep)
    else:
        # Если важности не найдены, всё равно добавим формулы индикаторов
        append_indicator_formulas(rep)

    analysis_dir = os.path.join(out_dir, "imoex_analysis")
    if os.path.isdir(analysis_dir):
        rep.append(md_h2("Цепной анализ IMOEX"))
        chain_csv = os.path.join(analysis_dir, "IMOEX_chain_growth_D.csv")
        summary_csv = os.path.join(analysis_dir, "IMOEX_chain_growth_summary.csv")
        stats_json = os.path.join(analysis_dir, "IMOEX_chain_growth_stats.json")
        if os.path.exists(summary_csv):
            rep.append("Сводка по месячным цепным приращениям (фрагмент):\n\n")
            rep.append(md_table(pd.read_csv(summary_csv).head(12)))
        if os.path.exists(chain_csv):
            rep.append("Дневные цепные приросты (фрагмент):\n\n")
            rep.append(md_table(pd.read_csv(chain_csv).head(12)))
            rep.append("Файл включает цепные/базисные относительные и абсолютные индексы и коэффициент структурного сдвига.\n\n")
        if os.path.exists(stats_json):
            stats = pd.read_json(stats_json, typ="series")
            rep.append("Ключевые показатели цепного роста:\n\n")
            rep.append(md_table(stats.reset_index().rename(columns={"index": "Metric", 0: "Value"})))
        fig_path = os.path.join(analysis_dir, "IMOEX_chain_growth.png")
        if os.path.exists(fig_path):
            rep.append("![IMOEX_chain_growth](outputs/imoex_analysis/IMOEX_chain_growth.png)\n\n")
        for extra in ("IMOEX_trend_component.png", "IMOEX_seasonal_component.png", "IMOEX_acf_chain.png", "IMOEX_pacf_chain.png"):
            epath = os.path.join(analysis_dir, extra)
            if os.path.exists(epath):
                rep.append(f"![{extra}](outputs/imoex_analysis/{extra})\n\n")

    # Sharpe confidence intervals
    # Sharpe CI раздел убран по запросу

    # Threshold optimization (optional)
    thr_path = os.path.join(cons_dir, "thresholds.csv")
    econ_opt_path = os.path.join(cons_dir, "economics_opt.csv")
    thr_hold = os.path.join(cons_dir, "thresholds_holdout.csv")
    econ_hold = os.path.join(cons_dir, "economics_opt_holdout.csv")
    if os.path.exists(thr_path):
        thr = pd.read_csv(thr_path)
        if "Tk" in thr.columns:
            thr = thr[thr["Tk"] == "IMOEX"]
        rep.append(md_h2("Оптимизация порога торговли"))
        rep.append("Порог no-trade подбирается по квантилям |прогноза| для каждого тикера/модели, максимизируя средний CumRet по фолдам.\n\n")
        rep.append(md_table(thr.sort_values(["Tk", "Model"])))
        if os.path.exists(econ_opt_path):
            eopt = pd.read_csv(econ_opt_path)
            if "Tk" in eopt.columns:
                eopt = eopt[eopt["Tk"] == "IMOEX"]
            rep.append("Средние экономические показатели при оптимальном пороге (по фолдам):\n\n")
            eopt_agg = eopt.groupby(["Tk","Model"]).agg(Threshold=("Threshold","median"), CumRet_mean=("CumRet","mean"), MaxDD_mean=("MaxDD","mean")).reset_index()
            rep.append(md_table(eopt_agg.sort_values(["Tk","CumRet_mean"], ascending=[True, False]).head(30)))
    # Holdout optimization (val/test split within fold file)
    if os.path.exists(thr_hold):
        thr = pd.read_csv(thr_hold)
        if "Tk" in thr.columns:
            thr = thr[thr["Tk"] == "IMOEX"]
        rep.append(md_h2("Оптимизация порога торговли (holdout)"))
        rep.append("Порог подбирается на валидационной части каждого файла предсказаний; оценка проводится на отложенной части.\n\n")
        rep.append(md_table(thr.sort_values(["Tk", "Model"])) )
        if os.path.exists(econ_hold):
            eopt = pd.read_csv(econ_hold)
            if "Tk" in eopt.columns:
                eopt = eopt[eopt["Tk"] == "IMOEX"]
            rep.append("Экономические показатели на тестовой части при holdout‑порогах:\n\n")
            eopt_agg = eopt.groupby(["Tk","Model"]).agg(Threshold=("Threshold","median"), CumRet_mean=("CumRet","mean"), MaxDD_mean=("MaxDD","mean")).reset_index()
            rep.append(md_table(eopt_agg.sort_values(["Tk","CumRet_mean"], ascending=[True, False]).head(30)))

    # Figures (link a few examples)
    rep.append(md_h2("Примеры графиков"))
    rep.append("Графики факта/прогнозов, ACF остатков, гистограммы и кривые капитала — см. папку `outputs/reports/`.\n\n")
    # Try to include few links
    rep_dir = os.path.join(out_dir, "reports")
    if os.path.isdir(rep_dir):
        imgs = [f for f in os.listdir(rep_dir) if f.endswith('.png')]
        for f in sorted(imgs)[:6]:
            rep.append(f"![{f}](reports/{f})\n\n")
    # Full-period plots (все фолды)
    full_pred_dir = os.path.join(out_dir, "full_preds")
    if not os.path.isdir(full_pred_dir):
        full_pred_dir = os.path.join(out_dir, "preds")
    full_figs = _full_period_plots(full_pred_dir, "IMOEX", rep_dir)
    if full_figs:
        rep.append(md_h2("Факт и прогноз на всём горизонте (все фолды)"))
        for p in full_figs:
            fname = os.path.relpath(p, ".")
            rep.append(f"![{os.path.basename(fname)}]({fname})\n\n")

    # Save report
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("".join(rep))
    print("REPORT.md создан.")

    # EDA section (append to REPORT.md if EDA exists)
    eda_dir = os.path.join("outputs", "eda")
    if os.path.isdir(eda_dir):
        eda = []
        eda.append(md_h2("EDA по рядам (описательная статистика и тесты)"))
        summ = os.path.join(eda_dir, "summary.csv")
        if os.path.exists(summ):
            sdf = pd.read_csv(summ)
            if "Tk" in sdf.columns:
                sdf = sdf[sdf["Tk"] == "IMOEX"]
            # покажем верхние строки для наглядности (полный CSV — в папке)
            eda.append(md_table(sdf.head(20)))
        # Картинки
        imgs = [f for f in os.listdir(eda_dir) if f.endswith('.png')]
        for f in sorted(imgs):
            eda.append(f"![{f}](outputs/eda/{f})\n\n")
        with open(report_path, "a", encoding="utf-8") as fh:
            fh.write("".join(eda))

    # Clustering section
    cl_dir = os.path.join("outputs", "clustering")
    if os.path.isdir(cl_dir):
        cl = []
        cl.append(md_h2("Кластеризация режимов по тикерам"))
        summ = os.path.join(cl_dir, "summary.csv")
        if os.path.exists(summ):
            sdf = pd.read_csv(summ)
            if "Tk" in sdf.columns:
                sdf = sdf[sdf["Tk"] == "IMOEX"]
            cl.append(md_table(sdf))
        imgs = [f for f in os.listdir(cl_dir) if f.endswith('_clusters_price.png')]
        pies = [f for f in os.listdir(cl_dir) if f.endswith('_cluster_pie.png')]
        for f in sorted(imgs):
            cl.append(f"![{f}](outputs/clustering/{f})\n\n")
        for f in sorted(pies):
            cl.append(f"![{f}](outputs/clustering/{f})\n\n")
        with open(report_path, "a", encoding="utf-8") as fh:
            fh.write("".join(cl))

    # VAR/VARMAX section
    var_dir = os.path.join("outputs", "var")
    if os.path.isdir(var_dir):
        var = []
        var.append(md_h2("VAR/VARMAX: многомерное моделирование"))
        mpt = os.path.join(var_dir, "metrics_per_ticker.csv")
        magg = os.path.join(var_dir, "metrics_aggregate.csv")
        dmp = os.path.join(var_dir, "dm_vs_sarimax.csv")
        dmb = os.path.join(var_dir, "dm_vs_best.csv")
        ordp = os.path.join(var_dir, "order_selection.csv")
        if os.path.exists(mpt):
            var.append("Метрики по тикерам (первые строки):\n\n")
            var.append(md_table(pd.read_csv(mpt).head(12)))
        if os.path.exists(magg):
            var.append("Агрегированные метрики по фолдам:\n\n")
            var.append(md_table(pd.read_csv(magg)))
        if os.path.exists(dmp):
            var.append("Сравнение с SARIMAX по DM (первые строки):\n\n")
            var.append(md_table(pd.read_csv(dmp).head(20)))
        if os.path.exists(dmb):
            var.append("Сравнение с лучшей моделью (§ 3.2) по DM (первые строки):\n\n")
            var.append(md_table(pd.read_csv(dmb).head(20)))
        if os.path.exists(ordp):
            var.append("Выбор порядков и диагностические показатели (первые строки):\n\n")
            var.append(md_table(pd.read_csv(ordp).head(10)))
        # IRF и FEVD файлы (перечень)
        irf_imgs = [f for f in os.listdir(var_dir) if f.startswith('irf_') and f.endswith('.png')]
        if irf_imgs:
            var.append("Импульсные отклики (IRF) — примеры:\n\n")
            for f in sorted(irf_imgs)[:5]:
                var.append(f"![{f}](outputs/var/{f})\n\n")
        roots_imgs = [f for f in os.listdir(var_dir) if f.startswith('roots_') and f.endswith('.png')]
        if roots_imgs:
            var.append("Спектры модулей корней VAR — примеры:\n\n")
            for f in sorted(roots_imgs)[:5]:
                var.append(f"![{f}](outputs/var/{f})\n\n")
        with open(report_path, "a", encoding="utf-8") as fh:
            fh.write("".join(var))


if __name__ == "__main__":
    main()
