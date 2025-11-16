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


def _regen_all_models_from_preds(pred_dir: str, out_dir: str, paths: Paths, timep: TimeParams) -> None:
    """Перестроить all_models графики по сохранённым предсказаниям (без повторного обучения)."""
    apply_gost_style()
    sns.set_theme(style="whitegrid")
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
        ax.plot(hist.index, hist.values, label="Факт (до прогноза)", color="black", linewidth=2.4)
        ax.plot(dt_fc, dfp["y_true"], label="Факт (прогнозный отрезок)", color="#1f77b4", linewidth=2.6)
        base_cols = {"Datetime", "y_true", "Close", "Close_prev", "Sigma"}
        cols = [c for c in dfp.columns if c not in base_cols]
        palette = sns.color_palette("tab10", n_colors=max(len(cols), 3))
        for idx, c in enumerate(cols):
            ax.plot(dt_fc, dfp[c], label=f"Прогноз {c}", linewidth=2.6, alpha=0.9, solid_capstyle="round", color=palette[idx % len(palette)])
        ax.axvline(start_fc, color="#666666", linestyle="--", linewidth=1.4, label="Старт прогноза")
        ax.set_title(f"{tk}: факт и прогнозы всех моделей (фолд {pf.split('_f')[1].split('.')[0]})")
        ax.set_xlabel("Дата и время (UTC)")
        ax.set_ylabel("ΔЦена")
        ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{pf.replace('.csv','')}_all_models.png")
        fig.savefig(out_path)
        plt.close(fig)


def _full_period_plots(pred_dir: str, tk: str, out_dir: str) -> list:
    """Build full-period fact vs model plots using all folds."""
    apply_gost_style()
    files = sorted([f for f in os.listdir(pred_dir) if f.startswith(f"{tk}_f") and f.endswith(".csv")])
    if not files:
        return []
    model_series = {}
    for f in files:
        df = pd.read_csv(os.path.join(pred_dir, f))
        dt = pd.to_datetime(df["Datetime"], utc=True)
        base_cols = {"Datetime", "y_true", "Close", "Close_prev", "Sigma"}
        for col in df.columns:
            if col in base_cols:
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
        ax.set_title(f"{tk}: факт и прогноз модели {mdl} на всём горизонте")
        ax.set_xlabel("Дата и время (UTC)")
        ax.set_ylabel("Изменение цены (ΔЦена)")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{tk}_full_{mdl}.png")
        fig.savefig(out_path)
        plt.close(fig)
        saved.append(out_path)
    return saved


def main():
    out_dir = "outputs"
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
        cols = {"MAE":"mean","RMSE":"mean","MAPE":"mean","WAPE":"mean","sMAPE":"mean","MdAPE":"mean"}
        if "MASE" in m.columns:
            cols["MASE"] = "mean"
        m_agg = m.groupby(["Tk", "Model"]).agg(**{k:(k,v) for k,v in cols.items()}).reset_index()
        e_agg = e.groupby(["Tk", "Model"]).agg(CumRet=("CumRet", "mean"), MaxDD=("MaxDD", "mean")).reset_index()
        res = m_agg.merge(e_agg, on=["Tk", "Model"], how="left")
        rep.append(md_h2("Сводные метрики по тикерам и моделям"))
        rep.append(md_table(res.sort_values(["Tk", "MAE"])) )

        rep.append(md_h2("Лучшие модели по MAE на тикер"))
        best_mae = res.loc[res.groupby("Tk")["MAE"].idxmin()].reset_index(drop=True)
        rep.append(md_table(best_mae))

        rep.append(md_h2("Лучшие модели по WAPE на тикер"))
        best_wape = res.loc[res.groupby("Tk")["WAPE"].idxmin()].reset_index(drop=True)
        rep.append(md_table(best_wape))

    # DM tests
    dm_path = os.path.join(cons_dir, "dm_test_pairs_all.csv")
    if os.path.exists(dm_path):
        dm = pd.read_csv(dm_path)
        if "Tk" in dm.columns:
            dm = dm[dm["Tk"] == "IMOEX"].reset_index(drop=True)
        rep.append(md_h2("Попарные DM-тесты"))
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
    full_figs = _full_period_plots(os.path.join(out_dir, "preds"), "IMOEX", rep_dir)
    if full_figs:
        rep.append(md_h2("Факт и прогноз на всём горизонте (все фолды)"))
        for p in full_figs:
            fname = os.path.relpath(p, ".")
            rep.append(f"![{os.path.basename(fname)}]({fname})\n\n")

    # Save report
    with open("REPORT.md", "w", encoding="utf-8") as fh:
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
        with open("REPORT.md", "a", encoding="utf-8") as fh:
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
        with open("REPORT.md", "a", encoding="utf-8") as fh:
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
        with open("REPORT.md", "a", encoding="utf-8") as fh:
            fh.write("".join(var))


if __name__ == "__main__":
    main()
