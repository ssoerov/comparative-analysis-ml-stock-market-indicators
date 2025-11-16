import os
import sys
import re
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
from vkr_fast.mpl_config import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
from vkr_fast.config import Paths


RU_TICKER = {
    'IMOEX': 'Индекс МосБиржи',
}

RU_MODEL = {
    'RF': 'Случайный лес',
    'SARIMAX': 'SARIMAX',
    'CatBoost': 'CatBoost',
    'LSTM_base': 'LSTM (базовая)',
    'LSTM_att': 'LSTM (с вниманием)',
    'Hybrid': 'Гибридная LSTM+SARIMAX',
}


def _ensure(d: str):
    os.makedirs(d, exist_ok=True)


def _safe_name(s: str) -> str:
    # replace spaces with underscores; keep Cyrillic
    s = s.replace(' ', '_')
    s = re.sub(r'[\\/:*?"<>|]', '_', s)
    return s


def _concat_preds_for_model(paths: Paths, ticker: str, model: str) -> pd.DataFrame:
    pred_dir = os.path.join(paths.out_dir, 'preds')
    files = sorted([f for f in os.listdir(pred_dir) if f.startswith(f"{ticker}_f") and f.endswith('.csv')])
    dfs: List[pd.DataFrame] = []
    for f in files:
        dfp = pd.read_csv(os.path.join(pred_dir, f))
        if model not in dfp.columns:
            continue
        dfp['Datetime'] = pd.to_datetime(dfp['Datetime'], utc=True)
        dfs.append(dfp[['Datetime', 'y_true', model]].dropna())
    if not dfs:
        return pd.DataFrame()
    dfc = pd.concat(dfs).sort_values('Datetime').drop_duplicates('Datetime')
    return dfc


def _plot_forecast_all_models(paths: Paths, out_dir: str):
    met_path = os.path.join(paths.out_dir, 'metrics.csv')
    if not os.path.exists(met_path):
        return
    m = pd.read_csv(met_path)
    tickers = sorted(m['Tk'].unique())
    models = sorted(m['Model'].unique())
    fig_dir = os.path.join(out_dir, 'figures')
    _ensure(fig_dir)
    for tk in tickers:
        name_tk = RU_TICKER.get(tk, tk)
        for mdl in models:
            dfc = _concat_preds_for_model(paths, tk, mdl)
            if dfc.empty:
                continue
            plt.figure(figsize=(10, 4))
            plt.plot(dfc['Datetime'], dfc['y_true'], label='Факт (ΔЦена)')
            plt.plot(dfc['Datetime'], dfc[mdl], label=f"Прогноз ({RU_MODEL.get(mdl, mdl)})")
            plt.title(f"{name_tk}: Итоговая траектория факта и прогноза модели {RU_MODEL.get(mdl, mdl)}")
            plt.xlabel('Дата и время (UTC)')
            plt.ylabel('Изменение цены (ΔЦена)')
            plt.legend()
            plt.tight_layout()
            fname = _safe_name(f"Рисунок_3_12_{name_tk}_Факт_и_Прогноз_{RU_MODEL.get(mdl, mdl)}.png")
            plt.savefig(os.path.join(fig_dir, fname))
            plt.close()


def _copy_existing_figures(paths: Paths, out_dir: str):
    figures = os.path.join(out_dir, 'figures')
    tables = os.path.join(out_dir, 'tables')
    _ensure(figures); _ensure(tables)
    # Главa 2 фигуры (из outputs/figs)
    src = os.path.join(paths.out_dir, 'figs')
    if os.path.isdir(src):
        name_map = {
            'close': 'Динамика_цены_закрытия',
            'dret': 'Динамика_лог-доходностей',
            'stl': 'Разложение_ряда',
            'roll_stats': 'Скользящие_статистики',
            'hist': 'Эмпирическое_распределение',
            'qq': 'Q–Q_график',
            'acf': 'АКФ',
            'pacf': 'ЧАКФ',
            'arch_acf': 'ARCH_коррелограмма',
            'subperiod_vol': 'Сравнение_режимов_волатильности_по_подвыборкам',
        }
        for f in os.listdir(src):
            if not f.lower().endswith(('.png', '.csv')):
                continue
            # Переименуем g2_* файлы в Рисунок_2.*
            # Пример: IMOEX_g2_1_close.png
            if '_g2_' in f and f.lower().endswith('.png'):
                tk, rest = f.split('_g2_')
                parts = rest.split('_')
                code = parts[0]
                kind = '_'.join(parts[1:]).replace('.png', '')
                ru_tk = RU_TICKER.get(tk, tk)
                label = name_map.get(kind, kind)
                dst = f"Рисунок_2_{code}_{_safe_name(ru_tk)}_{label}.png"
                shutil.copy2(os.path.join(src, f), os.path.join(figures, dst))
            else:
                # Прочее копируем как есть
                shutil.copy2(os.path.join(src, f), os.path.join(figures, f))
    # Кластеризация
    cl = os.path.join(paths.out_dir, 'clustering')
    if os.path.isdir(cl):
        for f in os.listdir(cl):
            if f.endswith('_clusters_price.png'):
                tk = f.split('_')[0]
                ru = RU_TICKER.get(tk, tk)
                dst = _safe_name(f"Рисунок_2_10_{ru}_Кластеры_волатильности.png")
                shutil.copy2(os.path.join(cl, f), os.path.join(figures, dst))
    # VAR IRF и корни
    var = os.path.join(paths.out_dir, 'var')
    if os.path.isdir(var):
        for f in os.listdir(var):
            if f.startswith('irf_') and f.endswith('.png'):
                shutil.copy2(os.path.join(var, f), os.path.join(figures, f"Рисунок_3_10_{f}"))
            if f.startswith('roots_') and f.endswith('.png'):
                shutil.copy2(os.path.join(var, f), os.path.join(figures, f"Рисунок_3_11_{f}"))
    # Таблицы — сводные
    cons = os.path.join(paths.out_dir, 'consolidated')
    if os.path.isdir(cons):
        mapping = {
            'metrics_all.csv': 'Таблица_3_10_Сводная_точность_по_моделям.csv',
            'economics_all.csv': 'Таблица_3_4_Экономические_метрики.csv',
            'dm_test_pairs_all.csv': 'Таблица_3_7_DM_попарные_проверки.csv',
            'thresholds.csv': 'Таблица_3_6_Оптимальные_пороги.csv',
        }
        for f in os.listdir(cons):
            if f in mapping:
                shutil.copy2(os.path.join(cons, f), os.path.join(tables, mapping[f]))
    # VAR таблицы
    if os.path.isdir(var):
        var_map = {
            'metrics_per_ticker.csv': 'Таблица_3_8_VAR_метрики_по_тикерам.csv',
            'metrics_aggregate.csv': 'Таблица_3_8_VAR_агрегированные_метрики.csv',
            'dm_vs_sarimax.csv': 'Таблица_3_7_DM_VAR_vs_SARIMAX.csv',
            'dm_vs_best.csv': 'Таблица_3_7_DM_VAR_vs_Лучшая.csv',
            'order_selection.csv': 'Таблица_3_8_VAR_порядки_и_устойчивость.csv',
            'granger_causality_summary.csv': 'Таблица_3_9_Грейнджер_сводка.csv',
        }
        for f in os.listdir(var):
            if f in var_map:
                shutil.copy2(os.path.join(var, f), os.path.join(tables, var_map[f]))

    # Сформировать zip‑архив пакета
    try:
        import zipfile
        zip_path = os.path.join(paths.out_dir, 'vkr_bundle.zip')
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(out_dir):
                for fn in files:
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, out_dir)
                    zf.write(full, rel)
    except Exception:
        pass


def main():
    paths = Paths()
    bundle_dir = os.path.join(paths.out_dir, 'vkr_bundle')
    _ensure(bundle_dir)
    _plot_forecast_all_models(paths, bundle_dir)
    _copy_existing_figures(paths, bundle_dir)
    print('Пакет ВКР сформирован: outputs/vkr_bundle (figures/, tables/)')


if __name__ == '__main__':
    main()
