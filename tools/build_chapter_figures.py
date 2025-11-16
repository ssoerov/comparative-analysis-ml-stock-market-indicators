import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
from vkr_fast.mpl_config import configure_matplotlib

configure_matplotlib()

from vkr_fast.config import Paths, TimeParams, TICKERS
from vkr_fast.data import fetch_moex, yahoo_csv, keyrate_series

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch
from scipy import stats as sps


RU_TICKER = {
    'IMOEX': 'Индекс МосБиржи',
}


def _style_gost():
    plt.rcParams.update({
        'figure.figsize': (10, 4),
        'figure.dpi': 150,
        'font.size': 12,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
    })


def _ensure(path: str):
    os.makedirs(path, exist_ok=True)


def _load_raw(paths: Paths, timep: TimeParams, tickers: Dict[str, tuple]) -> Dict[str, pd.DataFrame]:
    brent = yahoo_csv(paths.data_dir, "Brent_yahoo_1h_2023-01-01_2025-04-01 (1).csv")
    usdr = yahoo_csv(paths.data_dir, "USD_RUB_yahoo_1h_2023-01-01_2025-04-01.csv")
    krate = keyrate_series()
    out = {}
    sl = slice(timep.period_start, timep.period_end - pd.Timedelta(seconds=1))
    for tk in tickers:
        df = fetch_moex(paths.cache_dir, tk, *tickers[tk], timep.start_raw, timep.end_raw, timep.interval_minutes, use_cache_only=True)
        idx = df.index
        ext = pd.DataFrame(index=idx)
        ext['Brent'] = brent['Close'].reindex(idx).ffill()
        ext['USD'] = usdr['Close'].reindex(idx).ffill()
        ext['KeyRate'] = krate['KeyRate'].reindex(idx).ffill()
        ext.bfill(limit=1, inplace=True)
        out[tk] = df.join(ext, how='left').loc[sl]
    return out


def _price_fig(tk: str, df: pd.DataFrame, out_dir: str):
    name = RU_TICKER.get(tk, tk)
    dt = df.index
    p_start, p_end = dt.min(), dt.max()
    fig, ax = plt.subplots()
    ax.plot(dt, df['Close'], label='Цена закрытия')
    ax.set_title(f"{name}: Динамика ряда цены закрытия за период {p_start:%Y-%m-%d} — {p_end:%Y-%m-%d}")
    ax.set_xlabel('Дата и время (UTC)')
    ax.set_ylabel('Цена закрытия')
    ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tk}_g2_1_close.png"))
    plt.close(fig)


def _returns(df: pd.DataFrame) -> pd.Series:
    return np.log(df['Close'].astype(float)).diff().dropna()


def _returns_fig(tk: str, df: pd.DataFrame, out_dir: str):
    name = RU_TICKER.get(tk, tk)
    r = _returns(df)
    fig, ax = plt.subplots()
    ax.plot(r.index, r.values, label='Лог-доходность (Δ)')
    ax.set_title(f"{name}: Динамика лог-доходностей (ΔЦена)")
    ax.set_xlabel('Дата и время (UTC)')
    ax.set_ylabel('Изменение цены (ΔЦена)')
    ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tk}_g2_2_dret.png"))
    plt.close(fig)


def _stl_fig(tk: str, df: pd.DataFrame, out_dir: str):
    name = RU_TICKER.get(tk, tk)
    s = df['Close'].astype(float).asfreq('H')
    s = s.interpolate(limit_direction='both')
    stl = STL(s, period=24).fit()
    fig = stl.plot()
    fig.suptitle(f"{name}: Разложение ряда (тренд/сезонность/остаток)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tk}_g2_3_stl.png"))
    plt.close(fig)


def _rolling_stats_fig(tk: str, df: pd.DataFrame, out_dir: str, win: int = 50):
    name = RU_TICKER.get(tk, tk)
    r = _returns(df)
    m = r.rolling(win).mean()
    s = r.rolling(win).std()
    fig, ax = plt.subplots()
    ax.plot(m.index, m.values, label=f'Скользящее среднее (окно {win})')
    ax.plot(s.index, s.values, label=f'Скользящее σ (окно {win})')
    ax.set_title(f"{name}: Скользящие статистики доходностей")
    ax.set_xlabel('Дата и время (UTC)')
    ax.set_ylabel('Значение')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tk}_g2_4_roll_stats.png"))
    plt.close(fig)


def _hist_and_qq_figs(tk: str, df: pd.DataFrame, out_dir: str):
    name = RU_TICKER.get(tk, tk)
    r = _returns(df)
    # Histogram
    fig, ax = plt.subplots()
    ax.hist(r.values, bins=40, color='#4C72B0', edgecolor='black', alpha=0.8)
    ax.set_title(f"{name}: Эмпирическое распределение лог-доходностей")
    ax.set_xlabel('Лог-доходность')
    ax.set_ylabel('Частота')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tk}_g2_5_hist.png"))
    plt.close(fig)
    # QQ plot
    fig, ax = plt.subplots()
    sps.probplot(r.values, dist='norm', plot=ax)
    ax.set_title(f"{name}: Q–Q график лог-доходностей")
    ax.set_xlabel('Теоретические квантили')
    ax.set_ylabel('Наблюдаемые квантили')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tk}_g2_5_qq.png"))
    plt.close(fig)


def _acf_pacf_figs(tk: str, df: pd.DataFrame, out_dir: str):
    name = RU_TICKER.get(tk, tk)
    r = _returns(df)
    # ACF
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(r.values, lags=40, ax=ax)
    ax.set_title(f"{name}: Автокорреляционная функция доходностей")
    ax.set_xlabel('Лаг')
    ax.set_ylabel('Корреляция')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tk}_g2_6_acf.png"))
    plt.close(fig)
    # PACF
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_pacf(r.values, lags=40, ax=ax, method='ywm')
    ax.set_title(f"{name}: Частная автокорреляционная функция доходностей")
    ax.set_xlabel('Лаг')
    ax.set_ylabel('Частная корреляция')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tk}_g2_6_pacf.png"))
    plt.close(fig)


def _arch_test_fig(tk: str, df: pd.DataFrame, out_dir: str):
    name = RU_TICKER.get(tk, tk)
    r = _returns(df)
    r2 = r**2
    # ACF of squared returns
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(r2.values, lags=40, ax=ax)
    ax.set_title(f"{name}: Коррелограмма квадратов доходностей (ARCH-эффект)")
    ax.set_xlabel('Лаг')
    ax.set_ylabel('Корреляция')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tk}_g2_7_arch_acf.png"))
    plt.close(fig)
    # LM-ARCH p-value
    try:
        stat, p, _, _ = het_arch(r.values, nlags=10)
        pd.DataFrame([[float(stat), float(p)]], columns=['LM_stat', 'p_value']).to_csv(
            os.path.join(out_dir, f"{tk}_g2_7_arch_lm.csv"), index=False
        )
    except Exception:
        pass


def _subperiod_vol_fig(tk: str, df: pd.DataFrame, out_dir: str, win: int = 50):
    name = RU_TICKER.get(tk, tk)
    r = _returns(df)
    mid = r.index[int(len(r) / 2)]
    r1, r2 = r.loc[:mid], r.loc[mid:]
    s1, s2 = r1.rolling(win).std(), r2.rolling(win).std()
    fig, ax = plt.subplots()
    ax.plot(s1.index, s1.values, label='σ (первая половина)')
    ax.plot(s2.index, s2.values, label='σ (вторая половина)')
    ax.set_title(f"{name}: Сравнение режимов волатильности по подвыборкам")
    ax.set_xlabel('Дата и время (UTC)')
    ax.set_ylabel('σ')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tk}_g2_9_subperiod_vol.png"))
    plt.close(fig)


def _rmse_bars_from_metrics(paths: Paths, out_dir: str):
    # Используем outputs/metrics.csv для сравнения моделей
    m = pd.read_csv(os.path.join(paths.out_dir, 'metrics.csv'))
    # русские имена тикеров и моделей
    RU_MODEL = {
        'RF': 'Случайный лес',
        'SARIMAX': 'SARIMAX',
        'CatBoost': 'CatBoost',
        'LSTM_base': 'LSTM (базовая)',
        'LSTM_att': 'LSTM (с вниманием)',
        'Hybrid': 'Гибридная LSTM+SARIMAX',
    }
    RU_TK = {'IMOEX': 'Индекс МосБиржи'}
    m['Tk_ru'] = m['Tk'].map(RU_TK).fillna(m['Tk'])
    m['Model_ru'] = m['Model'].map(RU_MODEL).fillna(m['Model'])
    for tk_ru, g in m.groupby('Tk_ru'):
        agg = g.groupby('Model_ru')['RMSE'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(agg.index, agg.values, color='#4C72B0')
        ax.set_title(f"{tk_ru}: Сравнение моделей по RMSE (среднее по фолдам)")
        ax.set_xlabel('Модель')
        ax.set_ylabel('RMSE')
        ax.tick_params(axis='x', rotation=30)
        fig.tight_layout()
        fname = f"rmse_bar_{tk_ru}.png".replace(' ', '_')
        fig.savefig(os.path.join(out_dir, fname))
        plt.close(fig)


def _naive_fig(tk: str, df: pd.DataFrame, out_dir: str):
    name = RU_TICKER.get(tk, tk)
    r = _returns(df)
    # наивный прогноз (0) и сезонно-наивный (лаг 24)
    naive = pd.Series(0.0, index=r.index)
    seas = r.shift(24)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(r.index, r.values, label='Факт (ΔЦена)')
    ax.plot(seas.index, seas.values, label='Сезонно-наивный (лаг 24)')
    ax.plot(naive.index, naive.values, label='Наивный (ноль)')
    ax.set_title(f"{name}: Базовые прогнозы (наивный и сезонно-наивный)")
    ax.set_xlabel('Дата и время (UTC)')
    ax.set_ylabel('Изменение цены (ΔЦена)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tk}_g3_1_naive.png"))
    plt.close(fig)


def _final_trajectory_best(paths: Paths, out_dir: str):
    # Соберём лучшую модель по RMSE и нарисуем факт vs прогноз (склейка по фолдам)
    m = pd.read_csv(os.path.join(paths.out_dir, 'metrics.csv'))
    for tk in m['Tk'].unique():
        mm = m[m['Tk'] == tk].groupby('Model')['RMSE'].mean().sort_values()
        best = mm.index[0]
        # собрать предсказания по фолдам
        preds = []
        for f in range(0, 10):
            pth = os.path.join(paths.out_dir, 'preds', f"{tk}_f{f}.csv")
            if not os.path.exists(pth):
                continue
            dfp = pd.read_csv(pth)
            dfp['Datetime'] = pd.to_datetime(dfp['Datetime'], utc=True)
            dfp = dfp[['Datetime', 'y_true', best]].dropna()
            preds.append(dfp)
        if not preds:
            continue
        dfc = pd.concat(preds).sort_values('Datetime').drop_duplicates('Datetime')
        name = RU_TICKER.get(tk, tk)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dfc['Datetime'], dfc['y_true'], label='Факт (ΔЦена)')
        ax.plot(dfc['Datetime'], dfc[best], label=f'Прогноз ({best})')
        ax.set_title(f"{name}: Итоговая траектория факта и лучшего прогноза")
        ax.set_xlabel('Дата и время (UTC)')
        ax.set_ylabel('Изменение цены (ΔЦена)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{tk}_g3_12_final.png"))
        plt.close(fig)


def main():
    _style_gost()
    paths = Paths()
    timep = TimeParams()
    out_dir = os.path.join(paths.out_dir, 'figs')
    _ensure(out_dir)
    raw = _load_raw(paths, timep, TICKERS)
    # Глава 2: базовые EDA-графики
    for tk, df in raw.items():
        _price_fig(tk, df, out_dir)
        _returns_fig(tk, df, out_dir)
        _stl_fig(tk, df, out_dir)
        _rolling_stats_fig(tk, df, out_dir)
        _hist_and_qq_figs(tk, df, out_dir)
        _acf_pacf_figs(tk, df, out_dir)
        _arch_test_fig(tk, df, out_dir)
        _subperiod_vol_fig(tk, df, out_dir)
        _naive_fig(tk, df, out_dir)
    # Глава 3: сравнения по метрикам
    _rmse_bars_from_metrics(paths, out_dir)
    _final_trajectory_best(paths, out_dir)
    print('Фигуры глав сформированы: outputs/figs')


if __name__ == '__main__':
    main()
