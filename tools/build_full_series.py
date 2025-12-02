import os
import argparse
import sys
sys.path.append(os.getcwd())
from vkr_fast.mpl_config import configure_matplotlib

configure_matplotlib()

import pandas as pd
import matplotlib.pyplot as plt
from vkr_fast.config import Paths
from vkr_fast.plotting import apply_gost_style


ALLOWED_MODELS = {"LSTM_att", "CatBoost", "Hybrid", "SARIMAX"}


def stitch_full_series(out_dir: str, target: str):
    paths = Paths(out_dir=out_dir)
    full_dir = os.path.join(out_dir, "full_preds")
    pred_dir = os.path.join(out_dir, "preds")
    if not os.path.isdir(full_dir):
        print(f"full_preds не найдено в {out_dir}")
        return
    plot_dir = os.path.join(out_dir, "reports_full")
    os.makedirs(plot_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(full_dir) if f.endswith(".csv") and "_f" in f)
    for pf in files:
        tk = pf.split("_f")[0]
        fold = pf.split("_f")[1].split(".")[0]
        df_full = pd.read_csv(os.path.join(full_dir, pf))
        df_full["Datetime"] = pd.to_datetime(df_full["Datetime"], utc=True)
        df_full.set_index("Datetime", inplace=True)
        model_cols = [c for c in df_full.columns if c in ALLOWED_MODELS]

        # подтягиваем прогнозы из preds (хотя бы на отрезке прогноза) для заполнения NaN
        fc_start = None
        fc_end = None
        pred_file = os.path.join(pred_dir, f"{tk}_f{fold}.csv")
        preds_df = None
        if os.path.exists(pred_file):
            preds_df = pd.read_csv(pred_file)
            if "Datetime" in preds_df.columns:
                preds_df["Datetime"] = pd.to_datetime(preds_df["Datetime"], utc=True)
                preds_df.set_index("Datetime", inplace=True)
                if not preds_df.empty:
                    fc_start = preds_df.index.min()
                    fc_end = preds_df.index.max()
                for col in [c for c in preds_df.columns if c in ALLOWED_MODELS]:
                    if col not in df_full.columns:
                        df_full[col] = pd.NA
                    df_full[col] = df_full[col].fillna(preds_df[col])

        # сохраняем обновлённые full_preds (с подставленными прогнозами из preds)
        df_full.reset_index(inplace=True)
        df_full.to_csv(os.path.join(full_dir, pf), index=False)
        model_cols = [c for c in df_full.columns if c in ALLOWED_MODELS]

        apply_gost_style()
        palette = plt.cm.tab10.colors if len(model_cols) <= 10 else plt.cm.tab20.colors

        # Комбинированный график
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_full["Datetime"], df_full["y_true"], label="Факт", color="black", linewidth=1.5, alpha=0.9)
        for i, c in enumerate(model_cols):
            ax.plot(df_full["Datetime"], df_full[c], label=c, linewidth=1.7, alpha=0.9, color=palette[i % len(palette)])
        if fc_start is not None:
            ax.axvline(fc_start, color="#666666", linestyle="--", linewidth=1.2, label="Старт прогноза")
        ax.set_xlabel("Дата и время (UTC)")
        ax.set_ylabel("Значение")
        ax.legend(loc="upper left", fontsize=9)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{tk}_f{fold}_full_all_models.png"), bbox_inches="tight")
        plt.close(fig)

        # Отдельные графики «факт vs модель»
        for i, c in enumerate(model_cols):
            fig, ax = plt.subplots(figsize=(11, 3.5))
            ax.plot(df_full["Datetime"], df_full["y_true"], label="Факт", color="black", linewidth=1.4, alpha=0.9)
            ax.plot(df_full["Datetime"], df_full[c], label=f"Прогноз {c}", linewidth=1.6, alpha=0.9, color=palette[i % len(palette)])
            if fc_start is not None:
                ax.axvline(fc_start, color="#666666", linestyle="--", linewidth=1.1, label="Старт прогноза")
            ax.set_xlabel("Дата и время (UTC)")
            ax.set_ylabel("Значение")
            ax.legend(loc="upper left", fontsize=9)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"{tk}_f{fold}_full_{c}.png"), bbox_inches="tight")
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--target", default="dclose", choices=["dclose", "logret"])
    args = ap.parse_args()
    stitch_full_series(args.out_dir, args.target.lower())


if __name__ == "__main__":
    main()
