import os
import platform
import sys
import subprocess
from datetime import datetime

sys.path.append(os.getcwd())
from vkr_fast.config import TimeParams, CVParams


def _read_lockfile():
    if os.path.exists('requirements-lock.txt'):
        try:
            with open('requirements-lock.txt', 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            return lines[:50]  # короткая выдержка
        except Exception:
            return []
    return []


def main():
    tp = TimeParams()
    cv = CVParams()
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    lines = []
    lines.append('\n\n## Методологический протокол (для воспроизводимости)\n\n')
    lines.append(f"Дата формирования: {now}\n\n")
    lines.append("- Источники данных: MOEX (ISS API), Yahoo Finance (Brent, USD/RUB), ключевая ставка ЦБ РФ.\\n")
    lines.append(f"- Период анализа: {tp.period_start.date()} — {tp.period_end.date()} (частота: {tp.interval_minutes} минут).\\n")
    lines.append(f"- Схема walk‑forward: внешние фолды = {cv.outer_folds}, горизонт теста = {cv.test_horizon} баров.\\n")
    lines.append("- Граница train→test: поддерживается эмбарго (вырезка последних k баров из train) и сезонный наивный бейзлайн.\\n")
    lines.append("- Модели: SARIMAX, Случайный лес, CatBoost, LSTM (базовая/с вниманием), Гибрид LSTM+SARIMAX; GARCH‑σ как признак (опция).\\n")
    lines.append("- VAR/VARMAX: VAR(p) по Δценам индикаторов; VARMAX(p,0) с экзогенами dBrent, dUSD, KeyRate (при наличии устойчивой оценки).\\n")
    lines.append("- Метрики: MAE, RMSE, MAPE/sMAPE, WAPE, MdAPE, MASE; экономика: CumRet, MaxDD с учётом комиссии/проскальзывания.\\n")
    lines.append("- Статтесты: DM (MSE/MAE; HAC), Ljung–Box/ARCH для остатков, причинность по Грейнджеру (VAR), IRF и FEVD (при устойчивости).\\n")
    lines.append("- Порог торговли: оптимизация по holdout (валидация внутри окна предсказаний), оценка на отложенной части.\\n")
    lines.append("- Репликация: seed для NumPy/sklearn/CatBoost/TF устанавливается в конфигурации; доступен файл блокировки зависимостей.\\n\n")

    # Системная выдержка
    lines.append("Системная информация:\n\n")
    lines.append(f"- Python: {platform.python_version()} на {platform.system()} {platform.release()} ({platform.machine()})\\n")
    if os.path.exists('requirements-lock.txt'):
        lines.append("- Lockfile (фрагмент):\\n")
        for l in _read_lockfile():
            lines.append(f"  - {l}\\n")

    with open('REPORT.md', 'a', encoding='utf-8') as f:
        f.write(''.join(lines))
    # Скопируем в пакет ВКР
    out_dir = os.path.join('outputs', 'vkr_bundle')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'Методологический_протокол.md'), 'w', encoding='utf-8') as f:
        f.write(''.join(lines))
    print('Методологический протокол добавлен в REPORT.md и пакет ВКР.')


if __name__ == '__main__':
    main()
