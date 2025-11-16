import os
import sys
import re

sys.path.append(os.getcwd())
from vkr_fast.config import Paths


def _scan_figures(fig_dir: str):
    items = []
    if not os.path.isdir(fig_dir):
        return items
    for f in sorted(os.listdir(fig_dir)):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        if f.startswith('Рисунок_'):
            # Рисунок_3_10_...
            m = re.match(r'Рисунок_(\d+)_(\d+)_(.*)\.(?:png|jpg|jpeg)$', f)
            if m:
                ch, num, rest = m.groups()
                title = rest.replace('_', ' ')
                items.append((int(ch), int(num), f"Рисунок {ch}.{num} — {title}", f))
                continue
        # fallback: распознать g2_*, g3_* и rmse_bar_*
        if '_g2_' in f:
            parts = f.split('_g2_')[1].split('_')
            code = parts[0]
            name = f.split('_g2_')[0]
            tk_ru = name  # уже рус/лат; оставим как есть
            mapping = {
                '1': 'Динамика ряда цены закрытия',
                '2': 'Динамика лог-доходностей (ΔЦена)',
                '3': 'Разложение ряда (тренд/сезонность/остаток)',
                '4': 'Скользящие статистики доходностей',
                '5': 'Эмпирическое распределение / Q–Q',
                '6': 'ACF/PACF доходностей',
                '7': 'ARCH‑эффект: коррелограмма квадратов',
                '9': 'Сравнение режимов волатильности по подвыборкам',
            }
            items.append((2, int(code), f"Рисунок 2.{code} — {mapping.get(code, 'График главы 2')} ({tk_ru})", f))
            continue
        if f.startswith('rmse_bar_'):
            tk = f[len('rmse_bar_'):-4]
            items.append((3, 5, f"Рисунок 3.5 — Сравнение моделей по RMSE (среднее по фолдам) ({tk})", f))
    return sorted(items, key=lambda x: (x[0], x[1], x[2]))


def _scan_tables(tbl_dir: str):
    items = []
    if not os.path.isdir(tbl_dir):
        return items
    for f in sorted(os.listdir(tbl_dir)):
        if not f.lower().endswith(('.csv', '.xlsx', '.xls')):
            continue
        if f.startswith('Таблица_'):
            m = re.match(r'Таблица_(\d+)_(\d+)_(.*)\.(?:csv|xlsx|xls)$', f)
            if m:
                ch, num, rest = m.groups()
                title = rest.replace('_', ' ')
                items.append((int(ch), int(num), f"Таблица {ch}.{num} — {title}", f))
                continue
        # fallback — прочие таблицы
        items.append((3, 99, f"Таблица — {f}", f))
    return sorted(items, key=lambda x: (x[0], x[1], x[2]))


def main():
    paths = Paths()
    base = os.path.join(paths.out_dir, 'vkr_bundle')
    fig_dir = os.path.join(base, 'figures')
    tbl_dir = os.path.join(base, 'tables')
    fig_items = _scan_figures(fig_dir)
    tbl_items = _scan_tables(tbl_dir)
    out = []
    out.append('# Список рисунков и таблиц\n\n')
    out.append('## Список рисунков\n\n')
    for _, _, title, fname in fig_items:
        out.append(f'- {title} (файл: figures/{fname})\n')
    out.append('\n## Список таблиц\n\n')
    for _, _, title, fname in tbl_items:
        out.append(f'- {title} (файл: tables/{fname})\n')
    out_path = os.path.join(base, 'Список_рисунков_и_таблиц.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(''.join(out))
    # Добавим в конец REPORT.md
    try:
        with open('REPORT.md', 'a', encoding='utf-8') as f:
            f.write('\n\n')
            f.write(''.join(out))
    except Exception:
        pass
    print('Список рисунков и таблиц сформирован:', out_path)


if __name__ == '__main__':
    main()

