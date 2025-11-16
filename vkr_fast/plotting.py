import matplotlib.pyplot as plt


def apply_gost_style():
    """Set matplotlib defaults that align with ГОСТ‑style plotting."""
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5),
            "figure.dpi": 150,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


__all__ = ["apply_gost_style"]
