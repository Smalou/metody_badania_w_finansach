"""Wspolne style wykresow dla wszystkich zadan."""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

SECTOR_PALETTE = {
    "Financials":    "#1f77b4",
    "Energy":        "#d62728",
    "Communication": "#9467bd",
    "Technology":    "#2ca02c",
}


def setup_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(fig, name):
    out = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig -> {out.relative_to(FIG_DIR.parent)}")
    return out
