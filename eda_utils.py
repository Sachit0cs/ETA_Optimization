from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EDA_DIR = os.path.join("outputs", "eda")
PLOT_DIR = os.path.join(EDA_DIR, "plots")
REPORT_PATH = os.path.join(EDA_DIR, "EDA_REPORT.md")


def ensure_dirs() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Section registry
# ---------------------------------------------------------------------------

# Each entry: (order, name, description, func). `order` keeps the report stable
# while allowing later-added sections (node2vec / task4 / task5) to slot in.
_SECTIONS: list[tuple[float, str, str, Callable]] = []


def section(name: str, description: str = "", order: float = 100.0):
    """Register an EDA section. Use a higher `order` to append later sections."""

    def deco(fn: Callable) -> Callable:
        _SECTIONS.append((order, name, description, fn))
        return fn

    return deco


def get_sections(only: Optional[list[str]] = None) -> list[tuple[str, str, Callable]]:
    items = sorted(_SECTIONS, key=lambda t: (t[0], t[1]))
    out = [(n, d, f) for _, n, d, f in items]
    if only:
        wanted = {s.strip() for s in only}
        out = [t for t in out if t[0] in wanted]
    return out


# ---------------------------------------------------------------------------
# Shared context passed to every section
# ---------------------------------------------------------------------------

@dataclass
class EDAContext:
    raw: pd.DataFrame                       # segment/scan-level rows
    trips: pd.DataFrame                     # one row per trip (model granularity)
    nodes: Optional[pd.DataFrame] = None    # outputs/node_metrics.csv
    corridors: Optional[pd.DataFrame] = None  # outputs/corridor_audit.csv
    embeddings: Optional[pd.DataFrame] = None  # outputs/node_emb_graphsage.csv
    plot_dir: str = PLOT_DIR

    # Collected during the run, rendered to the markdown report.
    _report: list[str] = field(default_factory=list)

    # -- report builders ----------------------------------------------------
    def heading(self, text: str, level: int = 2) -> None:
        self._report.append(f"\n{'#' * level} {text}\n")

    def add_finding(self, text: str) -> None:
        """A bullet-point insight. Also echoed to stdout for live feedback."""
        print(f"    - {text}")
        self._report.append(f"- {text}")

    def add_note(self, text: str) -> None:
        self._report.append(f"\n{text}\n")

    def add_table(self, df: pd.DataFrame, caption: str = "") -> None:
        if caption:
            self._report.append(f"\n*{caption}*\n")
        self._report.append(df_to_md(df))

    def add_plot(self, rel_path: str, caption: str = "") -> None:
        # rel_path is relative to PLOT_DIR; report lives in EDA_DIR.
        link = os.path.join("plots", os.path.basename(rel_path)).replace("\\", "/")
        self._report.append(f"\n![{caption}]({link})\n")
        if caption:
            self._report.append(f"*{caption}*\n")

    def report_markdown(self) -> str:
        return "\n".join(self._report)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Distribution summary (count, missing, central tendency, tails, skew)."""
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        nonnull = s.dropna()
        rows.append(
            {
                "variable": c,
                "n": int(s.shape[0]),
                "missing": int(s.isna().sum()),
                "missing_%": round(100 * s.isna().mean(), 2),
                "mean": _r(nonnull.mean()),
                "std": _r(nonnull.std()),
                "min": _r(nonnull.min()),
                "p25": _r(nonnull.quantile(0.25)),
                "median": _r(nonnull.median()),
                "p75": _r(nonnull.quantile(0.75)),
                "p95": _r(nonnull.quantile(0.95)),
                "p99": _r(nonnull.quantile(0.99)),
                "max": _r(nonnull.max()),
                "skew": _r(nonnull.skew()),
            }
        )
    return pd.DataFrame(rows)


def categorical_summary(series: pd.Series, top: int = 20) -> pd.DataFrame:
    vc = series.astype("object").value_counts(dropna=False).head(top)
    pct = (vc / len(series) * 100).round(2)
    return pd.DataFrame({"value": vc.index.astype(str), "count": vc.values, "pct": pct.values})


def _r(x, nd: int = 3):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return None
        return round(float(x), nd)
    except (TypeError, ValueError):
        return x


def df_to_md(df: pd.DataFrame) -> str:
    """Minimal GitHub-markdown table (no `tabulate` dependency)."""
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.values) + " |")
    return "\n".join([head, sep, *body])


# ---------------------------------------------------------------------------
# Plot helpers (each returns the saved path)
# ---------------------------------------------------------------------------

def _save(fig, name: str) -> str:
    ensure_dirs()
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"      saved plot: {path}")
    return path


def plot_distribution(series: pd.Series, name: str, title: str,
                      bins: int = 50, logx: bool = False) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if logx:
        s = s[s > 0]
        ax.hist(np.log1p(s), bins=bins, color="#4c72b0", edgecolor="white")
        ax.set_xlabel(f"log1p({series.name})")
    else:
        ax.hist(s, bins=bins, color="#4c72b0", edgecolor="white")
        ax.set_xlabel(str(series.name))
    ax.set_ylabel("count")
    ax.set_title(title)
    return _save(fig, name)


def plot_box_by_group(df: pd.DataFrame, value: str, group: str, name: str,
                      title: str, max_groups: int = 12) -> str:
    sub = df[[group, value]].copy()
    sub[value] = pd.to_numeric(sub[value], errors="coerce")
    sub = sub.dropna()
    order = sub[group].value_counts().head(max_groups).index.tolist()
    data = [sub.loc[sub[group] == g, value].values for g in order]
    fig, ax = plt.subplots(figsize=(max(7, len(order) * 0.9), 4.5))
    ax.boxplot(data, labels=[str(g) for g in order], showfliers=False)
    ax.set_xlabel(group)
    ax.set_ylabel(value)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, name)


def plot_bar(x, height, name: str, title: str, xlabel: str, ylabel: str,
             rotate: int = 0) -> str:
    fig, ax = plt.subplots(figsize=(max(7, len(x) * 0.6), 4.5))
    ax.bar([str(v) for v in x], height, color="#55a868", edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")
    return _save(fig, name)


def plot_scatter(df: pd.DataFrame, x: str, y: str, name: str, title: str,
                 sample: int = 5000, logxy: bool = False) -> str:
    sub = df[[x, y]].copy()
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    sub = sub.dropna()
    if len(sub) > sample:
        sub = sub.sample(sample, random_state=42)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(sub[x], sub[y], s=6, alpha=0.25, color="#c44e52")
    if logxy:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    return _save(fig, name)


def plot_corr_heatmap(df: pd.DataFrame, cols: list[str], name: str, title: str) -> str:
    use = [c for c in cols if c in df.columns]
    corr = df[use].apply(pd.to_numeric, errors="coerce").corr()
    fig, ax = plt.subplots(figsize=(1.1 * len(use) + 2, 1.0 * len(use) + 2))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(use)))
    ax.set_yticks(range(len(use)))
    ax.set_xticklabels(use, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(use, fontsize=8)
    for i in range(len(use)):
        for j in range(len(use)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    return _save(fig, name), corr
