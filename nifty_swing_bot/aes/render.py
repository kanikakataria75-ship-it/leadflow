"""Chart rendering for visual validation of box detection.

This exists for one purpose: to let a human look at what the detector drew and
say whether it matches what they actually look at. That check is worth more
than any summary statistic, because a detector can produce a perfectly
respectable distribution of "boxes" that a trader would never call boxes.

Bars after the detection date are drawn deliberately, in a greyed panel to the
right of a marked cut line. Nothing in the detector has seen them -- the
look-ahead audit in ``test_aes_boxes.py`` proves that structurally -- but a
reviewer needs to see how the box actually resolved to judge whether the box
was drawn in the right place.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .boxes import NestedBox, swing_low_indices  # noqa: E402
from .params import BoxParams  # noqa: E402

logger = logging.getLogger(__name__)

UP = "#1a8f5a"
DOWN = "#c8382f"
BIG = "#2b6cb0"
SMALL = "#dd6b20"
FUTURE = "#f0f0f0"


def _candles(ax: plt.Axes, df: pd.DataFrame, x: np.ndarray) -> None:
    """Draw OHLC candles at integer x positions, so gaps do not distort width."""
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    low = df["low"].to_numpy()
    c = df["close"].to_numpy()
    for k in range(len(df)):
        colour = UP if c[k] >= o[k] else DOWN
        ax.vlines(x[k], low[k], h[k], color=colour, linewidth=0.7, zorder=2)
        height = abs(c[k] - o[k]) or (h[k] - low[k]) * 0.02 or 0.01
        ax.add_patch(
            Rectangle(
                (x[k] - 0.32, min(o[k], c[k])),
                0.64,
                height,
                facecolor=colour,
                edgecolor=colour,
                linewidth=0.5,
                zorder=3,
            )
        )


def render_box(
    df: pd.DataFrame,
    nested: NestedBox,
    out_path: str | Path,
    *,
    params: BoxParams | None = None,
    context_bars: int = 60,
    forward_bars: int = 30,
    note: str = "",
) -> Path:
    """Render one detection to a PNG.

    Args:
        df: Full OHLCV frame for the symbol.
        nested: The detection to draw.
        out_path: PNG destination.
        params: Used only to mark swing lows the same way the detector counted
            them.
        context_bars: Bars of history to show before the box starts.
        forward_bars: Bars to show after the detection date, greyed.
        note: Optional caption, e.g. why this example was chosen.

    Returns:
        The path written.
    """
    params = params or BoxParams()
    big, small = nested.big, nested.small
    i = nested.as_of_idx

    lo = max(0, big.start_idx - context_bars)
    hi = min(len(df) - 1, i + forward_bars)
    view = df.iloc[lo : hi + 1]
    x = np.arange(len(view))
    cut = i - lo                       # x position of the detection bar

    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(15.5, 9.0), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.06},
    )

    # --- the region the detector could not see ---
    if hi > i:
        ax.axvspan(cut + 0.5, x[-1] + 0.5, color=FUTURE, zorder=0)
        axv.axvspan(cut + 0.5, x[-1] + 0.5, color=FUTURE, zorder=0)
        ax.text(
            cut + 1.0, ax.get_ylim()[1], "  not visible to detector ",
            va="top", ha="left", fontsize=8, style="italic", color="#777",
            transform=ax.get_xaxis_transform(),
        )

    _candles(ax, view, x)

    bx0 = big.start_idx - lo
    ax.add_patch(
        Rectangle(
            (bx0 - 0.5, big.bottom), (cut - bx0) + 1.0, big.height,
            facecolor=BIG, alpha=0.09, edgecolor=BIG, linewidth=1.8, zorder=1,
        )
    )
    ax.hlines(big.mid, bx0 - 0.5, cut + 0.5, color=BIG, linestyle=":", linewidth=1.2, zorder=4)
    ax.hlines(big.top, bx0 - 0.5, x[-1] + 0.5, color=BIG, linestyle="--", linewidth=1.0, alpha=0.7, zorder=4)
    ax.hlines(big.bottom, bx0 - 0.5, x[-1] + 0.5, color=BIG, linestyle="--", linewidth=1.0, alpha=0.7, zorder=4)

    if small is not None:
        sx0 = small.start_idx - lo
        ax.add_patch(
            Rectangle(
                (sx0 - 0.5, small.bottom), (cut - sx0) + 1.0, small.height,
                facecolor=SMALL, alpha=0.16, edgecolor=SMALL, linewidth=1.6, zorder=5,
            )
        )

    # --- swing lows, marked exactly as the detector counted them ---
    box_low = df["low"].to_numpy()[big.start_idx : i + 1]
    for j in swing_low_indices(box_low, params.swing_k):
        ax.plot(bx0 + j, box_low[j], marker="^", markersize=7,
                color="#2f855a", zorder=6, markeredgecolor="white", markeredgewidth=0.6)

    sma10 = df["close"].rolling(10).mean().iloc[lo : hi + 1]
    ax.plot(x, sma10.to_numpy(), color="#805ad5", linewidth=1.3, label="SMA(10)", zorder=4)
    ax.axvline(cut + 0.5, color="#222", linestyle="-", linewidth=1.4, zorder=7)

    # --- volume ---
    vol = view["volume"].to_numpy()
    colours = [UP if c >= o else DOWN for o, c in zip(view["open"], view["close"])]
    axv.bar(x, vol, color=colours, width=0.64, alpha=0.65, zorder=2)
    vma = df["volume"].rolling(params.vol_ma_bars).mean().iloc[lo : hi + 1]
    axv.plot(x, vma.to_numpy(), color="#2d3748", linewidth=1.2,
             label=f"SMA(vol,{params.vol_ma_bars})", zorder=3)
    axv.axvline(cut + 0.5, color="#222", linewidth=1.4, zorder=7)
    axv.set_ylabel("volume", fontsize=9)
    axv.legend(fontsize=8, loc="upper left")

    # --- labels ---
    ticks = np.arange(0, len(view), max(1, len(view) // 14))
    axv.set_xticks(ticks)
    axv.set_xticklabels([view.index[t].strftime("%d-%b-%y") for t in ticks],
                        rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("price", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.15, zorder=0)
    axv.grid(alpha=0.15, zorder=0)

    title = (
        f"{nested.symbol}   detected {nested.as_of:%d-%b-%Y}   "
        f"big box {big.bars} bars / {big.range_pct:.1%}   "
        f"small box {'none' if small is None else f'{small.bars} bars / {small.range_pct:.1%}'}   "
        f"zone: {nested.small_zone}"
    )
    ax.set_title(title + (f"\n{note}" if note else ""), fontsize=11, loc="left")

    # Outside the axes, not floating over the candles -- the numbers are there
    # to be checked against the drawing, which means not hiding it.
    ax.text(
        1.015, 1.0, _metrics_text(nested), transform=ax.transAxes,
        fontsize=8.4, family="monospace", va="top", ha="left",
        bbox={"facecolor": "#fbfbfb", "edgecolor": "#bbb", "pad": 7},
        zorder=8,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=115, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _metrics_text(nested: NestedBox) -> str:
    """The numbers behind the drawing, so the picture can be checked."""
    b, s = nested.big, nested.small

    def opt(value: float | None, fmt: str) -> str:
        return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else format(value, fmt)

    lines = [
        "BIG BOX (spec 2.1/2.4)",
        f"  top / bottom      {b.top:,.1f} / {b.bottom:,.1f}",
        f"  duration          {b.bars} bars",
        f"  closes above mid  {b.pct_closes_above_mid:.0%}   (want >50%, target 60%+)",
        f"  higher lows       {b.higher_lows} of {max(0, b.swing_lows - 1)} transitions",
        f"  lows tilt         {b.lows_tilt:+.2f}",
        f"  top/bot touches   {b.top_touches} / {b.bottom_touches}",
        f"  quality           {b.quality:.3f}",
        "",
        "SMALL BOX (spec 2.1)",
    ]
    if s is None:
        lines.append("  none in 5-10% band")
    else:
        lines += [
            f"  top / bottom      {s.top:,.1f} / {s.bottom:,.1f}",
            f"  duration          {s.bars} bars",
            f"  position in big   {opt(nested.small_position, '.2f')}  -> {nested.small_zone}",
            f"  close containment {s.close_containment:.0%}",
        ]
    lines += [
        "",
        "CONTEXT (spec 2.2/2.3/2.5)",
        f"  expansion leg     {nested.prior_leg_pct:+.1%}",
        f"  volume dry-up     {opt(nested.vol_dryup_ratio, '.2f')}x vs pre-box",
        f"  own duration norm {opt(nested.own_norm_bars, '.0f')} bars"
        f"   -> {opt(nested.duration_vs_own_norm, '.2f')}x",
        f"  prior boxes       {nested.prior_boxes}  (resolved up: {nested.prior_cycles})",
    ]
    return "\n".join(lines)


__all__ = ["render_box"]
