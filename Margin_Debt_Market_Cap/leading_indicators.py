"""
Leading indicators of margin deleveraging.

Target
------
Entry into a deleveraging episode: FINRA margin debt falls >= 10% below its trailing 12-month
high at some point in the next h months (h = 3, 6). Only months not already in an episode are
evaluated. Base rate: ~15% (3 months), ~27% (6 months).

Candidates (monthly; daily/weekly series sampled at month end)
-------------------------------------------------------------
Financial conditions : NFCI level and 3m change (Chicago Fed), STLFSI4 (St. Louis Fed)
Volatility           : VIX, VIX / VIX3M (term structure)
Credit               : Baa - 10y Treasury spread, level and 6m change
Price trend          : S&P 500 vs 200-day MA, 1m return, drawdown from 52-week high
Rates                : Fed funds 12m change, 2y yield 6m change, 10y-2y curve
Margin-specific      : margin debt YoY growth and its 3m change, cash cushion (free credits / debit)
Tech                 : XLK 3m return minus S&P 500 3m return

Every candidate becomes a real-time percentile (expanding window over its own history,
>= 36 months, oriented so that higher = more risk), so thresholds use no future data.
Scores: AUC (0.5 = coin flip), hit rate when the percentile is >= 80%, lead time at each onset.

Composite (fixed in advance, one per family): NFCI level, VIX, Baa spread 6m change,
S&P 500 vs 200-day MA - equal-weight mean of the four percentiles.

Usage
-----
python leading_indicators.py [--spx-now 6946]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from deleveraging_sensitivity import DD_THRESHOLD, load_yahoo, month_end
from margin_debt_market_cap import load_finra, load_fred

HORIZONS = (3, 6)
MIN_HISTORY = 36
SIGNAL_PCT = 0.80
COMPOSITE = ["nfci", "vix", "baa_chg6", "spx_ma200"]
COMPOSITE_THRESHOLD = 0.70

INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, VIOLET, BAND = "#2a78d6", "#eb6834", "#6250d6", "#f0efec"

# key: (label, family, direction: +1 = higher is riskier)
SPEC = {
    "nfci": ("NFCI level", "Financial conditions", 1),
    "nfci_chg3": ("NFCI 3m change", "Financial conditions", 1),
    "stlfsi": ("St. Louis Fed stress index", "Financial conditions", 1),
    "vix": ("VIX", "Volatility", 1),
    "vix_ts": ("VIX / VIX3M", "Volatility", 1),
    "baa": ("Baa-10y spread", "Credit", 1),
    "baa_chg6": ("Baa-10y spread 6m change", "Credit", 1),
    "spx_ma200": ("S&P 500 vs 200d MA", "Price trend", -1),
    "spx_ret1": ("S&P 500 1m return", "Price trend", -1),
    "spx_dd": ("S&P 500 vs 52w high", "Price trend", -1),
    "ff_chg12": ("Fed funds 12m change", "Rates", 1),
    "dgs2_chg6": ("2y yield 6m change", "Rates", 1),
    "curve": ("10y-2y curve", "Rates", -1),
    "md_yoy": ("Margin debt YoY", "Margin-specific", 1),
    "md_yoy_chg3": ("Margin debt YoY, 3m change", "Margin-specific", -1),
    "cushion": ("Cash cushion (credits / debit)", "Margin-specific", -1),
    "xlk_rel3": ("XLK vs S&P 500, 3m", "Tech", 1),
}


# --------------------------------------------------------------------------- data
def build_indicators(spx_now: float | None = None):
    finra = load_finra()
    md = finra["margin_debt"].astype(float)
    credit = finra["free_credit_total"].astype(float)

    spx_d, spx_quote = load_yahoo("^GSPC", start="1985-01-01")
    xlk_d, _ = load_yahoo("XLK", start="1998-12-01")
    if spx_now:
        spx_d.loc[pd.Timestamp.today().normalize()] = spx_now
        spx_d = spx_d.sort_index()

    f = {k: load_fred(k) for k in ("VIXCLS", "VXVCLS", "BAA10Y", "NFCI", "STLFSI4", "DGS2", "T10Y2Y")}
    ff = load_fred("FEDFUNDS")
    ff.index = ff.index.to_period("M")

    spx = month_end(spx_d)
    xlk = month_end(xlk_d)
    x = {
        "nfci": month_end(f["NFCI"]),
        "stlfsi": month_end(f["STLFSI4"]),
        "vix": month_end(f["VIXCLS"]),
        "vix_ts": month_end(f["VIXCLS"]) / month_end(f["VXVCLS"]),
        "baa": month_end(f["BAA10Y"]),
        "spx_ma200": spx / month_end(spx_d.rolling(200).mean()) - 1,
        "spx_ret1": np.log(spx).diff(),
        "spx_dd": spx / month_end(spx_d.rolling(252, min_periods=20).max()) - 1,
        "ff_chg12": ff.diff(12),
        "dgs2_chg6": month_end(f["DGS2"]).diff(6),
        "curve": month_end(f["T10Y2Y"]),
        "md_yoy": md.pct_change(12, fill_method=None),
        "cushion": credit / md,
        "xlk_rel3": np.log(xlk).diff(3) - np.log(spx).diff(3),
    }
    x["nfci_chg3"] = x["nfci"].diff(3)
    x["baa_chg6"] = x["baa"].diff(6)
    x["md_yoy_chg3"] = x["md_yoy"].diff(3)
    X = pd.DataFrame(x)[list(SPEC)]
    return X, md


def realtime_percentile(s: pd.Series, min_history: int = MIN_HISTORY) -> pd.Series:
    s = s.dropna()
    v = s.values
    out = np.full(len(v), np.nan)
    for i in range(min_history, len(v)):
        out[i] = (v[: i + 1] <= v[i]).mean()
    return pd.Series(out, index=s.index)


# ---------------------------------------------------------------------- analysis
def deleveraging_target(md: pd.DataFrame):
    dd = md / md.rolling(12, min_periods=1).max() - 1
    in_ep = dd <= DD_THRESHOLD
    onsets = dd.index[in_ep & ~in_ep.shift(1, fill_value=False) & ~in_ep.shift(2, fill_value=False)
                      & ~in_ep.shift(3, fill_value=False)]

    def hit(h):
        fwd = pd.concat([dd.shift(-k) for k in range(1, h + 1)], axis=1).min(axis=1)
        return (fwd <= DD_THRESHOLD).astype(float).where(dd.shift(-h).notna())

    return dd, ~in_ep, {h: hit(h) for h in HORIZONS}, onsets


def auc(score: pd.Series, y: pd.Series) -> float:
    d = pd.DataFrame({"s": score, "y": y}).dropna()
    n1 = d["y"].sum()
    n0 = len(d) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = d["s"].rank()
    return (r[d["y"] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def lead_months(p: pd.Series, onset: pd.Period, th: float) -> int:
    """Consecutive months before onset with the percentile at or above the threshold."""
    k = 0
    while (onset - k - 1) in p.index and p.get(onset - k - 1, 0) >= th:
        k += 1
    return k


def score_all(P: pd.DataFrame, eligible: pd.Series, hits: dict, onsets) -> pd.DataFrame:
    rows = {}
    for c in P:
        p = P[c].where(eligible)
        row = {f"auc_{h}m": auc(p, hits[h]) for h in HORIZONS}
        sig = p >= SIGNAL_PCT
        row["months_signal_pct"] = sig[p.notna()].mean()
        row["hit_3m_if_signal"] = hits[3][sig].mean()
        row["hit_6m_if_signal"] = hits[6][sig].mean()
        th = COMPOSITE_THRESHOLD if c == "composite" else SIGNAL_PCT
        leads = [lead_months(P[c], o, th) for o in onsets if pd.notna(P[c].get(o - 1))]
        row["onsets_signaled_ahead"] = f"{sum(l > 0 for l in leads)}/{len(leads)}"
        row["median_lead_m"] = float(np.median(leads)) if leads else np.nan
        rows[c] = row
    out = pd.DataFrame(rows).T
    label = {k: v[0] for k, v in SPEC.items()} | {"composite": "Composite (4 families)"}
    family = {k: v[1] for k, v in SPEC.items()} | {"composite": "Composite"}
    out.insert(0, "family", out.index.map(family))
    out.insert(0, "label", out.index.map(label))
    return out.sort_values("auc_3m", ascending=False)


def event_study(p: pd.Series, onsets, window=range(-12, 4)) -> pd.DataFrame:
    rows = {}
    for o in onsets:
        rows[str(o)] = {k: p.get(o + k, np.nan) for k in window}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- plot
def _style(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": BASE,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })


def _title(ax, text, grid_axis="y"):
    ax.set_title(text, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, lw=1)


def plot_auc(scores: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    s = scores.iloc[::-1]
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(10, 7.5))
    h = 0.36
    ax.barh(y + h / 2, s["auc_3m"].astype(float) - 0.5, left=0.5, height=h * 0.9, color=BLUE,
            label="Deleveraging starts within 3 months")
    ax.barh(y - h / 2, s["auc_6m"].astype(float) - 0.5, left=0.5, height=h * 0.9, color=ORANGE,
            label="… within 6 months")
    for yy, v in zip(y, s["auc_3m"].astype(float)):
        ax.text(max(v, 0.5) + 0.005, yy + h / 2, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.axvline(0.5, color=INK2, lw=1)
    ax.text(0.5, len(s) - 0.4, " 0.5 = coin flip", fontsize=8, color=INK2, va="bottom")
    ax.set_yticks(y, [f"{lab}  ·  {fam}" for lab, fam in zip(s["label"], s["family"])])
    ax.set_xlim(0.35, 0.85)
    ax.set_xlabel("AUC (real-time percentiles, 1997–2026)")
    ax.legend(loc="lower right", frameon=False, labelcolor=INK2)
    _title(ax, "Which indicators lead margin deleveraging?", grid_axis="x")
    fig.text(0.01, 0.005, "Target: FINRA margin debt >= 10% below its 12-month high within the horizon. "
             "Months already in an episode excluded.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_composite(P, dd, onsets, ev, now_pcts, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    _style(plt)
    ts = lambda idx: idx.to_timestamp(how="end")  # noqa: E731

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1.1, 1])
    ax = fig.add_subplot(gs[0, :])
    in_ep = dd <= DD_THRESHOLD
    for m in dd.index[in_ep]:
        ax.axvspan(m.to_timestamp(how="start"), m.to_timestamp(how="end"), color=BAND, lw=0, zorder=0)
    comp = P["composite"].loc[dd.index[0]:].dropna()
    ax.plot(ts(comp.index), comp.values, color=VIOLET, lw=1.6)
    ax.axhline(COMPOSITE_THRESHOLD, color=INK2, lw=1)
    ax.text(ts(comp.index)[0], COMPOSITE_THRESHOLD, f" signal ≥ {COMPOSITE_THRESHOLD:.0%}", fontsize=8,
            color=INK2, va="bottom")
    ax.plot(ts(comp.index[-1:]), comp.values[-1:], "o", ms=8, color=VIOLET, mec=SURFACE, mew=2)
    ax.annotate(f"now {comp.iloc[-1]:.0%}", (ts(comp.index)[-1], comp.iloc[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _title(ax, "Composite risk percentile (NFCI, VIX, Baa spread Δ6m, S&P vs 200d MA)  ·  "
               "shaded = margin debt ≥ 10% below 12m high")

    ax = fig.add_subplot(gs[1, 0])
    k = np.array(ev.columns, dtype=int)
    med, q1, q3 = ev.median(), ev.quantile(0.25), ev.quantile(0.75)
    ax.fill_between(k, q1.values, q3.values, color=VIOLET, alpha=0.10, lw=0)
    ax.plot(k, med.values, color=VIOLET, lw=2)
    ax.axvline(0, color=INK2, lw=1)
    ax.axhline(0.5, color=BASE, lw=1)
    ax.text(0, 0.02, " onset", fontsize=8, color=INK2)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_xlabel("Months relative to deleveraging onset")
    _title(ax, f"Composite around the {len(ev)} onsets (median, IQR)")

    ax = fig.add_subplot(gs[1, 1])
    names = list(now_pcts.index)
    yy = np.arange(len(names))[::-1]
    ax.barh(yy, now_pcts.values, height=0.5, color=[VIOLET if n == "Composite" else BLUE for n in names])
    for y_, v in zip(yy, now_pcts.values):
        ax.text(v + 0.01, y_, f"{v:.0%}", va="center", fontsize=9, color=INK)
    ax.axvline(COMPOSITE_THRESHOLD, color=INK2, lw=1)
    ax.set_yticks(yy, names)
    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    _title(ax, "Today: risk percentile vs. own history", grid_axis="x")

    fig.text(0.01, 0.005, "Sources: FINRA; FRED (NFCI, VIXCLS, BAA10Y); Yahoo Finance (^GSPC). "
             "Percentiles are real-time (expanding window).", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--spx-now", type=float, help="current S&P 500 level (default: latest Yahoo close)")
    args = ap.parse_args()

    X, md = build_indicators(args.spx_now)
    P = pd.DataFrame({c: realtime_percentile(X[c] * SPEC[c][2]) for c in X})
    P["composite"] = P[COMPOSITE].mean(axis=1, skipna=False)

    dd, eligible, hits, onsets = deleveraging_target(md)
    P = P.reindex(P.index.union(dd.index))
    scores = score_all(P.loc[dd.index], eligible, hits, onsets)
    ev = event_study(P["composite"], onsets)

    latest = P.index[-1]
    labels = {c: SPEC[c][0] for c in COMPOSITE} | {"composite": "Composite"}
    now_pcts = P.loc[latest, COMPOSITE + ["composite"]].rename(labels)

    (args.out_dir / "data").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "images").mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out_dir / "data" / "leading_indicators_scores.csv", float_format="%.4f")
    pd.concat({"value": X, "risk_percentile": P}, axis=1).to_csv(
        args.out_dir / "data" / "leading_indicators_monthly.csv", float_format="%.4f")
    plot_auc(scores, args.out_dir / "images" / "leading_indicators_auc.png")
    plot_composite(P, dd, onsets, ev, now_pcts, args.out_dir / "images" / "leading_indicators_composite.png")

    pd.set_option("display.width", 200)
    base = {h: hits[h][eligible].mean() for h in HORIZONS}
    print(f"Base rate: {base[3]:.0%} (3m), {base[6]:.0%} (6m). Onsets: {[str(o) for o in onsets]}")
    num = [c for c in scores if c not in ("label", "family", "onsets_signaled_ahead")]
    print(scores.astype({c: float for c in num}).round(2).to_string())
    comp = P["composite"]
    sig = (comp >= COMPOSITE_THRESHOLD) & eligible.reindex(comp.index, fill_value=False)
    print(f"\nComposite >= {COMPOSITE_THRESHOLD:.0%}: {sig[comp.notna()].mean():.0%} of months; "
          f"P(onset within 3m) {hits[3].reindex(comp.index)[sig].mean():.0%}, "
          f"within 6m {hits[6].reindex(comp.index)[sig].mean():.0%}")
    print("\nComposite around onsets (median by month):", ev.median().round(2).to_dict())
    print(f"\nLatest ({latest}) raw values and risk percentiles:")
    print(pd.DataFrame({"value": X.loc[latest] if latest in X.index else np.nan,
                        "risk_pct": P.loc[latest, list(SPEC)]}).round(3).to_string())
    print(f"composite: {P.loc[latest, 'composite']:.2f}")


if __name__ == "__main__":
    main()
