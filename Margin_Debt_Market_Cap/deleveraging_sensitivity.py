"""
Sensitivity of the S&P 500 and XLK to abrupt margin deleveraging.

Margin debt  : FINRA debit balances in customers' securities margin accounts (monthly).
Prices       : Yahoo Finance chart API, month-end closes (^GSPC from 1997, XLK from 1998-12),
               split-adjusted price (no dividends), plus the latest intraday quote for "today".

Three lenses
1. Episodes  - months where margin debt is >= 10% below its trailing 12-month high are grouped
               into episodes; each runs from the 12-month high to the episode trough.
               Elasticity = index change / margin debt change over that window.
2. Monthly   - OLS of monthly index log return on monthly margin debt log change
               (all months, down months, months with margin debt <= -5%) and lead/lag correlations.
3. Scenarios - margin debt -10/-20/-30/-40% from the latest FINRA month, mapped to index moves
               with the episode elasticity (through-origin fit; range = interquartile range of the
               episode ratios), compared with the move already realized since that month end.

Usage
-----
python deleveraging_sensitivity.py [--spx-now 6946] [--xlk-now 150]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from margin_debt_market_cap import load_finra

YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1=852076800&period2={}&interval=1d"
TICKERS = {"S&P 500": "^GSPC", "XLK": "XLK"}
DD_THRESHOLD = -0.10
SCENARIOS = [-0.10, -0.20, -0.30, -0.40]

INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = {"S&P 500": "#2a78d6", "XLK": "#1baf7a"}
MARGIN_COLOR = "#6250d6"
BAND = "#f0efec"


# --------------------------------------------------------------------------- data
def load_yahoo(ticker: str) -> tuple[pd.Series, float]:
    """Daily split-adjusted closes and the latest quote."""
    url = YAHOO.format(requests.utils.quote(ticker), int(pd.Timestamp.now().timestamp()) + 86400)
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    s = pd.Series(res["indicators"]["quote"][0]["close"],
                  index=pd.to_datetime(res["timestamp"], unit="s").normalize(), name=ticker)
    return s.dropna().astype(float), float(res["meta"]["regularMarketPrice"])


def month_end(s: pd.Series) -> pd.Series:
    m = s.resample("ME").last().dropna()
    m.index = m.index.to_period("M")
    return m


def build_panel():
    finra = load_finra()
    prices, quotes = {}, {}
    for name, tk in TICKERS.items():
        daily, quotes[name] = load_yahoo(tk)
        prices[name] = month_end(daily)
    last = finra["margin_debt"].dropna().index[-1]
    p = pd.DataFrame({"margin_debt": finra["margin_debt"], **prices}).loc[:last]
    return p, quotes


# ---------------------------------------------------------------------- analysis
def episodes(p: pd.DataFrame) -> pd.DataFrame:
    md = p["margin_debt"]
    dd = md / md.rolling(12, min_periods=1).max() - 1
    hits = dd[dd <= DD_THRESHOLD].index
    clusters, cur = [], [hits[0]]
    for m in hits[1:]:
        if (m - cur[-1]).n <= 3:
            cur.append(m)
        else:
            clusters.append(cur)
            cur = [m]
    clusters.append(cur)

    rows = []
    for c in clusters:
        trough = dd.loc[c[0]:c[-1]].idxmin()
        peak = md.loc[trough - 12:trough].idxmax()
        row = {"peak": peak, "trough": trough, "months": (trough - peak).n,
               "margin_debt_chg": md[trough] / md[peak] - 1}
        for name in TICKERS:
            px = p[name]
            ok = pd.notna(px.get(peak))
            seg = px.loc[peak:trough]
            row[f"{name}_chg"] = px[trough] / px[peak] - 1 if ok else np.nan
            row[f"{name}_max_dd"] = (seg / seg.cummax() - 1).min() if ok else np.nan
            fwd = px.shift(-3).get(trough)
            row[f"{name}_next3m"] = fwd / px[trough] - 1 if ok and pd.notna(fwd) else np.nan
            row[f"{name}_elasticity"] = row[f"{name}_chg"] / row["margin_debt_chg"]
        rows.append(row)
    return pd.DataFrame(rows)


def elasticities(ep: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for name in TICKERS:
        d = ep.dropna(subset=[f"{name}_chg"])
        x, y = d["margin_debt_chg"], d[f"{name}_chg"]
        e = d[f"{name}_elasticity"]
        out[name] = {"n_episodes": len(d), "slope_thru_origin": (x * y).sum() / (x ** 2).sum(),
                     "median": e.median(), "p25": e.quantile(0.25), "p75": e.quantile(0.75)}
    return pd.DataFrame(out).T


def monthly_regressions(p: pd.DataFrame) -> pd.DataFrame:
    r = np.log(p).diff().dropna()
    g = r["margin_debt"]
    rows = {}
    for name in TICKERS:
        y = r[name]
        row = {"beta_all": np.polyfit(g, y, 1)[0], "corr_all": g.corr(y),
               "beta_down": np.polyfit(g[g < 0], y[g < 0], 1)[0]}
        big = g <= -0.05
        row.update({"n_months_md_le_-5%": int(big.sum()), "avg_md_chg_those_months": g[big].mean(),
                    "avg_index_ret_those_months": y[big].mean()})
        for lag in (-1, 0, 1):
            row[f"corr_md_t_vs_ret_t{lag:+d}"] = g.corr(y.shift(-lag))
        rows[name] = row
    down = r["S&P 500"] < 0
    rows["XLK"]["beta_to_spx_down_months"] = np.polyfit(r.loc[down, "S&P 500"], r.loc[down, "XLK"], 1)[0]
    return pd.DataFrame(rows).T


def scenarios(p: pd.DataFrame, el: pd.DataFrame, now: dict) -> pd.DataFrame:
    last = p.index[-1]
    rows = []
    for s in SCENARIOS:
        row = {"margin_debt_chg": s, "margin_debt_musd": p["margin_debt"].iloc[-1] * (1 + s),
               "deleveraging_musd": -p["margin_debt"].iloc[-1] * s}
        for name in TICKERS:
            base = p.at[last, name]
            row[f"{name}_chg"] = el.at[name, "slope_thru_origin"] * s
            row[f"{name}_chg_lo"] = el.at[name, "p75"] * s
            row[f"{name}_chg_hi"] = el.at[name, "p25"] * s
            row[f"{name}_level"] = base * (1 + row[f"{name}_chg"])
        rows.append(row)
    sc = pd.DataFrame(rows)
    realized = {n: now[n] / p.at[last, n] - 1 for n in TICKERS}
    implied = {n: realized[n] / el.at[n, "slope_thru_origin"] for n in TICKERS}
    return sc, realized, implied


# --------------------------------------------------------------------------- plot
def _style(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": BASE,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })


def _title(ax, text):
    ax.set_title(text, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=1)


def _pct(ax):
    import matplotlib.ticker as mt
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:.0%}"))


def plot_timeline(p, ep, now, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    _style(plt)
    ts = lambda idx: idx.to_timestamp(how="end")  # noqa: E731
    t_now = pd.Timestamp.today().normalize()

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax in axes:
        for _, e in ep.iterrows():
            ax.axvspan(e["peak"].to_timestamp(how="end"), e["trough"].to_timestamp(how="end"),
                       color=BAND, lw=0, zorder=0)

    md = p["margin_debt"]
    dd = md / md.rolling(12, min_periods=1).max() - 1
    ax = axes[0]
    ax.plot(ts(dd.index), dd.values, color=MARGIN_COLOR, lw=2)
    ax.axhline(DD_THRESHOLD, color=MUTED, lw=1)
    ax.text(ts(dd.index)[0], DD_THRESHOLD, " -10% threshold", color=INK2, fontsize=8, va="bottom")
    ax.axhline(0, color=BASE, lw=1)
    _pct(ax)
    _title(ax, "Margin debt vs. trailing 12-month high  ·  shaded = deleveraging episodes")

    ax = axes[1]
    for name in TICKERS:
        px = p[name].dropna()
        d = px / px.rolling(12, min_periods=1).max() - 1
        hi = px.iloc[-12:].max()
        x = list(ts(d.index)) + [t_now]
        y = list(d.values) + [min(now[name] / hi - 1, 0)]
        ax.plot(x[:-1], y[:-1], color=COLORS[name], lw=1.6, label=name)
        ax.plot(x[-2:], y[-2:], color=COLORS[name], lw=1.6, ls=(0, (2, 2)))
        ax.plot(x[-1:], y[-1:], "o", ms=8, color=COLORS[name], mec=SURFACE, mew=2)
        ax.annotate(f"{name}\ntoday {y[-1]:.0%}", (x[-1], y[-1]), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=9, color=INK)
    ax.axhline(0, color=BASE, lw=1)
    _pct(ax)
    ax.legend(loc="lower left", frameon=False, labelcolor=INK2)
    _title(ax, "Index vs. trailing 12-month high (month-end; dot = today intraday)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Margin deleveraging episodes and equity drawdowns", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "Sources: FINRA margin statistics; Yahoo Finance (^GSPC, XLK, split-adjusted price).",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sensitivity(ep, el, sc, realized, implied, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    import matplotlib.ticker as mt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})

    # episodes scatter with through-origin fits
    xs = np.linspace(-0.6, 0, 50)
    for name in TICKERS:
        d = ep.dropna(subset=[f"{name}_chg"])
        b = el.at[name, "slope_thru_origin"]
        a1.plot(xs, b * xs, color=COLORS[name], lw=1, alpha=0.6)
        a1.scatter(d["margin_debt_chg"], d[f"{name}_chg"], s=56, color=COLORS[name],
                   edgecolors=SURFACE, linewidths=2, zorder=3, label=f"{name}  (fit: {b:.2f}× margin debt)")
    for _, e in ep.iterrows():
        a1.annotate(str(e["trough"].year), (e["margin_debt_chg"], e["S&P 500_chg"]), xytext=(5, 4),
                    textcoords="offset points", fontsize=8, color=INK2)
    a1.axhline(0, color=BASE, lw=1)
    a1.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _pct(a1)
    a1.set_xlabel("Margin debt change, 12-month high → trough")
    a1.set_ylabel("Index change over the same window")
    a1.legend(loc="upper left", frameon=False, labelcolor=INK2, fontsize=9)
    _title(a1, "Deleveraging episodes since 1998")

    # scenario bars
    x = np.arange(len(sc))
    w = 0.26
    for i, name in enumerate(TICKERS):
        vals = sc[f"{name}_chg"].values
        lo, hi = sc[f"{name}_chg_lo"].values, sc[f"{name}_chg_hi"].values
        pos = x + (i - 0.5) * w
        a2.bar(pos, vals, width=w * 0.92, color=COLORS[name], label=name, zorder=2)
        a2.vlines(pos, lo, hi, color=INK2, lw=1, zorder=3)  # fit can sit outside the IQR
        for xx, v, l in zip(pos, vals, lo):
            a2.text(xx, min(v, l) - 0.012, f"{v:.0%}", ha="center", va="top", fontsize=8, color=INK)
        a2.axhline(realized[name], color=COLORS[name], lw=1, zorder=1,
                   label=f"{name} already {realized[name]:.0%} since month end (≈ margin debt {implied[name]:.0%})")
    a2.axhline(0, color=BASE, lw=1)
    a2.set_xticks(x, [f"{s:.0%}\n(${-s * sc['margin_debt_musd'].iloc[0] / (1 + sc['margin_debt_chg'].iloc[0]) / 1e6:.2f} tn)"
                      for s in sc["margin_debt_chg"]])
    a2.set_xlabel("Margin debt shock from latest FINRA month")
    _pct(a2)
    a2.legend(loc="lower left", frameon=False, labelcolor=INK2, fontsize=9)
    _title(a2, "Implied index move (bar = fit, whisker = episode IQR)")

    fig.text(0.01, 0.005, "Elasticities are co-movement, not causality: prices lead margin debt by about a month.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--spx-now", type=float, help="current S&P 500 (default: latest Yahoo quote)")
    ap.add_argument("--xlk-now", type=float, help="current XLK (default: latest Yahoo quote)")
    args = ap.parse_args()

    p, quotes = build_panel()
    now = {"S&P 500": args.spx_now or quotes["S&P 500"], "XLK": args.xlk_now or quotes["XLK"]}

    ep = episodes(p)
    el = elasticities(ep)
    reg = monthly_regressions(p)
    sc, realized, implied = scenarios(p, el, now)

    (args.out_dir / "data").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "images").mkdir(parents=True, exist_ok=True)
    ep.to_csv(args.out_dir / "data" / "deleveraging_episodes.csv", index=False, float_format="%.4f")
    sc.to_csv(args.out_dir / "data" / "deleveraging_scenarios.csv", index=False, float_format="%.4f")
    plot_timeline(p, ep, now, args.out_dir / "images" / "deleveraging_timeline.png")
    plot_sensitivity(ep, el, sc, realized, implied, args.out_dir / "images" / "deleveraging_sensitivity.png")

    pd.set_option("display.width", 200)
    last = p.index[-1]
    print(f"Latest FINRA month {last}: margin debt ${p.at[last, 'margin_debt'] / 1e6:.3f} tn")
    print("Now:", {k: round(v, 2) for k, v in now.items()},
          "| since month end:", {k: f"{v:.1%}" for k, v in realized.items()},
          "| implied margin debt move:", {k: f"{v:.1%}" for k, v in implied.items()})
    cols = ["peak", "trough", "months", "margin_debt_chg"] + [f"{n}_{k}" for n in TICKERS
                                                                for k in ("chg", "max_dd", "next3m")]
    print("\nEpisodes\n", ep[cols].astype({"peak": str, "trough": str}).round(3).to_string(index=False))
    print("\nEpisode elasticities (index change / margin debt change)\n", el.round(2).to_string())
    print("\nMonthly regressions (log changes)\n", reg.astype(float).round(2).to_string())
    print("\nScenarios\n", sc.drop(columns=[c for c in sc if c.endswith(("_lo", "_hi"))]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
