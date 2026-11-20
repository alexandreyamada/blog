"""
Correlation between net margin debt / market cap and equity indices.

Net margin debt = FINRA debit balances - free credit balances (cash + margin accounts).
Denominator = Fed Z.1 nonfinancial corporate equities (FRED NCBEILQ027S), quarter end.
Everything is measured at quarter ends, the only dates where the ratio uses real Z.1 data.

Indices (FRED, no API key)
* SP500      - S&P 500 price index (FRED only carries the last ~10 years: 2016-11 onward)
* NASDAQ100  - NASDAQ-100 price index, 1986 onward (long-history tech proxy)
* XLK        - optional, from a local CSV (Date + Close/Adj Close), e.g. a Yahoo export:
               --xlk-csv path/to/XLK.csv

Statistics per index
* level   : corr(net ratio, log price)                        - trend-driven, read with care
* change  : corr(quarterly change in net ratio, quarterly log return)
* forward : corr(net ratio at t, log return t -> t+h quarters), h = 1, 2, 4, 8, 12
Forward windows overlap, so the effective sample is roughly n / h.

Usage
-----
python margin_vs_indices.py [--spx-now 6946] [--xlk-now 150] [--xlk-csv XLK.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from margin_debt_market_cap import (
    MCAP_SERIES, SP500_SERIES, build_indicator, load_finra, load_fred, quarter_end_month,
)

HORIZONS = [1, 2, 4, 8, 12]
ROLL_Q = 12  # rolling window, quarters

# Palette (light). Identity: each index keeps its hue in every panel.
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = {"S&P 500": "#2a78d6", "NASDAQ-100": "#eb6834", "XLK": "#1baf7a"}
RATIO_COLOR = "#6250d6"


# --------------------------------------------------------------------------- data
def month_end(s: pd.Series) -> pd.Series:
    m = s.resample("ME").last().dropna()
    m.index = m.index.to_period("M")
    return m


def load_xlk_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    date_col = next(c for c in df.columns if c.lower() in ("date", "datetime", "timestamp"))
    px_col = next(c for c in ("Close", "close", "Adj Close", "adj_close", "adjclose") if c in df.columns)
    s = pd.Series(pd.to_numeric(df[px_col], errors="coerce").values,
                  index=pd.to_datetime(df[date_col]), name="XLK").dropna().sort_index()
    return s


def load_inputs(xlk_csv: Path | None):
    ind = build_indicator(load_finra(), load_fred(MCAP_SERIES), load_fred(SP500_SERIES))
    sp_daily = load_fred(SP500_SERIES)
    prices = {"S&P 500": month_end(sp_daily), "NASDAQ-100": month_end(load_fred("NASDAQ100"))}
    if xlk_csv:
        prices["XLK"] = month_end(load_xlk_csv(xlk_csv))
    first = ind["margin_debt"].dropna().index[0]
    prices = {k: v.loc[first:] for k, v in prices.items()}
    return ind, prices, sp_daily


def today_estimate(ind: pd.DataFrame, mcap_q: pd.Series, sp_daily: pd.Series, spx_now: float) -> dict:
    """Net ratio today: latest FINRA balances over market cap scaled to the S&P level given."""
    mcap = quarter_end_month(mcap_q)
    sp_m = month_end(sp_daily)
    mcap_now = mcap.iloc[-1] * spx_now / sp_m[mcap.index[-1]]
    last = ind["margin_debt"].dropna().index[-1]
    net = ind.at[last, "margin_debt"] - ind.at[last, "free_credit_total"]
    return {"finra_month": str(last), "market_cap_musd": mcap_now,
            "net_ratio_pct": net / mcap_now * 100,
            "gross_ratio_pct": ind.at[last, "margin_debt"] / mcap_now * 100}


# ---------------------------------------------------------------------- analysis
def quarterly_panel(ind: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    z1 = ind[ind["mcap_source"] == "Z.1"]
    q = pd.DataFrame({"net_ratio": z1["net_margin_debt_to_mcap_pct"]})
    for name, s in prices.items():
        q[name] = s.reindex(q.index)
    return q


def correlations(q: pd.DataFrame, names: list[str], start: str | None = None) -> pd.DataFrame:
    d = q.loc[start:] if start else q
    rows = {}
    for name in names:
        lp = np.log(d[name])
        ok = lp.notna()
        r = {"window": f"{lp[ok].index[0]}..{lp[ok].index[-1]}", "n_quarters": int(ok.sum()),
             "level": d["net_ratio"].corr(lp),
             "change": d["net_ratio"].diff().corr(lp.diff())}
        for h in HORIZONS:
            r[f"fwd_{h}q"] = d["net_ratio"].corr(lp.shift(-h) - lp)
        rows[name] = r
    return pd.DataFrame(rows).T


def bucket_table(q: pd.DataFrame, name: str, h: int = 4) -> pd.DataFrame:
    """Forward h-quarter simple return by net-ratio quartile."""
    fwd = (q[name].shift(-h) / q[name] - 1) * 100
    d = pd.DataFrame({"net_ratio": q["net_ratio"], "fwd": fwd}).dropna()
    d["quartile"] = pd.qcut(d["net_ratio"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    g = d.groupby("quartile", observed=True)
    return pd.DataFrame({
        "net_ratio_range": g["net_ratio"].agg(lambda s: f"{s.min():.2f}..{s.max():.2f}"),
        "n": g.size(),
        "median_fwd_pct": g["fwd"].median().round(1),
        "share_negative_pct": (g["fwd"].apply(lambda s: (s < 0).mean() * 100)).round(0),
        "worst_fwd_pct": g["fwd"].min().round(1),
    })


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


def plot_timeseries(ind, q, prices, today, now_levels, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    _style(plt)
    ts = lambda idx: idx.to_timestamp(how="end")  # noqa: E731

    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1.5, 1.2]})

    # 1) net ratio
    ax = axes[0]
    s = q["net_ratio"].dropna()
    ax.plot(ts(s.index), s.values, color=RATIO_COLOR, lw=2)
    nc = ind.loc[ind["mcap_source"] == "nowcast_sp500", "net_margin_debt_to_mcap_pct"]
    t_now = pd.Timestamp.today().normalize()
    bridge_x = list(ts(s.index[-1:])) + list(ts(nc.index)) + [t_now]
    bridge_y = list(s.values[-1:]) + list(nc.values) + [today["net_ratio_pct"]]
    ax.plot(bridge_x, bridge_y, color=RATIO_COLOR, lw=2, ls=(0, (2, 2)))
    ax.plot([t_now], [today["net_ratio_pct"]], "o", ms=8, color=RATIO_COLOR, mec=SURFACE, mew=2)
    ax.axhline(0, color=BASE, lw=1)
    _title(ax, f"Net margin debt / market cap (%)  ·  today ≈ {today['net_ratio_pct']:.2f}% "
               f"(FINRA {today['finra_month']}, S&P {now_levels['S&P 500']:,.0f})  ·  dashed = nowcast")

    # 2) index levels, log, rebased
    ax = axes[1]
    base = "2016-12"
    for name, px in prices.items():
        p = px.dropna()
        if pd.Period(base, "M") < p.index[0]:
            continue
        reb = p / p[base] * 100
        ax.plot(ts(reb.index), reb.values, color=COLORS[name], lw=2, label=name)
        last_x, last_y = ts(reb.index)[-1], reb.values[-1]
        if name in now_levels and now_levels[name]:
            y_now = now_levels[name] / p[base] * 100
            ax.plot([last_x, t_now], [last_y, y_now], color=COLORS[name], lw=2, ls=(0, (2, 2)))
            last_x, last_y = t_now, y_now
        ax.plot([last_x], [last_y], "o", ms=8, color=COLORS[name], mec=SURFACE, mew=2)
        ax.annotate(name, (last_x, last_y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([10, 20, 50, 100, 200, 500, 1000]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(loc="upper left", frameon=False, labelcolor=INK2)
    _title(ax, f"Index level, log scale ({base} = 100)")

    # 3) rolling correlation of changes
    ax = axes[2]
    dnet = q["net_ratio"].diff()
    for name in prices:
        rc = dnet.rolling(ROLL_Q).corr(np.log(q[name]).diff()).dropna()
        if rc.empty:
            continue
        ax.plot(ts(rc.index), rc.values, color=COLORS[name], lw=2, label=name)
        ax.plot(ts(rc.index[-1:]), rc.values[-1:], "o", ms=8, color=COLORS[name], mec=SURFACE, mew=2)
    ax.axhline(0, color=BASE, lw=1)
    ax.set_ylim(-1, 1)
    ax.legend(loc="lower left", frameon=False, labelcolor=INK2)
    _title(ax, f"Rolling {ROLL_Q}-quarter correlation: Δ net ratio vs index return")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Net margin debt / market cap vs. equity indices", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "Sources: FINRA margin statistics; Fed Z.1 (FRED NCBEILQ027S); FRED SP500, NASDAQ100. "
             "Quarter-end data; dashed = nowcast with latest FINRA month.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forward(q, corr_tbl, today, path: Path, h: int = 4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.3, 1]})

    # scatter: ratio vs next-h-quarter return (longest history drawn first, shorter on top)
    for name in sorted((n for n in COLORS if n in q), key=lambda n: -q[n].notna().sum()):
        fwd = (q[name].shift(-h) / q[name] - 1) * 100
        d = pd.DataFrame({"x": q["net_ratio"], "y": fwd}).dropna()
        if d.empty:
            continue
        label = f"{name} ({d.index[0].year}–, ρ = {d['x'].corr(np.log1p(d['y'] / 100)):+.2f})"
        a1.scatter(d["x"], d["y"], s=40, color=COLORS[name], edgecolors=SURFACE, linewidths=1.5,
                   alpha=0.9, label=label, zorder=3)
    a1.axhline(0, color=BASE, lw=1)
    a1.axvline(today["net_ratio_pct"], color=INK2, lw=1)
    a1.text(today["net_ratio_pct"], a1.get_ylim()[1], f" today ≈ {today['net_ratio_pct']:.2f}%",
            va="top", ha="right", fontsize=9, color=INK2)
    a1.set_xlabel("Net margin debt / market cap at quarter end (%)")
    a1.set_ylabel(f"Return over next {h} quarters (%)")
    a1.legend(loc="lower left", frameon=False, labelcolor=INK2, fontsize=9)
    _title(a1, f"Net ratio vs. next-{h * 3}-month index return")

    # correlation by horizon
    names = [n for n in corr_tbl.index]
    width = min(0.8 / len(names), 0.3)
    x = np.arange(len(HORIZONS))
    for i, name in enumerate(names):
        vals = corr_tbl.loc[name, [f"fwd_{k}q" for k in HORIZONS]].astype(float).values
        a2.bar(x + (i - (len(names) - 1) / 2) * width, vals, width=width * 0.9,
               color=COLORS.get(name.split(" (")[0], MUTED), label=f"{name} {corr_tbl.loc[name, 'window']}")
    a2.axhline(0, color=BASE, lw=1)
    a2.set_xticks(x, [f"{k * 3}m" for k in HORIZONS])
    a2.set_ylim(-1, 1)
    a2.set_xlabel("Forward horizon")
    a2.legend(loc="lower left", frameon=False, labelcolor=INK2, fontsize=8)
    _title(a2, "Corr(net ratio, forward return) by horizon")

    fig.text(0.01, 0.005, "Forward windows overlap: effective sample ≈ quarters / horizon. "
             "ρ on log returns.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--spx-now", type=float, help="current S&P 500 level (default: last FRED close)")
    ap.add_argument("--xlk-now", type=float, help="current XLK price (used only with --xlk-csv)")
    ap.add_argument("--xlk-csv", type=Path, help="XLK daily prices CSV (Date + Close/Adj Close)")
    args = ap.parse_args()

    ind, prices, sp_daily = load_inputs(args.xlk_csv)
    spx_now = args.spx_now or float(sp_daily.iloc[-1])
    now_levels = {"S&P 500": spx_now, "XLK": args.xlk_now if args.xlk_csv else None}
    today = today_estimate(ind, load_fred(MCAP_SERIES), sp_daily, spx_now)

    q = quarterly_panel(ind, prices)
    names = list(prices)
    full = correlations(q, names)
    common_start = max(prices[n].index[0] for n in names)
    common = correlations(q, names, start=str(max(common_start, q.index[0])))
    common = common[common["n_quarters"] != full["n_quarters"]]  # only indices with longer history
    common.index = [f"{n} (common window)" for n in common.index]
    tbl = pd.concat([full, common])

    (args.out_dir / "data").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "images").mkdir(parents=True, exist_ok=True)
    tbl.to_csv(args.out_dir / "data" / "net_margin_vs_indices_corr.csv", float_format="%.4f")
    q.to_csv(args.out_dir / "data" / "net_margin_vs_indices_quarterly.csv", float_format="%.4f")

    plot_timeseries(ind, q, prices, today, now_levels,
                    args.out_dir / "images" / "net_margin_vs_indices_timeseries.png")
    plot_forward(q, full, today, args.out_dir / "images" / "net_margin_vs_indices_forward.png")

    pd.set_option("display.width", 160)
    print("Today:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in today.items()})
    print("\nCorrelations with net margin debt / market cap")
    print(tbl.drop(columns="window").astype(float).round(2).assign(window=tbl["window"]).to_string())
    longest = max(names, key=lambda n: q[n].notna().sum())
    print(f"\nNext-12-month {longest} return by net-ratio quartile")
    print(bucket_table(q, longest).to_string())


if __name__ == "__main__":
    main()
