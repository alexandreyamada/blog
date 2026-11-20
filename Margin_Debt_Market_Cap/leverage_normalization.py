"""
S&P 500 level at which net margin leverage returns to its historical average.

Net ratio = (margin debt D - free credits C) / market cap M.
A falling S&P alone RAISES the ratio (M falls, D does not). Leverage normalizes only if margin
debt is repaid, which historically happens together with the price decline. So the S&P level is
solved on a joint path calibrated on past deleveraging episodes:

    S&P change p (from the latest FINRA month end), M = M0 (1 + p)
    D = D0 (1 + a p)        a = debit elasticity to the S&P in episodes
    C = C0 (1 + c p)        c = free-credit elasticity to the S&P in episodes
    solve (D - C) / M = target  ->  p = (D0 - C0 - T M0) / (T M0 - a D0 + c C0)

Scenarios: "central" uses through-origin episode fits; "fast" uses the episode medians
(debit falls faster and investors raise cash, as in 2000, 2011 and 2020).
Targets: average net ratio over the full sample and over recent windows - free credits fell from
~1x margin debt (2002-12) to ~0.3x, so the full-sample mean is a biased anchor.

Monthly market cap = last Z.1 quarter-end level scaled by the S&P 500.

Usage
-----
python leverage_normalization.py [--spx-now 6946]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from deleveraging_sensitivity import load_yahoo, month_end
from margin_debt_market_cap import MCAP_SERIES, load_finra, load_fred, quarter_end_month

EPISODES = [("1998-07", "1998-10"), ("2000-09", "2001-08"), ("2008-02", "2009-01"), ("2010-04", "2010-06"),
            ("2011-04", "2011-09"), ("2015-04", "2016-02"), ("2018-05", "2018-12"), ("2019-07", "2020-03"),
            ("2021-10", "2022-10"), ("2023-07", "2023-10")]  # from deleveraging_sensitivity.episodes()
WINDOWS = {"1997-2026 mean": ("1997", None), "2013-2026 mean": ("2013", None), "Last 10y mean": (None, 120)}

INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SCEN_COLORS = {"central": "#2a78d6", "fast": "#eb6834", "debt frozen": "#898781"}


def load(spx_now: float | None):
    finra = load_finra()
    D, C = finra["margin_debt"].astype(float), finra["free_credit_total"].astype(float)
    spx_d, quote = load_yahoo("^GSPC", start="1996-01-01")
    spx = month_end(spx_d)
    z1 = quarter_end_month(load_fred(MCAP_SERIES))
    mc = pd.Series(np.nan, index=D.index)
    for t in D.index:
        q = z1[z1.index <= t]
        if len(q) and t in spx.index and q.index[-1] in spx.index:
            mc[t] = q.iloc[-1] * spx[t] / spx[q.index[-1]]
    return D, C, mc, spx, spx_now or quote


def elasticities(D, C, spx):
    e = pd.DataFrame([{"spx": spx[pd.Period(b, "M")] / spx[pd.Period(a, "M")] - 1,
                       "debit": D[pd.Period(b, "M")] / D[pd.Period(a, "M")] - 1,
                       "credit": C[pd.Period(b, "M")] / C[pd.Period(a, "M")] - 1} for a, b in EPISODES])
    fit = lambda k: (e[k] * e["spx"]).sum() / (e["spx"] ** 2).sum()  # noqa: E731
    return {"central": (fit("debit"), fit("credit")),
            "fast": ((e["debit"] / e["spx"]).median(), (e["credit"] / e["spx"]).median()),
            "debt frozen": (0.0, 0.0)}


def net_path(p, D0, C0, M0, a, c):
    return (D0 * (1 + a * p) - C0 * (1 + c * p)) / (M0 * (1 + p)) * 100


def solve_p(T, D0, C0, M0, a, c):
    t = T / 100
    den = t * M0 - a * D0 + c * C0
    p = (D0 - C0 - t * M0) / den
    return p if -0.95 < p < 3 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--spx-now", type=float)
    args = ap.parse_args()

    D, C, mc, spx, spx_now = load(args.spx_now)
    net = ((D - C) / mc * 100).dropna()
    gross = (D / mc * 100).dropna()
    base = net.index[-1]
    D0, C0, M0, S0 = D[base], C[base], mc[base], spx[base]

    targets = {"1997-2026 mean": net.mean(), "2013-2026 mean": net.loc["2013":].mean(),
               "Last 10y mean": net.iloc[-120:].mean()}
    el = elasticities(D, C, spx)

    rows = []
    for name, T in targets.items():
        row = {"target": name, "net_ratio_pct": T}
        for scen, (a, c) in el.items():
            p = solve_p(T, D0, C0, M0, a, c)
            row[f"{scen}_spx"] = S0 * (1 + p) if pd.notna(p) else np.nan
            row[f"{scen}_vs_now"] = S0 * (1 + p) / spx_now - 1 if pd.notna(p) else np.nan
            row[f"{scen}_margin_debt_chg"] = a * p if pd.notna(p) else np.nan
        rows.append(row)
    tbl = pd.DataFrame(rows)

    p_now = spx_now / S0 - 1
    now_est = {s: net_path(p_now, D0, C0, M0, a, c) for s, (a, c) in el.items()}
    gross_flat_spx = S0 * gross[base] / gross.mean()

    (args.out_dir / "data").mkdir(parents=True, exist_ok=True)
    tbl.to_csv(args.out_dir / "data" / "leverage_normalization.csv", index=False, float_format="%.4f")
    plot(D0, C0, M0, S0, spx_now, el, targets, now_est, args.out_dir / "images" / "leverage_normalization.png")

    pd.set_option("display.width", 200)
    print(f"Base {base}: margin debt ${D0 / 1e6:.3f} tn, free credits ${C0 / 1e6:.3f} tn, S&P {S0:,.0f}, "
          f"net {net[base]:.2f}%, gross {gross[base]:.2f}% (gross mean {gross.mean():.2f}%)")
    print("Elasticities to S&P (debit, credit):", {k: tuple(round(x, 2) for x in v) for k, v in el.items()})
    print(f"S&P now {spx_now:,.0f} ({p_now:+.1%} vs base). Net ratio now by scenario:",
          {k: round(v, 2) for k, v in now_est.items()})
    print(f"Gross ratio is at its mean with debt unchanged and S&P at {gross_flat_spx:,.0f}")
    print(tbl.round(3).to_string(index=False))


def plot(D0, C0, M0, S0, spx_now, el, targets, now_est, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": BASE,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })
    fig, ax = plt.subplots(figsize=(11, 6))
    levels = np.linspace(S0, S0 * 0.45, 200)
    p = levels / S0 - 1
    for scen, (a, c) in el.items():
        y = net_path(p, D0, C0, M0, a, c)
        per = f"per 1% S&P drop: margin debt {-a:+.2f}%, free credits {-c:+.2f}%"
        lbl = {"central": f"Central ({per})", "fast": f"Fast ({per})",
               "debt frozen": "No deleveraging (balances frozen)"}[scen]
        ax.plot(levels, y, color=SCEN_COLORS[scen], lw=2, label=lbl)
    for i, (name, T) in enumerate(targets.items()):
        ax.axhline(T, color=BASE, lw=1, zorder=0)
        ax.text(S0, T, f" {name} {T:.2f}%", ha="left", va="top" if i == 1 else "bottom", fontsize=8, color=INK2)
        for scen in ("central", "fast"):
            pp = solve_p(T, D0, C0, M0, *el[scen])
            if pd.notna(pp):
                x = S0 * (1 + pp)
                ax.plot([x], [T], "o", ms=8, color=SCEN_COLORS[scen], mec=SURFACE, mew=2, zorder=4)
                if i != 1:
                    ax.annotate(f"{x:,.0f}", (x, T), xytext=(0, 9 if i == 0 else -9), textcoords="offset points",
                                ha="center", va="bottom" if i == 0 else "top", fontsize=8, color=INK)
    ax.axvline(spx_now, color=INK2, lw=1)
    ax.text(spx_now, ax.get_ylim()[1], f" S&P now {spx_now:,.0f}", va="top", fontsize=9, color=INK2)
    ax.invert_xaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("S&P 500 level")
    ax.set_ylabel("Net margin debt / market cap (%)")
    ax.legend(loc="lower center", frameon=False, labelcolor=INK2, fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_title("Net leverage on a deleveraging path vs. S&P 500 level", loc="left", color=INK,
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.005, f"Start: latest FINRA month end (S&P {S0:,.0f}). Elasticities from 10 deleveraging "
             "episodes since 1998.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
