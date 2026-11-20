"""
Margin debt / total US equity market cap.

Sources (no API key needed)
---------------------------
* FINRA Margin Statistics, monthly, USD millions
  - Debit Balances in Customers' Securities Margin Accounts  -> margin debt
  - Free Credit Balances in Customers' Cash / Margin Accounts -> investor cash cushion
  https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics
* FRED / Fed Z.1 Financial Accounts, quarterly, USD millions
  - NCBEILQ027S: Nonfinancial corporate business; corporate equities; liability, level
    (market value of US corporate equity at quarter end - the usual FRED proxy for total
    US market cap since the Wilshire 5000 series was removed from FRED in 2024)
* FRED SP500 (daily) - used only to nowcast market cap for the months after the
  latest Z.1 release (Z.1 is published ~10 weeks after quarter end).

Output
------
data/margin_debt_market_cap.csv  monthly table (ratio filled at quarter ends + nowcast months)
images/margin_debt_market_cap.png

Usage
-----
python margin_debt_market_cap.py [--out-dir DIR] [--no-plot]
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import pandas as pd
import requests

FINRA_PAGE = "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics"
FINRA_XLSX_FALLBACK = "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

MCAP_SERIES = "NCBEILQ027S"
SP500_SERIES = "SP500"

# FINRA's CDN rejects the default python-requests agent; FRED rejects browser-like agents.
BROWSER_UA = {"User-Agent": "Mozilla/5.0"}


# --------------------------------------------------------------------------- data
def _get(url: str) -> bytes:
    headers = BROWSER_UA if "finra.org" in url else None
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.content


def finra_xlsx_url() -> str:
    """FINRA occasionally moves the file; read the current link from the page."""
    try:
        html = _get(FINRA_PAGE).decode("utf-8", "ignore")
        m = re.search(r'href="([^"]*margin-statistics[^"]*\.xlsx)"', html)
        if m:
            href = m.group(1)
            return href if href.startswith("http") else "https://www.finra.org" + href
    except requests.RequestException:
        pass
    return FINRA_XLSX_FALLBACK


def load_finra() -> pd.DataFrame:
    """Monthly FINRA margin statistics (USD millions), PeriodIndex[M], ascending."""
    raw = pd.read_excel(io.BytesIO(_get(finra_xlsx_url())))
    raw.columns = ["year_month", "margin_debt", "free_credit_cash", "free_credit_margin"]
    raw = raw.dropna(subset=["year_month", "margin_debt"])
    raw.index = pd.PeriodIndex(raw.pop("year_month").astype(str), freq="M")
    df = raw.apply(pd.to_numeric, errors="coerce").sort_index()
    # Until Jan-2010 (NYSE collection) the cash-account column carries all free credits;
    # from Feb-2010 (FINRA Rule 4521) they are split. The sum is continuous across the break.
    df["free_credit_total"] = df["free_credit_cash"] + df["free_credit_margin"].fillna(0)
    return df


def load_fred(series_id: str) -> pd.Series:
    s = pd.read_csv(io.BytesIO(_get(FRED_CSV.format(series_id))), index_col=0, parse_dates=True)
    return pd.to_numeric(s.iloc[:, 0], errors="coerce").dropna().rename(series_id)


# ---------------------------------------------------------------------- indicator
def quarter_end_month(s: pd.Series) -> pd.Series:
    """FRED stamps Z.1 levels on the quarter's first day; the level is as of quarter end."""
    out = s.copy()
    out.index = out.index.to_period("Q").asfreq("M", "end")
    return out


def build_indicator(finra: pd.DataFrame, mcap_q: pd.Series, sp500: pd.Series) -> pd.DataFrame:
    df = finra[["margin_debt", "free_credit_total"]].copy()
    df["margin_debt_yoy_pct"] = df["margin_debt"].pct_change(12, fill_method=None) * 100

    mcap = quarter_end_month(mcap_q)
    df["market_cap"] = mcap.reindex(df.index)
    df["mcap_source"] = pd.Series("Z.1", index=mcap.index).reindex(df.index)

    # Nowcast: months after the last Z.1 quarter, scaled by S&P 500 month-end moves.
    last_q = mcap.index[-1]
    sp_m = sp500.resample("ME").last()
    sp_m.index = sp_m.index.to_period("M")
    after = df.index[df.index > last_q]
    if len(after) and last_q in sp_m.index:
        df.loc[after, "market_cap"] = mcap.iloc[-1] * sp_m.reindex(after) / sp_m[last_q]
        df.loc[after, "mcap_source"] = "nowcast_sp500"

    df["margin_debt_to_mcap_pct"] = df["margin_debt"] / df["market_cap"] * 100
    df["net_margin_debt_to_mcap_pct"] = (
        (df["margin_debt"] - df["free_credit_total"]) / df["market_cap"] * 100
    )
    return df


def summarize(df: pd.DataFrame) -> dict:
    hist = df.loc[df["mcap_source"] == "Z.1", "margin_debt_to_mcap_pct"].dropna()
    net = df.loc[df["mcap_source"] == "Z.1", "net_margin_debt_to_mcap_pct"].dropna()
    cur = df["margin_debt_to_mcap_pct"].dropna()
    last = cur.index[-1]
    return {
        "latest_month": str(last),
        "latest_source": df.at[last, "mcap_source"],
        "latest_ratio_pct": round(cur.iloc[-1], 3),
        "latest_z1_quarter": str(hist.index[-1]),
        "latest_z1_ratio_pct": round(hist.iloc[-1], 3),
        "hist_mean_pct": round(hist.mean(), 3),
        "hist_std_pct": round(hist.std(), 3),
        "hist_max_pct": round(hist.max(), 3),
        "hist_max_month": str(hist.idxmax()),
        "latest_percentile": round((hist < cur.iloc[-1]).mean() * 100, 1),
        "latest_net_ratio_pct": round(net.iloc[-1], 3),
        "net_ratio_max_month": str(net.idxmax()),
        "margin_debt_latest_musd": int(df["margin_debt"].iloc[-1]),
        "margin_debt_record_month": str(df["margin_debt"].idxmax()),
        "margin_debt_yoy_pct": round(df["margin_debt_yoy_pct"].iloc[-1], 1),
    }


# --------------------------------------------------------------------------- plot
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, SURFACE = "#2a78d6", "#fcfcfb"
EVENTS = {"2000-03": "Dot-com", "2007-06": "GFC", "2021-09": "Meme/2021"}


def plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": BASE,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })

    z1 = df[df["mcap_source"] == "Z.1"]
    nc = df[df["mcap_source"] == "nowcast_sp500"]
    ts = lambda idx: idx.to_timestamp(how="end")  # noqa: E731

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.2, 1.2]})
    panels = [
        ("margin_debt_to_mcap_pct", "Margin debt / market cap (%)"),
        ("net_margin_debt_to_mcap_pct", "Net margin debt (debit - free credits) / market cap (%)"),
    ]
    for ax, (col, title) in zip(axes[:2], panels):
        s = z1[col].dropna()
        ax.plot(ts(s.index), s.values, color=BLUE, lw=2, solid_capstyle="round")
        if len(nc):
            bridge = pd.concat([s.iloc[[-1]], nc[col]])
            ax.plot(ts(bridge.index), bridge.values, color=BLUE, lw=2, ls=(0, (2, 2)))
        mean = s.mean()
        ax.axhline(mean, color=MUTED, lw=1)
        title = f"{title}  ·  grey line = {s.index[0].year}-{s.index[-1].year} avg {mean:.2f}%"
        last = (nc[col] if len(nc) else s).dropna()
        ax.plot(ts(last.index)[-1:], last.values[-1:], "o", ms=8, color=BLUE, mec=SURFACE, mew=2)
        ax.annotate(f"{last.values[-1]:.2f}%\n{last.index[-1]}", (ts(last.index)[-1], last.values[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
        ax.grid(axis="y", color=GRID, lw=1)

    for ym, label in EVENTS.items():
        p = pd.Period(ym, "M")
        for ax in axes:
            ax.axvline(p.to_timestamp(how="end"), color=GRID, lw=1, zorder=0)
        axes[0].text(p.to_timestamp(how="end"), axes[0].get_ylim()[1], f" {label}",
                     color=MUTED, fontsize=8, va="top")

    yoy = df["margin_debt_yoy_pct"].dropna()
    axes[2].plot(ts(yoy.index), yoy.values, color=BLUE, lw=1.5)
    axes[2].axhline(0, color=BASE, lw=1)
    axes[2].set_title("Margin debt, YoY growth (%)", loc="left", color=INK, fontsize=11, fontweight="bold")
    axes[2].grid(axis="y", color=GRID, lw=1)
    axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("US margin debt vs. total equity market cap", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 0.005,
             "Sources: FINRA margin statistics (monthly); Fed Z.1 via FRED NCBEILQ027S (quarter end). "
             "Dashed = nowcast: last Z.1 level scaled by FRED SP500.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    finra = load_finra()
    mcap_q = load_fred(MCAP_SERIES)
    sp500 = load_fred(SP500_SERIES)
    df = build_indicator(finra, mcap_q, sp500)

    out_csv = args.out_dir / "data" / "margin_debt_market_cap.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.rename_axis("month").to_csv(out_csv, float_format="%.4f")
    print(f"saved {out_csv}")

    if not args.no_plot:
        png = args.out_dir / "images" / "margin_debt_market_cap.png"
        plot(df, png)
        print(f"saved {png}")

    for k, v in summarize(df).items():
        print(f"{k:>26}: {v}")


if __name__ == "__main__":
    main()
