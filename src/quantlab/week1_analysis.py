from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["SPY", "QQQ"]
START = "2020-01-01"
END = "2026-01-01" #Exclusive end date

def download_prices() -> pd.DataFrame:
    """Download dividend/split-adjusted daily close prices."""
    data = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False, threads=False)
    if data.empty:
        raise ValueError("No data downloaded. Check ticker symbols and date range.")

    prices = data["Close"][TICKERS].copy()
    prices = prices.dropna()

    if prices.empty:
        raise RuntimeError("No aligned price observations")

    return prices

def calculate_metrics(prices: pd.DataFrame):
    """Calculate returns and annualized volatility"""
    simple_returns = prices.pct_change(fill_method=None).dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    annual_volatility = simple_returns.std(ddof=1) * np.sqrt(252)  

    cumulative_returns = (1 + simple_returns).prod() - 1 

    metrics = pd.DataFrame({"Cumulative Return": cumulative_returns, "Annualized Volatility": annual_volatility,})
    return simple_returns, log_returns, metrics

def main():
    output = Path("reports")
    output.mkdir(exist_ok=True)

    prices = download_prices()
    simple_ret, log_ret, metrics = calculate_metrics(prices)

    normalized = prices / prices.iloc[0] * 100

    ax = normalized.plot(title="SPY vs QQQ - Normalized AdjustedPrices", figsize=(10, 6))

    ax.set_ylabel("Index (First Day = 100)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output / "week01_price.png", dpi=150)
    plt.close()

    metrics.to_csv(output / "week01_metrics.csv")

    print("First five daily returns:")
    print(simple_ret.head())
    print("\nFirst five log returns:")
    print(log_ret.head())
    print("\nPerformance summary:")
    print(metrics.round(4))

if __name__ == "__main__":
    main()
    
