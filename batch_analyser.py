import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import itertools
import matplotlib.pyplot as plt
import time
import os
import requests

# --- Stage 0: Setup and Data Collection ---
MIN_MARKET_CAP = 2_000_000_000  # Mid cap and above

# --- CORRECTED FUNCTION ---
def get_nasdaq_tickers():
    """Fetches the list of Nasdaq-listed tickers from Nasdaq Trader."""
    print("Fetching Nasdaq ticker list...")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        }
        url = 'https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt'
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        lines = response.text.strip().splitlines()
        header = lines[0].split('|')
        rows = [line.split('|') for line in lines[1:] if 'File Creation Time' not in line]
        df = pd.DataFrame(rows, columns=header)

        # Filter out test issues and empty symbols
        df = df[(df['Test Issue'] == 'N') & (df['Symbol'].notna())]

        tickers = df['Symbol'].tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        print(f"✅ Found {len(tickers)} Nasdaq tickers.")
        return tickers
    except Exception as e:
        print(f"❌ Error fetching Nasdaq tickers: {e}")
        print("Falling back to a smaller, hardcoded list of stocks.")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'INTC', 'CSCO', 'ADBE', 'QCOM']
    
def download_data(tickers, start_date, end_date):
    """Downloads historical closing prices for a list of tickers."""
    print(f"Downloading historical data for {len(tickers)} tickers...")
    # FIX 2: Changed 'end_date' keyword to 'end'. This resolves the TypeError.
    data = yf.download(tickers, start=start_date, end=end_date, progress=True)['Close']
    cleaned_data = data.dropna(axis=1)
    print(f"Successfully loaded and cleaned data for {len(cleaned_data.columns)} tickers.")
    return cleaned_data

def filter_by_market_cap(tickers, min_cap=None, max_cap=None):
    """Filters tickers by market cap using yfinance."""
    if min_cap is None and max_cap is None:
        return tickers

    print("Filtering tickers by market cap...")
    kept = []
    missing_cap = 0
    total = len(tickers)

    for idx, ticker in enumerate(tickers, start=1):
        try:
            yf_ticker = yf.Ticker(ticker)
            cap = yf_ticker.fast_info.get('market_cap')
            if cap is None:
                cap = yf_ticker.info.get('marketCap')

            if cap is None:
                missing_cap += 1
                continue

            if min_cap is not None and cap < min_cap:
                continue
            if max_cap is not None and cap > max_cap:
                continue

            kept.append(ticker)
        except Exception:
            continue

        if idx % 100 == 0 or idx == total:
            print(f"  Checked {idx}/{total} tickers; kept {len(kept)}")

    print(f"Kept {len(kept)} tickers; skipped {missing_cap} with missing cap data.")
    return kept

# --- Stage 1: Prospecting for Cointegrated Pairs ---

def find_cointegrated_pairs(data, p_value_threshold=0.0005):
    """Scans a dataframe of prices to find all cointegrated pairs."""
    num_assets = data.shape[1]
    asset_names = data.columns
    total_pairs = int(num_assets * (num_assets - 1) / 2)
    print(f"\nScanning {total_pairs} unique pairs for cointegration (p < {p_value_threshold})...")
    
    cointegrated_pairs = []
    start_scan_time = time.time()

    for i, j in itertools.combinations(range(num_assets), 2):
        stock1_name, stock2_name = asset_names[i], asset_names[j]
        stock1, stock2 = data[stock1_name], data[stock2_name]
        
        model = sm.OLS(stock2, sm.add_constant(stock1)).fit()
        hedge_ratio = model.params.iloc[1]
        spread = stock2 - hedge_ratio * stock1
        
        adf_result = adfuller(spread.dropna())
        p_value = adf_result[1]

        if p_value < p_value_threshold:
            cointegrated_pairs.append((stock1_name, stock2_name, p_value, hedge_ratio))
    
    end_scan_time = time.time()
    print(f"Scan completed in {end_scan_time - start_scan_time:.2f} seconds.")
    cointegrated_pairs.sort(key=lambda x: x[2])
    return cointegrated_pairs

# --- Stage 2: Batch Evaluation and File Export ---

def analyze_and_save_pair(pair_data, prices_df, rank, output_dir):
    """Performs analysis for a pair and saves the chart to a file."""
    stock1_name, stock2_name, p_value, hedge_ratio = pair_data
    print(f"  Analyzing Pair #{rank}: ({stock1_name}, {stock2_name}) | P-Value: {p_value:.6f}")

    stock1 = prices_df[stock1_name]
    stock2 = prices_df[stock2_name]
    spread = stock2 - hedge_ratio * stock1
    z_score = (spread - spread.mean()) / spread.std()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f'Analysis for Pair: {stock1_name} vs {stock2_name} (p={p_value:.4f})', fontsize=16)

    ax1.plot(prices_df.index, stock1, label=stock1_name); ax1.plot(prices_df.index, stock2, label=stock2_name)
    ax1.set(ylabel='Price ($)', title='Asset Prices'); ax1.legend(); ax1.grid(True)

    ax2.plot(spread.index, spread, label='Spread'); ax2.axhline(spread.mean(), color='red', linestyle='--', label='Mean')
    ax2.set(ylabel='Spread Value', title='Cointegrated Spread'); ax2.legend(); ax2.grid(True)

    ax3.plot(z_score.index, z_score, label='Z-Score'); ax3.axhline(2.0, color='red', linestyle='--'); ax3.axhline(-2.0, color='green', linestyle='--'); ax3.axhline(0.0, color='black')
    ax3.set(ylabel='Z-Score', title='Spread Z-Score for Trading Signals'); ax3.legend(['Z-Score', 'Sell Signal (-2σ)', 'Buy Signal (+2σ)']); ax3.grid(True)
    
    plt.xlabel('Date'); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = f"Pair_Analysis_{stock1_name}-{stock2_name}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

# --- Main Execution ---

if __name__ == "__main__":
    # --- Parameters ---
    start_date = '2023-01-01'
    end_date = '2025-10-23'
    p_value_threshold = 0.0001 
    
    output_directory = "pair_analysis_charts"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    tickers = get_nasdaq_tickers()
    tickers = filter_by_market_cap(tickers, min_cap=MIN_MARKET_CAP)
    prices_df = download_data(tickers, start_date, end_date)
    
    if not prices_df.empty and len(prices_df.columns) > 1:
        found_pairs = find_cointegrated_pairs(prices_df, p_value_threshold)
        
        if not found_pairs:
            print("\nNo statistically significant cointegrated pairs found.")
        else:
            print(f"\n--- ✅ Found {len(found_pairs)} Cointegrated Pairs (p < {p_value_threshold}) ---")
            print(f"Generating and saving analysis charts to the '{output_directory}' folder...")
            for i, pair in enumerate(found_pairs):
                analyze_and_save_pair(pair, prices_df, rank=i+1, output_dir=output_directory)
            print("\n--- All analysis charts have been saved successfully. ---")