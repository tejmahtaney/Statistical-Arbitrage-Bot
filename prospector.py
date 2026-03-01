# prospector.py
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import itertools
import time
import os
import requests
from datetime import date

# --- Configuration ---
START_DATE = '2023-01-01'
END_DATE = date.today().strftime('%Y-%m-%d') # Use current date for most recent data
P_VALUE_THRESHOLD = 0.0001 # Stricter threshold for better quality pairs
OUTPUT_FILE = 'cointegrated_pairs.csv'
MIN_MARKET_CAP = 2_000_000_000  # Mid cap and above

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
        print(f" Found {len(tickers)} Nasdaq tickers.")
        return tickers
    except Exception as e:
        print(f" Error fetching Nasdaq tickers: {e}")
        return []

def download_data(tickers, start, end):
    """Downloads historical closing prices."""
    print(f"Downloading historical data for {len(tickers)} tickers...")
    data = yf.download(tickers, start=start, end=end, progress=True)['Close']
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

def find_cointegrated_pairs(data, p_value_threshold):
    """Scans for cointegrated pairs and returns a DataFrame with their parameters."""
    num_assets = data.shape[1]
    asset_names = data.columns
    total_pairs = int(num_assets * (num_assets - 1) / 2)
    print(f"\nScanning {total_pairs} unique pairs for cointegration...")
    
    results = []
    start_scan_time = time.time()

    # Progress tracking
    pair_counter = 0
    progress_interval = max(1, total_pairs // 100)  # ~100 updates across full scan

    for i, j in itertools.combinations(range(num_assets), 2):
        pair_counter += 1
        stock1_name, stock2_name = asset_names[i], asset_names[j]
        stock1_prices = data[stock1_name]
        stock2_prices = data[stock2_name]
        
        # Calculate hedge ratio and spread from historical data
        model = sm.OLS(stock2_prices, sm.add_constant(stock1_prices)).fit()
        hedge_ratio = model.params.iloc[1]
        spread = stock2_prices - hedge_ratio * stock1_prices
        
        # Run ADF test for cointegration
        adf_result = adfuller(spread.dropna())
        p_value = adf_result[1]

        if p_value < p_value_threshold:
            # If cointegrated, save all necessary parameters for the trading bot
            results.append({
                'stock1': stock1_name,
                'stock2': stock2_name,
                'p_value': p_value,
                'hedge_ratio': hedge_ratio,
                'spread_mean': spread.mean(),
                'spread_std': spread.std()
            })

        # Print progress periodically
        if pair_counter % progress_interval == 0 or pair_counter == total_pairs:
            elapsed = time.time() - start_scan_time
            avg_per_pair = elapsed / pair_counter if pair_counter else 0
            remaining_pairs = total_pairs - pair_counter
            est_remaining = remaining_pairs * avg_per_pair
            pct_done = (pair_counter / total_pairs) * 100 if total_pairs else 100.0
            print(f"Progress: {pair_counter}/{total_pairs} ({pct_done:.1f}%) - "
                  f"Elapsed: {elapsed:.1f}s - Est remaining: {est_remaining:.1f}s")

    end_scan_time = time.time()
    print(f"Scan completed in {end_scan_time - start_scan_time:.2f} seconds.")
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='p_value', ascending=True)
    return results_df

if __name__ == "__main__":
    tickers = get_nasdaq_tickers()
    tickers = filter_by_market_cap(tickers, min_cap=MIN_MARKET_CAP)
    if tickers:
        prices_df = download_data(tickers, START_DATE, END_DATE)
        if not prices_df.empty and len(prices_df.columns) > 1:
            found_pairs_df = find_cointegrated_pairs(prices_df, P_VALUE_THRESHOLD)
            
            if not found_pairs_df.empty:
                top_1000_pairs = found_pairs_df.head(1000)

                print(f"\n---  Found {len(found_pairs_df)} Cointegrated Pairs ---")
                print(f"Saving the best {len(top_1000_pairs)} pairs to the output file.")
                print(top_1000_pairs.head())
                top_1000_pairs.to_csv(OUTPUT_FILE, index=False)
                print(f"\nResults saved to '{OUTPUT_FILE}'.")
            else:
                print("\nNo statistically significant pairs found.")