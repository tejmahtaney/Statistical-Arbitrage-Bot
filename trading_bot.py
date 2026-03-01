# trading_bot.py
import alpaca_trade_api as tradeapi
import pandas as pd
import time
import os
import math
import datetime

# --- Configuration ---
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
PAIRS_FILE = 'cointegrated_pairs.csv'
POSITIONS_FILE = 'open_positions.csv'

# --- Strategy Parameters ---
NUM_PAIRS_TO_TRADE = 500 # Number of top pairs to trade from the CSV
ENTRY_Z_SCORE = 2.0
EXIT_Z_SCORE = 0.5

# --- Trade Sizing ---
TRADE_CAPITAL_USD = 10000.00

class PairsTradingBot:
    def __init__(self, api_key, secret_key, base_url, auto_close_incomplete=None, run_once_when_closed=False):
        self.api = tradeapi.REST(api_key, secret_key, base_url=base_url, api_version='v2')
        # Auto-close policy: constructor param overrides env var.
        if auto_close_incomplete is None:
            env = os.getenv('BOT_AUTO_CLOSE_INCOMPLETE', 'false').lower()
            self.auto_close_incomplete = env in ('1', 'true', 'yes')
        else:
            self.auto_close_incomplete = bool(auto_close_incomplete)

        self.run_once_when_closed = bool(run_once_when_closed)
        self.trading_pairs = self.load_pairs_to_trade()
        if self.trading_pairs:
            self.close_positions_not_in_pairs_file()
            # self.initial_cleanup()  # Optional: close all managed positions on startup.
            self.sync_positions_on_startup()
            self.update_positions_file()

    def is_in_trade(self, pair):
        """Check if there is an open position for BOTH stocks in a specific pair."""
        try:
            self.api.get_position(pair['stock1'])
            self.api.get_position(pair['stock2'])
            return True
        except tradeapi.rest.APIError:
            return False

    def sync_positions_on_startup(self):
        """
        Checks Alpaca for existing positions on startup and syncs the bot's internal state.
        This allows the bot to resume managing trades that were open before a restart.
        """
        print("Syncing positions on startup...")
        positions = self.api.list_positions()
        position_symbols = {pos.symbol: pos for pos in positions}

        for pair in self.trading_pairs:
            stock1, stock2 = pair['stock1'], pair['stock2']
            
            has_pos1 = stock1 in position_symbols
            has_pos2 = stock2 in position_symbols

            if has_pos1 and has_pos2:
                print(f"Found existing position for pair {stock1}/{stock2}.")
                pos1_qty = float(position_symbols[stock1].qty)
                if pos1_qty > 0:
                    pair['position_side'] = 'short_spread' # Bought stock1
                elif pos1_qty < 0:
                    pair['position_side'] = 'long_spread' # Sold stock1
                
                print(f"  -> Synced as '{pair['position_side']}'")
            
            elif has_pos1 or has_pos2:
                msg = f"Warning: Found an incomplete position for pair {stock1}/{stock2}."
                if self.auto_close_incomplete:
                    print(msg + " Closing it now (BOT_AUTO_CLOSE_INCOMPLETE=True).")
                    self.close_pair_position(pair)
                else:
                    print(msg + " Leaving it in place. To enable automatic cleanup set the environment variable BOT_AUTO_CLOSE_INCOMPLETE=1 or pass auto_close_incomplete=True to the bot constructor.")

        print("Position sync complete.")

    def update_positions_file(self):
        """Writes a CSV snapshot of all open positions and sizing from Alpaca."""
        try:
            positions = self.api.list_positions()
        except Exception as e:
            print(f"Warning: Could not fetch positions for CSV snapshot: {e}")
            return

        rows = []
        for pos in positions:
            rows.append({
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'side': pos.side,
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
            })

        df = pd.DataFrame(rows, columns=[
            'symbol',
            'qty',
            'side',
            'avg_entry_price',
            'current_price',
            'market_value',
            'unrealized_pl',
            'unrealized_plpc',
        ])
        df.to_csv(POSITIONS_FILE, index=False)
        print(f"Updated positions snapshot: {POSITIONS_FILE}")

    def close_positions_not_in_pairs_file(self):
        """
        Closes any open positions whose symbols are not present in the current pairs CSV.
        This helps clean up trades from pairs that were removed from the file.
        """
        print("Reconciling positions against pairs CSV...")
        allowed_symbols = set()
        for pair in self.trading_pairs:
            allowed_symbols.add(pair['stock1'])
            allowed_symbols.add(pair['stock2'])

        positions = self.api.list_positions()
        for pos in positions:
            if pos.symbol not in allowed_symbols:
                print(f"Found position in {pos.symbol} not in pairs CSV. Closing it now.")
                try:
                    self.api.close_position(pos.symbol)
                    time.sleep(1)
                except tradeapi.rest.APIError:
                    pass
        print("Reconciliation complete.")
        self.update_positions_file()

    def initial_cleanup(self):
        """Cancel all open orders and close positions for all managed pairs to start fresh."""
        print("Performing initial cleanup...")
        self.api.cancel_all_orders()
        
        all_stocks = set()
        for pair in self.trading_pairs:
            all_stocks.add(pair['stock1'])
            all_stocks.add(pair['stock2'])

        for stock in all_stocks:
            try:
                self.api.close_position(stock)
                print(f"Closed existing position in {stock}.")
                time.sleep(1)
            except tradeapi.rest.APIError:
                pass
        
            time.sleep(2)
        print("Cleanup complete.")
        self.update_positions_file()

    def load_pairs_to_trade(self):
        """Loads the top N ranked pairs from the prospector's CSV file."""
        if not os.path.exists(PAIRS_FILE):
            print(f"Error: Pairs file '{PAIRS_FILE}' not found. Run prospector.py first.")
            return []
        
        pairs_df = pd.read_csv(PAIRS_FILE)
        if pairs_df.empty:
            print("Error: Pairs file is empty. No pairs to trade.")
            return []
            
        top_pairs = pairs_df.head(NUM_PAIRS_TO_TRADE).to_dict('records')
        
        for pair in top_pairs:
            pair['stock1'] = pair['stock1'].replace('-', '.')
            pair['stock2'] = pair['stock2'].replace('-', '.')

        symbol_counts = {}
        for pair in top_pairs:
            symbol_counts[pair['stock1']] = symbol_counts.get(pair['stock1'], 0) + 1
            symbol_counts[pair['stock2']] = symbol_counts.get(pair['stock2'], 0) + 1
        repeated_symbols = [sym for sym, count in symbol_counts.items() if count > 1]
        if repeated_symbols:
            print(f"Warning: Symbols appear in multiple pairs: {', '.join(repeated_symbols)}")

        print("--- Trading Bot Initialized ---")
        print(f"Loaded {len(top_pairs)} pairs to trade.")
        print(f"Trade Capital per Leg: ${TRADE_CAPITAL_USD:.2f}")
        print("-------------------------------")

        for pair in top_pairs:
            pair['position_side'] = None
        
        return top_pairs

    def place_pairs_trade(self, pair, side, price1, price2):
        """Calculates quantities and places the two orders for a given pair."""
        stock1, stock2, hedge_ratio = pair['stock1'], pair['stock2'], pair['hedge_ratio']
        print(f"\nPlacing trade for {stock1}/{stock2}, side: {side}")
        try:
            qty1 = math.floor(TRADE_CAPITAL_USD / price1)
            qty2 = math.floor((qty1 * hedge_ratio * price1) / price2)

            if side == 'long_spread':
                side1, side2 = 'sell', 'buy'
            else:
                side1, side2 = 'buy', 'sell'

            print(f"Submitting {side1} order for {qty1} shares of {stock1}.")
            self.api.submit_order(symbol=stock1, qty=qty1, side=side1, type='market', time_in_force='day')
            
            print(f"Submitting {side2} order for {qty2} shares of {stock2}.")
            self.api.submit_order(symbol=stock2, qty=qty2, side=side2, type='market', time_in_force='day')
            
            pair['position_side'] = side
            print(f"---  Trade Orders Submitted for {stock1}/{stock2} ---")
            self.update_positions_file()

        except Exception as e:
            print(f"---  Error placing trade for {stock1}/{stock2}: {e} ---")
            self.close_pair_position(pair)

    def close_pair_position(self, pair):
        """Closes any open positions for a specific pair."""
        stock1, stock2 = pair['stock1'], pair['stock2']
        print(f"\n--- Closing all positions for pair {stock1}/{stock2} ---")
        try:
            self.api.close_position(stock1)
        except tradeapi.rest.APIError:
            pass
        try:
            self.api.close_position(stock2)
        except tradeapi.rest.APIError:
            pass
        
        pair['position_side'] = None
        print(f"---  Positions Closed for {stock1}/{stock2} ---")
        self.update_positions_file()

    def check_pair_signal(self, pair):
        """Fetches live prices and checks for trading signals for a single pair."""
        stock1, stock2 = pair['stock1'], pair['stock2']
        hedge_ratio, spread_mean, spread_std = pair['hedge_ratio'], pair['spread_mean'], pair['spread_std']

        try:
            price1 = self.api.get_latest_trade(stock1).price
            price2 = self.api.get_latest_trade(stock2).price
        except Exception as e:
            print(f"Warning: Could not fetch prices for {stock1}/{stock2}. Skipping. Error: {e}")
            return
            
        current_spread = price2 - hedge_ratio * price1
        current_z_score = (current_spread - spread_mean) / spread_std
        
        in_trade = self.is_in_trade(pair)

        # Re-sync position side if bot restarted while in a trade.
        if in_trade and not pair['position_side']:
            try:
                pos1_qty = float(self.api.get_position(stock1).qty)
                if pos1_qty > 0: pair['position_side'] = 'short_spread' # Bought stock1
                elif pos1_qty < 0: pair['position_side'] = 'long_spread' # Sold stock1
            except Exception as e:
                print(f"Could not determine position side for {stock1}/{stock2} from API: {e}")

        print(f"[{time.strftime('%H:%M:%S')}] {stock1}/{stock2} | Z: {current_z_score:+.4f} | In Trade: {in_trade} ({pair['position_side'] or 'None'})")

        if not in_trade:
            if current_z_score > ENTRY_Z_SCORE:
                print(f"ENTRY SIGNAL: {stock1}/{stock2} Z-Score > {ENTRY_Z_SCORE}. Shorting spread.")
                self.place_pairs_trade(pair, 'short_spread', price1, price2)
            elif current_z_score < -ENTRY_Z_SCORE:
                print(f"ENTRY SIGNAL: {stock1}/{stock2} Z-Score < -{ENTRY_Z_SCORE}. Longing spread.")
                self.place_pairs_trade(pair, 'long_spread', price1, price2)
        else:
            if pair['position_side'] == 'short_spread' and current_z_score < EXIT_Z_SCORE:
                print(f"EXIT SIGNAL: {stock1}/{stock2} Z-Score crossed below {EXIT_Z_SCORE}. Closing short spread.")
                self.close_pair_position(pair)
            elif pair['position_side'] == 'long_spread' and current_z_score > -EXIT_Z_SCORE:
                print(f"EXIT SIGNAL: {stock1}/{stock2} Z-Score crossed above -{EXIT_Z_SCORE}. Closing long spread.")
                self.close_pair_position(pair)

    def run(self):
        """Main loop for the trading bot; run checks only while market is open using Alpaca's clock."""
        if not self.trading_pairs:
            print("No pairs to trade. Exiting.")
            return

        if not API_KEY or not SECRET_KEY:
            print("Error: ALPACA_API_KEY or ALPACA_SECRET_KEY not set in environment. Exiting.")
            return

        print("\nBot is running. Press Ctrl+C to stop.")
        closed_market_pass_done = False
        try:
            while True:
                clock = self.api.get_clock()
                if clock.is_open:
                    print(f"Market is OPEN. Next close: {clock.next_close}")
                    while self.api.get_clock().is_open:
                        cycle_start = time.time()
                        print(f"\n--- New Check Cycle at {time.strftime('%H:%M:%S')} ---")
                        for pair in self.trading_pairs:
                            self.check_pair_signal(pair)
                            time.sleep(2)
                        self.update_positions_file()
                        elapsed = time.time() - cycle_start
                        sleep_for = max(0, 60 - elapsed)
                        print(f"\n--- Cycle Complete. Waiting {sleep_for:.0f} seconds. ---")
                        time.sleep(sleep_for)
                    print("Market closed. Waiting for next open...")
                else:
                    if self.run_once_when_closed and not closed_market_pass_done:
                        print("Market is CLOSED. Running a single check cycle by user request.")
                        cycle_start = time.time()
                        print(f"\n--- New Check Cycle at {time.strftime('%H:%M:%S')} ---")
                        for pair in self.trading_pairs:
                            self.check_pair_signal(pair)
                            time.sleep(2)
                        elapsed = time.time() - cycle_start
                        print(f"\n--- Cycle Complete in {elapsed:.0f} seconds. ---")
                        closed_market_pass_done = True

                    next_open = clock.next_open
                    now = datetime.datetime.now(datetime.timezone.utc)
                    wait_seconds = (next_open - now).total_seconds()
                    wait_seconds = max(wait_seconds, 60)
                    hours = wait_seconds / 3600.0
                    print(f"Market closed. Sleeping until next open at {next_open} (~{hours:.2f} hours).")
                    time.sleep(wait_seconds)
        except KeyboardInterrupt:
            print("Stopping bot (KeyboardInterrupt).")

def prompt_yes_no(prompt_text, default_no=True):
    suffix = " [Y/N]: " if default_no else " [Y/N]: "
    response = input(prompt_text + suffix).strip().lower()
    if not response:
        return not default_no
    return response in ["y", "Y", "yes"]


if __name__ == "__main__":
    run_once = os.getenv('BOT_RUN_ONCE_WHEN_CLOSED', 'false').lower() in ('1', 'true', 'yes')
    bot = PairsTradingBot(API_KEY, SECRET_KEY, BASE_URL, run_once_when_closed=run_once)
    bot.run()