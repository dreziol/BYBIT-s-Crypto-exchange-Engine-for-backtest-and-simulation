import pandas as pd
import json
import logging
import os
import re
import html as html_lib
from collections import defaultdict
from typing import Dict, List, Any
import datetime
import glob
import sys
import bisect
import numpy as np
import ccxt
import io
from functools import lru_cache

@lru_cache(maxsize=1024)
def normalize_symbol_ccxt(symbol: str) -> str:
    if not symbol:
        return ""
    clean = symbol.split(':')[0]
    if "/" not in clean and clean.endswith("USDT"):
        base = clean[:-4]
        return f"{base}/USDT:USDT"
    if "/" not in clean:
        return f"{clean}/USDT:USDT"
    if ":" not in symbol:
        return f"{symbol}:USDT"
    return symbol

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
try:
    _stdout = sys.stdout
    if getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(getattr(sys.stdout, "buffer", sys.stdout), encoding="utf-8", errors="replace")
except Exception:
    pass

# Terminal UI Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_MAGENTA = "\033[95m"
C_CYAN = "\033[96m"
C_WHITE = "\033[97m"

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# გამორთეთ ზედმეტი ლოგები სისწრაფისთვის
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(message)s")

class MockCCXTBybit:
    # Bybit USDT linear hedge (approximate) — mark-to-market, not entry price
    LEVERAGE_DEFAULT = 10.0
    MAINTENANCE_MARGIN_RATE = 0.005  # 0.5% of max leg notional (per symbol)
    MAKER_FEE_RATE = 0.0002
    TAKER_FEE_RATE = 0.00055

    def __init__(self, config_path: str, csv_dir: str = "."):
        self.start_balance = 10000.0
        self.balance = {"USDT": {"free": 10000.0, "used": 0.0, "total": 10000.0}}
        self.leverage = self.LEVERAGE_DEFAULT
        # Bybit Hedge Mode იზოლაცია: LONG და SHORT ცალ-ცალკეა
        self.positions = defaultdict(lambda: {
            "LONG": {"size": 0.0, "entryPrice": 0.0, "unrealizedPnl": 0.0, "leverage": 10},
            "SHORT": {"size": 0.0, "entryPrice": 0.0, "unrealizedPnl": 0.0, "leverage": 10}
        })
        
        self.trade_log = []
        self.equity_curve = []
        self.global_step = 0
        self.current_timestamp_ms = 0
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.verbose_trade_console = bool(self.config.get("verbose_trade_console", False))
        self.is_hedge_mode = str(self.config.get("position_mode", "hedge")).lower() == "hedge"
        self.pairs = self.config.get("multi_bot", {}).get("pairs", [])
        self._symbols_list = [f"{p}/USDT:USDT" if "USDT" not in p else p for p in self.pairs]
        
        self.open_orders = {}
        self.active_open_orders = {}  # სწრაფი კოლექცია მხოლოდ აქტიური ღია ორდერებისთვის (მეხსიერების და CPU ოპტიმიზაცია)
        self.orders_by_symbol = defaultdict(list)
        self.order_counter = 0
        self.options = {"defaultType": "linear", "adjustForTimeDifference": True}
        
        self.fast_times: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
        self.fast_ohlcv_arrays: Dict[str, Dict[str, dict]] = defaultdict(dict)
        self.fast_indicators: Dict[str, Dict[str, Dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
        
        self.ohlcv_cache = {}
        self.price_cache = {}
        self.indicator_cache = {}
        self.audit_buffer = []  # RAM-based bulk logging
        self.audit_flush_threshold = int(self.config.get("audit_flush_threshold", 1000) or 1000)
        self._audit_initialized = True
        try:
            open("full_audit_log.jsonl", "w", encoding="utf-8").close()
        except Exception:
            pass
        self.total_fees_paid = 0.0
        self.symbol_fees = defaultdict(float) # ტრეკინგი პაირების მიხედვით
        self.total_volume = 0.0
        self.bot_perceived_pnl = 0.0
        self.balance_cache = None
        self.unrealized_pnl_cache = None
        self.realized_pnl_by_symbol = defaultdict(float) # Net realized (pnl - fees)
        self.gross_pnl_by_symbol = defaultdict(float)    # Gross realized (pnl only)
        self.tf_indices = defaultdict(lambda: defaultdict(int)) # Symbol -> TF -> Current Index
        self.needs_sync = True
        
        self.max_idx = 0
        self._load_and_resample_csvs(csv_dir)
        
        if hasattr(self, "master_times") and len(self.master_times) > 0:
            self.global_step = 0
            self.current_timestamp_ms = int(self.master_times[0])

    @property
    def symbols(self):
        return self._symbols_list

    def _normalize_to_ccxt(self, symbol: str) -> str:
        return normalize_symbol_ccxt(symbol)

    def _to_v5_symbol(self, ccxt_symbol: str) -> str:
        ccxt_norm = self._normalize_to_ccxt(ccxt_symbol)
        return ccxt_norm.split(':')[0].replace('/', '')

    def _calculate_rsi(self, series, period=14):
        from ta.momentum import RSIIndicator
        rsi_series = RSIIndicator(close=series, window=period).rsi()
        
        diff = series.diff().values
        gains = np.where(diff > 0, diff, 0.0)
        losses = np.where(diff < 0, -diff, 0.0)
        
        gains[0] = 0.0
        losses[0] = 0.0
        
        # Use pandas ewm with alpha=1/period to match the exact smoothed metrics calculated in TA library
        avg_gain = pd.Series(gains, index=series.index).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = pd.Series(losses, index=series.index).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        
        return (
            rsi_series,
            avg_gain,
            avg_loss
        )

    def _csv_line_count(self, path: str) -> int:
        try:
            with open(path, "rb") as fh:
                return max(0, sum(1 for _ in fh) - 1)
        except OSError:
            return 0

    def _select_csv_file(self, files: list, base_coin: str) -> str:
        """
        Pick history CSV like a human would: prefer long-history names (1000d/365d),
        then most rows. Never prefer a larger 7d slice over a proper history file.
        """
        if len(files) == 1:
            if self._csv_line_count(files[0]) < 10:
                print(
                    f"❌ {base_coin}: REJECTED {os.path.basename(files[0])} — "
                    f"only {self._csv_line_count(files[0])} rows. Use full history CSV."
                )
                return None
            return files[0]

        def name_rank(path: str) -> int:
            name = os.path.basename(path).lower()
            if "1000d" in name or "1000_d" in name:
                return 4
            if "365d" in name or "365_d" in name:
                return 3
            if "100d" in name:
                return 2
            if "7d" in name:
                return 0
            return 1

        ranked = []
        for path in files:
            rows = self._csv_line_count(path)
            ranked.append((name_rank(path), rows, path))

        ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        selected = ranked[0][2]
        best_rows = ranked[0][1]

        if best_rows < 10:
            print(
                f"❌ {base_coin}: REJECTED {os.path.basename(selected)} — only {best_rows} rows. "
                f"Use *_1000d.csv with 500k+ rows."
            )
            return None

        if ranked[0][0] == 0:
            print(
                f"⚠️ {base_coin}: only short-window CSV candidates found; "
                f"using {os.path.basename(selected)} ({best_rows} rows)."
            )
        skipped = [os.path.basename(p) for _, _, p in ranked[1:3]]
        if skipped:
            print(f"   ↳ skipped: {', '.join(skipped)}")

        return selected

    def _generate_synthetic_csv_files(self, csv_dir):
        print("🛠️ Generating high-quality synthetic market data for backtesting...")
        import os
        import pandas as pd
        import numpy as np
        
        os.makedirs(csv_dir, exist_ok=True)
        
        starting_prices = {
            "DOGE": 0.15,
            "BNB": 580.0,
            "DOT": 6.5,
            "NEAR": 7.2,
            "LINK": 15.0,
            "ADA": 0.45,
            "AVAX": 35.0,
            "LTC": 82.0,
            "XLM": 0.11,
            "XRP": 0.52
        }
        
        num_minutes = 15000
        start_time = pd.Timestamp.now() - pd.Timedelta(minutes=num_minutes)
        timestamps = [int((start_time + pd.Timedelta(minutes=i)).value // 10**6) for i in range(num_minutes)]
        
        for coin, start_price in starting_prices.items():
            filename = f"{csv_dir}/{coin}_USDT_1m_synthetic.csv"
            np.random.seed(42 + hash(coin) % 1000)
            returns = np.random.normal(loc=0.00001, scale=0.0015, size=num_minutes)
            cycles = 0.005 * np.sin(np.linspace(0, 15 * np.pi, num_minutes))
            price_multipliers = np.exp(np.cumsum(returns + cycles))
            prices = start_price * price_multipliers
            
            opens = prices * (1 + np.random.uniform(-0.0005, 0.0005, size=num_minutes))
            closes = prices
            highs = np.maximum(opens, closes) * (1 + np.random.uniform(0.0, 0.001, size=num_minutes))
            lows = np.minimum(opens, closes) * (1 - np.random.uniform(0.0, 0.001, size=num_minutes))
            volumes = np.random.exponential(scale=10000.0, size=num_minutes)
            
            df = pd.DataFrame({
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes
            })
            df.to_csv(filename, index=False)
            print(f"   ✅ Created: {filename} ({num_minutes} rows)")

    def _load_and_resample_csvs(self, csv_dir):
        # Timeframes რასაც ბოტი იყენებს
        tf_map = {
            "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", 
            "30m": "30min", "1h": "1h", "2h": "2h", "4h": "4h", 
            "12h": "12h", "1d": "1D"
        }
        
        rsi_len = self.config.get("rsi_length", 14)
        ema_len = self.config.get("ema_trend_period", 50)

        self.fast_ohlcv_lists = defaultdict(lambda: defaultdict(list))

        # More robust CSV searching
        print(f"🔍 Searching for CSV files in: {os.path.abspath(csv_dir)}")
        all_csvs = glob.glob(f"{csv_dir}/*.csv")
        if not all_csvs:
            print(f"⚠️ No CSV files found in {os.path.abspath(csv_dir)}")
            print("Current directory contents:", os.listdir(csv_dir))
            self._generate_synthetic_csv_files(csv_dir)
            all_csvs = glob.glob(f"{csv_dir}/*.csv")

        for symbol in self.symbols:
            base_coin = symbol.split('/')[0]
            # Try multiple common patterns
            patterns = [
                f"{csv_dir}/{base_coin}_USDT*1m*.csv",
                f"{csv_dir}/{base_coin}USDT*1m*.csv",
                f"{csv_dir}/{base_coin}_USDT*.csv",
                f"{csv_dir}/{base_coin}USDT*.csv",
                f"{csv_dir}/{base_coin}_*.csv",
                f"{csv_dir}/{base_coin}*.csv"
            ]
            
            files = []
            for p in patterns:
                files = glob.glob(p)
                if files: break
            
            if not files:
                print(f"❌ Could not find CSV for {symbol}")
                continue

            selected_file = self._select_csv_file(files, base_coin)
            if not selected_file:
                print(f"❌ Skipping {symbol} — no valid long-history CSV.")
                continue

            print(f"⏳ იტვირთება და მუშავდება {symbol} ({selected_file})...")
            try:
                df = pd.read_csv(selected_file)
            except Exception as e:
                print(f"❌ Error reading {selected_file}: {e}")
                continue
            # Handle timestamps more intelligently
            first_val = df['timestamp'].iloc[0]
            if isinstance(first_val, (int, np.integer, float, np.floating)) or str(first_val).isdigit():
                fv = float(first_val)
                if fv < 10**11: # Looks like seconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                elif fv < 10**14: # Looks like milliseconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                else: # nanoseconds or other
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Resample base df to a perfectly uniform, continuous 1min intervals first to avoid any gaps!
            df = df.resample('1min').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            })
            df['close'] = df['close'].ffill()
            df['open'] = df['open'].fillna(df['close'])
            df['high'] = df['high'].fillna(df['close'])
            df['low'] = df['low'].fillna(df['close'])
            df['volume'] = df['volume'].fillna(0.0)

            self.max_idx = max(self.max_idx, len(df))
            
            for tf_str, resample_str in tf_map.items():
                if tf_str == "1m":
                    resampled = df.copy()
                else:
                    resampled = df.resample(resample_str, label='left', closed='left').agg({
                        'open': 'first', 'high': 'max', 'low': 'min',
                        'close': 'last', 'volume': 'sum'
                    })
                    # Forward-fill closing prices
                    resampled['close'] = resampled['close'].ffill()
                    resampled['open'] = resampled['open'].fillna(resampled['close'])
                    resampled['high'] = resampled['high'].fillna(resampled['close'])
                    resampled['low'] = resampled['low'].fillna(resampled['close'])
                    resampled['volume'] = resampled['volume'].fillna(0.0)
                
                resampled.dropna(subset=['close'], inplace=True)
                
                resampled['rsi'], resampled['rsi_gain'], resampled['rsi_loss'] = self._calculate_rsi(resampled['close'], rsi_len)
                resampled['ema'] = resampled['close'].ewm(span=ema_len, adjust=False).mean()
                
                # Keep warmup NaNs. Backfilling would leak future indicator values
                # into the first candles and can create fake early reset/signal events.
                
                t_arr = resampled.index.values.astype('datetime64[ms]').astype(np.int64)
                self.fast_times[symbol][tf_str] = t_arr
                
                # OHLCV arrays for ultra-fast access
                self.fast_ohlcv_arrays[symbol][tf_str] = {
                    'o': resampled['open'].values.astype(np.float64),
                    'c': resampled['close'].values.astype(np.float64),
                    'h': resampled['high'].values.astype(np.float64),
                    'l': resampled['low'].values.astype(np.float64),
                    'v': resampled['volume'].values.astype(np.float64)
                }

                self.fast_indicators[symbol][tf_str] = {
                    'rsi': resampled['rsi'].values.astype(np.float64),
                    'rsi_gain': resampled['rsi_gain'].values.astype(np.float64),
                    'rsi_loss': resampled['rsi_loss'].values.astype(np.float64),
                    'ema': resampled['ema'].values.astype(np.float64)
                }
                
                # Print pre-processing logs for all active timeframes so that hybrids are visible to the user
                print(f"   ⚡ Pre-processing {symbol} {tf_str} data...")
                
                # Do NOT materialize full OHLCV as Python list-of-lists here.
                # With 1000d 1m CSVs this can allocate millions of Python objects per symbol
                # and the process may silently die during preprocessing. fetch_ohlcv() now
                # builds only the requested slice from the numpy arrays on demand.
                
        # After the loop - Identify global common timeline
        all_starts = []
        all_ends = []
        for s in self.symbols:
            if "1m" in self.fast_times[s]:
                all_starts.append(self.fast_times[s]["1m"][0])
                all_ends.append(self.fast_times[s]["1m"][-1])
        
        if not all_starts:
            print("❌ შეცდომა: მონაცემები ვერ ჩაიტვირთა!")
            sys.exit(1)
            
        global_start = min(all_starts)
        global_end = max(all_ends)
        
        # Create a master 1m timeline (minutes from start to end)
        self.master_times = np.arange(global_start, global_end + 60000, 60000, dtype=np.int64)
        self.max_idx = len(self.master_times)
        
        # Precompute timeframe index offsets for each symbol and timeframe
        # This speeds up the tick loop by ~10[x] and solves timeline differences elegantly!
        self.tf_indices_precomputed = defaultdict(dict)
        
        # Duration map in milliseconds for standard text timeframes
        tf_durations = {
            "1m": 60000, "3m": 180000, "5m": 300000, "15m": 900000,
            "30m": 1800000, "1h": 3600000, "2h": 7200000, "4h": 14400000,
            "12h": 43200000, "1d": 86400000
        }
        
        for symbol in self.symbols:
            if symbol not in self.fast_times:
                continue
            for tf_str, t_arr in self.fast_times[symbol].items():
                duration_ms = tf_durations.get(tf_str, 60000)
                
                # A candle (t_arr is open_time) is only COMPLETE and mathematically available 
                # after its duration has passed: current_time >= open_time + duration.
                # So we search for master_time in the array of CLOSE times!
                close_times = t_arr + duration_ms
                
                # For each timestamp in self.master_times, find the index of the latest bar
                indices = np.searchsorted(close_times, self.master_times, side='right') - 1
                indices = np.clip(indices, -1, len(t_arr) - 1).astype(np.int32)
                self.tf_indices_precomputed[symbol][tf_str] = indices
        
        start_dt = pd.to_datetime(global_start, unit='ms')
        end_dt = pd.to_datetime(global_end, unit='ms')
        print(f"📅 ბექტიესტის პერიოდი: {start_dt} -დან {end_dt} -მდე")
        print(f"✅ მონაცემების ჩატვირთვა და ოპტიმიზაცია დასრულდა! ({self.max_idx} ნაბიჯი)")
        print(
            "🔧 Engine: limit-fill float OHLC + LTC data-end order cancel + backtest_daily.log",
            flush=True,
        )
        try:
            open("backtest_daily.log", "w", encoding="utf-8").write(
                f"# backtest started {start_dt} -> {end_dt} | steps={self.max_idx}\n"
            )
        except Exception:
            pass

    def _get_tf_idx(self, symbol, timeframe) -> int:
        symbol = self._normalize_to_ccxt(symbol)
        try:
            return int(self.tf_indices_precomputed[symbol][timeframe][self.global_step])
        except KeyError:
            return 0

    def _get_1m_candle_ohlc(self, symbol: str, idx: int):
        """Return (open, high, low) as Python floats for limit-fill checks."""
        symbol = self._normalize_to_ccxt(symbol)
        try:
            i = int(idx)
            if i < 0:
                return None
            arr = self.fast_ohlcv_arrays[symbol]["1m"]
            return (
                float(arr["o"][i]),
                float(arr["h"][i]),
                float(arr["l"][i]),
            )
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    def _cancel_resting_orders_for_symbol(self, symbol: str) -> int:
        """Cancel open limits for a symbol (e.g. when CSV data ends). Returns count canceled."""
        symbol = self._normalize_to_ccxt(symbol)
        to_remove = [
            oid for oid, o in self.active_open_orders.items() if o.get("symbol") == symbol
        ]
        for oid in to_remove:
            o = self.active_open_orders.pop(oid, None)
            if o:
                o["status"] = "canceled"
            if oid in self.open_orders:
                self.open_orders[oid]["status"] = "canceled"
        if to_remove:
            self.needs_sync = True
        return len(to_remove)

    def is_symbol_active(self, symbol) -> bool:
        symbol = self._normalize_to_ccxt(symbol)
        if symbol not in self.fast_times or "1m" not in self.fast_times[symbol]:
            return False
        times = self.fast_times[symbol]["1m"]
        if len(times) == 0:
            return False
        return times[0] <= self.current_timestamp_ms <= times[-1] + 60000

    def get_accounting_snapshot(self) -> dict:
        """Wallet/equity/unrealized + engine trade_log realized + bot ledger state."""
        wallet = float(self.balance["USDT"]["total"])
        unrealized = float(self._update_pnl())
        open_notional = 0.0
        open_legs = 0
        exposures = {}
        for symbol, pos_dict in self.positions.items():
            price = self.get_latest_price(symbol) or 0.0
            ln = pos_dict["LONG"]["size"] * price
            sn = pos_dict["SHORT"]["size"] * price
            if ln > 0:
                open_legs += 1
                open_notional += ln
            if sn > 0:
                open_legs += 1
                open_notional += sn
            if ln > 0 or sn > 0:
                exposures[symbol] = {"long": ln, "short": sn, "total": ln + sn}

        eng_pnl = sum(float(t.get("pnl", 0) or 0) for t in self.trade_log)
        bot_p = getattr(self, "bot_perceived_pnl", 0.0)
        return {
            "wallet": wallet,
            "unrealized": unrealized,
            "equity": wallet + unrealized,
            "wallet_realized": wallet - float(self.start_balance),
            "engine_trade_log_pnl": eng_pnl,
            "sync_drift": bot_p - (wallet - float(self.start_balance)),
            "open_notional": open_notional,
            "open_legs": open_legs,
            "trade_closes": len(self.trade_log),
            "bot_perceived_pnl": bot_p,
            "exposures": exposures
        }

    def _mark_price(self, symbol: str) -> float:
        px = self.get_latest_price(symbol)
        if px is None or px <= 0:
            return 0.0
        return float(px)

    def _calc_unrealized_pnl(self, symbol: str) -> float:
        symbol = self._normalize_to_ccxt(symbol)
        pos_dict = self.positions.get(symbol)
        if not pos_dict:
            return 0.0
        current_price = self.get_latest_price(symbol)
        if current_price is None or current_price <= 0:
            return 0.0
        sym_pnl = 0.0
        for side, pos in pos_dict.items():
            if pos["size"] > 0:
                pnl = (current_price - pos["entryPrice"]) * pos["size"] if side == "LONG" else (pos["entryPrice"] - current_price) * pos["size"]
                sym_pnl += pnl
        return sym_pnl

    def _leg_notional_mark(self, symbol: str, side_key: str) -> float:
        """Hedge leg notional at mark (Bybit PM uses mark, not entry)."""
        pos = self.positions[symbol][side_key]
        if pos["size"] <= 0:
            return 0.0
        px = self._mark_price(symbol) or float(pos.get("entryPrice") or 0.0)
        return pos["size"] * px

    def _symbol_hedge_notional(self, symbol: str, extra_long_qty: float = 0.0, extra_short_qty: float = 0.0, price: float = None) -> tuple:
        """
        Per-symbol hedge netting: IM is based on max(LONG, SHORT) notional, not sum.
        Returns (long_notional, short_notional, max_leg_notional).
        """
        px = price if price and price > 0 else self._mark_price(symbol)
        if px <= 0:
            return 0.0, 0.0, 0.0
        long_n = self._leg_notional_mark(symbol, "LONG") + extra_long_qty * px
        short_n = self._leg_notional_mark(symbol, "SHORT") + extra_short_qty * px
        return long_n, short_n, max(long_n, short_n)

    def _coerce_position_idx(self, params) -> int:
        if not params:
            return None
        pos_idx_raw = params.get("positionIdx")
        if pos_idx_raw is None:
            return None
        try:
            return int(pos_idx_raw)
        except (ValueError, TypeError):
            return pos_idx_raw

    def _order_position_context(self, symbol: str, side: str, params=None) -> dict:
        """
        Single source of truth for order direction.
        Opening/increasing orders may consume margin; reduce/TP/closing orders never do.
        """
        symbol = self._normalize_to_ccxt(symbol)
        params = params or {}
        side_lower = str(side).lower()
        reduce_only = bool(params.get("reduceOnly", False))

        if not self.is_hedge_mode:
            # One-Way Mode Context Logic
            pos_dict = self.positions.get(symbol, {})
            long_pos = pos_dict.get("LONG", {})
            short_pos = pos_dict.get("SHORT", {})
            long_sz = float(long_pos.get("size", 0.0) or 0.0)
            short_sz = float(short_pos.get("size", 0.0) or 0.0)

            if side_lower == "sell":
                if long_sz > 0:
                    pos_key = "LONG"
                    closes_position = True
                else:
                    pos_key = "SHORT"
                    closes_position = False
            else:  # side_lower == "buy"
                if short_sz > 0:
                    pos_key = "SHORT"
                    closes_position = True
                else:
                    pos_key = "LONG"
                    closes_position = False
            
            is_reduce = reduce_only or closes_position
            return {
                "pos_idx": 0,
                "pos_key": pos_key,
                "side_lower": side_lower,
                "reduce_only": reduce_only,
                "closes_position": closes_position,
                "is_reduce": is_reduce,
            }
        else:
            # Original Hedge Mode Logic
            pos_idx = self._coerce_position_idx(params)
            if pos_idx == 1:
                pos_key = "LONG"
            elif pos_idx == 2:
                pos_key = "SHORT"
            elif reduce_only:
                pos_key = "SHORT" if side_lower == "buy" else "LONG"
            else:
                pos_dict = self.positions.get(symbol, {})
                long_pos = pos_dict.get("LONG", {})
                short_pos = pos_dict.get("SHORT", {})
                if side_lower == "sell" and float(long_pos.get("size", 0.0) or 0.0) > 0:
                    pos_key = "LONG"
                elif side_lower == "buy" and float(short_pos.get("size", 0.0) or 0.0) > 0:
                    pos_key = "SHORT"
                else:
                    pos_key = "LONG" if side_lower == "buy" else "SHORT"

            closes_position = (pos_key == "LONG" and side_lower == "sell") or (pos_key == "SHORT" and side_lower == "buy")
            return {
                "pos_idx": pos_idx,
                "pos_key": pos_key,
                "side_lower": side_lower,
                "reduce_only": reduce_only,
                "closes_position": closes_position,
                "is_reduce": reduce_only or closes_position,
            }

    def _amount_dust_threshold(self, symbol: str) -> float:
        """Smallest meaningful amount for this simulated market.

        Bybit/CCXT will not keep an executable remainder below market min qty.
        Treating that residue as open creates ghost ladders in the backtest.
        """
        try:
            min_amount = float(self.market(symbol).get("limits", {}).get("amount", {}).get("min") or 0.0)
        except Exception:
            min_amount = 0.0
        return max(1e-12, min_amount)

    def _finalize_order_fill_status(self, order: dict, symbol: str, exec_price: float):
        remaining = max(0.0, float(order["amount"]) - float(order["filled"]))
        if remaining <= self._amount_dust_threshold(symbol):
            order["filled"] = float(order["amount"])
            order["remaining"] = 0.0
            order["status"] = "closed"
        else:
            order["remaining"] = remaining
            order["status"] = "open"
        order["average"] = exec_price
        order["lastTradeTimestamp"] = self.current_timestamp_ms

    def _total_initial_margin_required(self, include_open_orders: bool = True) -> float:
        """Sum per-symbol max(LONG, SHORT) / leverage, matching cross-hedge netting."""
        used = 0.0
        active_orders_by_symbol = defaultdict(list)
        if include_open_orders:
            for order in self.active_open_orders.values():
                active_orders_by_symbol[order.get("symbol")].append(order)

        for symbol in self.symbols:
            pos_dict = self.positions[symbol]
            long_n = self._leg_notional_mark(symbol, "LONG")
            short_n = self._leg_notional_mark(symbol, "SHORT")
            if include_open_orders:
                px = self._mark_price(symbol)
                if px > 0:
                    open_long_val = 0.0
                    open_short_val = 0.0
                    for o in active_orders_by_symbol.get(symbol, ()):
                        ctx = self._order_position_context(symbol, o.get("side"), o.get("params", {}) or {})
                        if ctx["is_reduce"]:
                            continue
                        o_price = o.setdefault("price", px)
                        if o_price is None:
                            o_price = px
                        if ctx["pos_key"] == "LONG":
                            open_long_val += o["remaining"] * o_price
                        elif ctx["pos_key"] == "SHORT":
                            open_short_val += o["remaining"] * o_price
                    long_n += open_long_val
                    short_n += open_short_val
            used += max(long_n, short_n) / self.leverage
        return used

    def _margin_breakdown(self) -> dict:
        """Detailed margin view for diagnosing cross-hedge rejects."""
        rows = []
        position_margin = 0.0
        open_order_margin = 0.0
        active_orders_by_symbol = defaultdict(list)
        for order in self.active_open_orders.values():
            active_orders_by_symbol[order.get("symbol")].append(order)

        for symbol in self.symbols:
            base_long = self._leg_notional_mark(symbol, "LONG")
            base_short = self._leg_notional_mark(symbol, "SHORT")
            open_long = 0.0
            open_short = 0.0
            reduce_orders = 0
            opening_orders = 0

            px = self._mark_price(symbol)
            if px > 0:
                for o in active_orders_by_symbol.get(symbol, ()):
                    ctx = self._order_position_context(symbol, o.get("side"), o.get("params", {}) or {})
                    if ctx["is_reduce"]:
                        reduce_orders += 1
                        continue
                    o_price = o.get("price") or px
                    o_val = float(o.get("remaining", 0.0) or 0.0) * float(o_price)
                    opening_orders += 1
                    if ctx["pos_key"] == "LONG":
                        open_long += o_val
                    elif ctx["pos_key"] == "SHORT":
                        open_short += o_val

            base_max = max(base_long, base_short)
            with_orders_max = max(base_long + open_long, base_short + open_short)
            position_margin += base_max / self.leverage
            open_order_margin += max(0.0, with_orders_max - base_max) / self.leverage

            if with_orders_max > 0:
                rows.append({
                    "symbol": symbol,
                    "long_notional": base_long,
                    "short_notional": base_short,
                    "open_long_notional": open_long,
                    "open_short_notional": open_short,
                    "max_notional": with_orders_max,
                    "position_margin": base_max / self.leverage,
                    "open_order_margin": max(0.0, with_orders_max - base_max) / self.leverage,
                    "reduce_orders": reduce_orders,
                    "opening_orders": opening_orders,
                })

        rows.sort(key=lambda x: x["max_notional"], reverse=True)
        total_used = position_margin + open_order_margin
        return {
            "position_margin": position_margin,
            "open_order_margin": open_order_margin,
            "used_margin": total_used,
            "top_symbols": rows[:10],
        }

    def _incremental_im_for_order(self, symbol: str, pos_key: str, amount: float, price: float) -> float:
        """Extra initial margin if this opening order increases max hedge leg (mark-based)."""
        extra_long = amount if pos_key == "LONG" else 0.0
        extra_short = amount if pos_key == "SHORT" else 0.0
        _, _, new_max = self._symbol_hedge_notional(symbol, extra_long, extra_short, price)
        _, _, cur_max = self._symbol_hedge_notional(symbol)
        return max(0.0, (new_max - cur_max) / self.leverage)

    def _get_free_margin(self) -> float:
        total_unrealized = self._update_pnl()
        equity = self.balance["USDT"]["total"] + total_unrealized
        used_margin = self._total_initial_margin_required(include_open_orders=True)
        return equity - used_margin

    def _check_liquidation(self) -> bool:
        total_unrealized = self._update_pnl()
        equity = self.balance["USDT"]["total"] + total_unrealized
        
        tot_position_val = 0.0
        max_leg_position_val = 0.0
        for symbol in self.symbols:
            long_n, short_n, max_leg = self._symbol_hedge_notional(symbol)
            tot_position_val += long_n + short_n
            max_leg_position_val += max_leg

        # Maintenance on max hedge leg per symbol (unrealized PnL already in equity — cross-leg offset)
        maintenance_margin = max_leg_position_val * self.MAINTENANCE_MARGIN_RATE
        free_margin = self._get_free_margin()

        # Liquidate when available margin exhausted OR equity cannot cover maintenance
        should_liquidate = (
            equity <= 0.0
            or (max_leg_position_val > 0 and equity <= maintenance_margin)
        )
        if should_liquidate:
            top_exposures = []
            for symbol in self.symbols:
                px = self.get_latest_price(symbol)
                pos = self.positions[symbol]
                long_size = float(pos["LONG"]["size"])
                short_size = float(pos["SHORT"]["size"])
                long_entry = float(pos["LONG"]["entryPrice"])
                short_entry = float(pos["SHORT"]["entryPrice"])
                long_upnl = (px - long_entry) * long_size if long_size > 0 else 0.0
                short_upnl = (short_entry - px) * short_size if short_size > 0 else 0.0
                top_exposures.append({
                    "symbol": symbol,
                    "price": px,
                    "long_qty": long_size,
                    "short_qty": short_size,
                    "long_entry": long_entry,
                    "short_entry": short_entry,
                    "long_notional": long_size * px,
                    "short_notional": short_size * px,
                    "long_unrealized": long_upnl,
                    "short_unrealized": short_upnl,
                    "total_unrealized": long_upnl + short_upnl,
                })
            top_exposures.sort(key=lambda x: x["total_unrealized"])
            self.record_audit(
                "liquidation",
                {
                    "wallet": self.balance["USDT"]["total"],
                    "unrealized": total_unrealized,
                    "equity": equity,
                    "free_margin": free_margin,
                    "open_notional": tot_position_val,
                    "max_leg_notional": max_leg_position_val,
                    "maintenance_margin": maintenance_margin,
                    "top_exposures": top_exposures[:10],
                },
            )
            self.flush_audit(force=True)
            self.stop_reason = "liquidation"
            print(f"\n🚨🚨🚨 [MARGIN CALL / LIQUIDATION EVENT DETECTED] 🚨🚨🚨")
            print(
                f"📉 Equity: ${equity:.2f} | Free margin: ${free_margin:.2f} | "
                f"Open notional: ${tot_position_val:.2f} | Maint req: ${maintenance_margin:.2f}"
            )
            print("💥 Force-liquidating all open positions immediately to prevent negative balance...")
            print("🛑 Canceling all active open orders because of account liquidation...")
            
            # Reset active open orders completely
            self.active_open_orders.clear()
            self.open_orders.clear()
            for s in self.orders_by_symbol:
                self.orders_by_symbol[s].clear()
                
            syms = list(self.positions.keys())
            for symbol in syms:
                for side in ["LONG", "SHORT"]:
                    pos = self.positions[symbol][side]
                    if pos["size"] > 0:
                        price = self.get_latest_price(symbol)
                        self._execute_order(symbol, "sell" if side == "LONG" else "buy", pos["size"], price, {"reduceOnly": True, "positionIdx": 1 if side=="LONG" else 2})
            
            # Recalculate final balance/equity bounds
            final_unrealized = self._update_pnl()
            final_equity = self.balance["USDT"]["total"] + final_unrealized
            if final_equity < 0:
                print(f"💸 Account bankrupt! Resetting balance to $0.00 (from ${self.balance['USDT']['total']:.2f})")
                self.balance["USDT"]["total"] = 0.0
                self.balance["USDT"]["free"] = 0.0
            else:
                self.balance["USDT"]["total"] = max(0.0, final_equity)
                self.balance["USDT"]["free"] = max(0.0, final_equity)
                
            self.finalize_backtest()
            return False
            
        return True

    def record_audit(self, event_type: str, details: dict = None):
        """Chunked RAM-based fast audit logging to prevent OOM."""
        entry = {"ts_ms": self.current_timestamp_ms, "event": event_type}
        if details:
            entry.update(details)
        self.audit_buffer.append(entry)
        
        # Flush in chunks. Writing JSONL every few events dominates long backtests.
        if len(self.audit_buffer) >= self.audit_flush_threshold:
            self._flush_audit_chunk()

    def set_bot_perceived_pnl(self, value: float):
        """Used to track divergence between bot ledger and engine realization."""
        self.bot_perceived_pnl = float(value)

    def _flush_audit_chunk(self):
        try:
            if not self.audit_buffer:
                return
            mode = 'a' if getattr(self, '_audit_initialized', False) else 'w'
            self._audit_initialized = True
            with open("full_audit_log.jsonl", mode, encoding="utf-8") as f:
                for item in self.audit_buffer:
                    f.write(json.dumps(item, default=str) + "\n")
            self.audit_buffer.clear()
        except Exception as e:
            print(f"Error flushing audit log: {e}", flush=True)
            self.audit_buffer.clear()

    def flush_audit(self, force: bool = False):
        """Write buffered audit events to disk (call on crash/finalize)."""
        if force or len(self.audit_buffer) >= self.audit_flush_threshold:
            self._flush_audit_chunk()

    def _append_daily_progress_file(self, line: str):
        try:
            with open("backtest_daily.log", "a", encoding="utf-8") as f:
                f.write(line.rstrip() + "\n")
        except Exception:
            pass

    def tick(self) -> bool:
        if self.global_step >= self.max_idx - 1:
            self.stop_reason = "data_end"
            self.finalize_backtest()
            return False
            
        self.global_step += 1
        self.current_timestamp_ms = self.master_times[self.global_step]
        
        # Invalidate caches
        self.balance_cache = None
        self.unrealized_pnl_cache = None
        self.ohlcv_cache = {} 
        self.price_cache = {}
        self.indicator_cache = {}

        # When a symbol's data ends, stop simulating new fills for its resting orders,
        # but keep open positions marked at the last available price. Force-closing here
        # creates artificial realized PnL and hides unfinished ladders in the final report.
        for symbol in self.symbols:
            if symbol in self.fast_times and "1m" in self.fast_times[symbol]:
                last_ts = self.fast_times[symbol]["1m"][-1]
                if self.current_timestamp_ms > last_ts:
                    canceled = self._cancel_resting_orders_for_symbol(symbol)
                    if canceled > 0:
                        print(f"⚠️ Data ended for {symbol}. Kept open position(s) marked at last price; canceled {canceled} resting order(s).")

        self._process_open_orders()
        
        # Check for liquidation / margin call
        if not self._check_liquidation():
            return False
        
        # Equity curve snapshot (once every hour)
        if self.global_step % 60 == 0:
            total_unrealized = self._update_pnl()
            total_realized = sum(self.realized_pnl_by_symbol.values())
            wallet = self.balance["USDT"]["total"]
            eq = wallet + total_unrealized
            
            open_notional = 0.0
            symbol_snapshots = {}
            for symbol in self.symbols:
                price = self.get_latest_price(symbol)
                pos_dict = self.positions[symbol]
                
                # Active size for this symbol
                ls = pos_dict["LONG"]["size"]
                ss = pos_dict["SHORT"]["size"]
                sym_notional = (ls + ss) * price
                open_notional += sym_notional
                
                # Unrealized for this symbol
                upnl = self._calc_unrealized_pnl(symbol)
                realized_raw = self.realized_pnl_by_symbol[symbol]
                fee_total = self.symbol_fees[symbol]
                
                # Metadata for intelligence dossier
                active_orders = [o for o in self.active_open_orders.values() if self._normalize_to_ccxt(o["symbol"]) == symbol]
                
                # Unique ladder rungs (grouping by price to avoid double-counting)
                ladders_active = set()
                for o in active_orders:
                    p = o.get('params', {}) or {}
                    if (p.get('rung_index') is not None or 
                        p.get('ladder_id') is not None or 
                        p.get('ladder_entry_price') is not None):
                        ladders_active.add(o.get('price', 0))
                
                # Retrieve technical indicators (snapshot available TFs)
                rsi_vals = {}
                for tf, data in self.fast_indicators[symbol].items():
                    if 'rsi' in data:
                        rsi_vals[tf] = round(float(self.get_indicator(symbol, tf, "rsi") or 50.0), 2)
                
                symbol_snapshots[symbol] = {
                    "price": price,
                    "notional": sym_notional,
                    "long_notional": ls * price,
                    "short_notional": ss * price,
                    "pnl": realized_raw + upnl,
                    "realized": realized_raw,
                    "unrealized": upnl,
                    "fees": fee_total,
                    "entry_long": pos_dict["LONG"]["entryPrice"],
                    "entry_short": pos_dict["SHORT"]["entryPrice"],
                    "size_long": ls,
                    "size_short": ss,
                    "rsi_map": rsi_vals,
                    "ladders": len(ladders_active)
                }

            self.equity_curve.append({
                "time": self.current_timestamp_ms,
                "equity": eq,
                "wallet": wallet,
                "notional": open_notional,
                "realized_pnl": total_realized,
                "unrealized_pnl": total_unrealized,
                "total_fees": getattr(self, "total_fees_paid", 0.0),
                "bot_pnl": getattr(self, "bot_perceived_pnl", 0.0),
                "symbols": symbol_snapshots
            })
            snap = self.get_accounting_snapshot()
            self.record_audit(
                "equity_snapshot",
                {
                    "wallet": wallet,
                    "unrealized": total_unrealized,
                    "equity": eq,
                    "free_margin": self._get_free_margin(),
                    "open_notional": open_notional,
                    "wallet_realized": snap["wallet_realized"],
                    "engine_trade_log_pnl": snap["engine_trade_log_pnl"],
                    "open_legs": snap["open_legs"],
                },
            )
            if self.global_step % 1440 == 0:
                dt_str = str(pd.to_datetime(self.current_timestamp_ms, unit='ms'))
                pct = 100.0 * self.global_step / max(1, self.max_idx - 1)
                line = (
                    f"📈 [PASS] {dt_str} | step {self.global_step}/{self.max_idx} ({pct:.1f}%) | "
                    f"Equity: ${eq:.2f} | Wallet: ${wallet:.2f} | Unrealized: ${total_unrealized:.2f} | "
                    f"Realized(cash): ${snap['wallet_realized']:.2f} | Engine closes: ${snap['engine_trade_log_pnl']:.2f} | "
                    f"Open legs: {snap['open_legs']} | Notional: ${open_notional:.0f}"
                )
                print(line, flush=True)
                self._append_daily_progress_file(line)
                self.flush_audit(force=True)

        return True

    def finalize_backtest(self):
        if getattr(self, "is_finalized", False):
            return
        self.is_finalized = True
        try:
            print("\n⏳ Finishing backtest: Calculating final metrics...")
            total_unrealized = self._update_pnl()
            wallet = float(self.balance["USDT"]["total"])
            equity = wallet + total_unrealized
            if not self.equity_curve or int(self.equity_curve[-1].get("time", 0) or 0) != int(self.current_timestamp_ms):
                open_notional = 0.0
                symbol_snapshots = {}
                for symbol in self.symbols:
                    price = self.get_latest_price(symbol)
                    pos_dict = self.positions[symbol]
                    ls = float(pos_dict["LONG"]["size"])
                    ss = float(pos_dict["SHORT"]["size"])
                    sym_notional = (ls + ss) * price
                    open_notional += sym_notional
                    upnl = self._calc_unrealized_pnl(symbol)
                    realized_raw = self.realized_pnl_by_symbol[symbol]
                    fee_total = self.symbol_fees[symbol]
                    active_orders = [o for o in self.active_open_orders.values() if self._normalize_to_ccxt(o["symbol"]) == symbol]
                    ladders_active = set()
                    for o in active_orders:
                        p = o.get("params", {}) or {}
                        if p.get("rung_index") is not None or p.get("ladder_id") is not None or p.get("ladder_entry_price") is not None:
                            ladders_active.add(o.get("price", 0))
                    rsi_vals = {}
                    for tf, data in self.fast_indicators[symbol].items():
                        if "rsi" in data:
                            rsi_vals[tf] = round(float(self.get_indicator(symbol, tf, "rsi") or 50.0), 2)
                    symbol_snapshots[symbol] = {
                        "price": price,
                        "notional": sym_notional,
                        "long_notional": ls * price,
                        "short_notional": ss * price,
                        "pnl": realized_raw + upnl,
                        "realized": realized_raw,
                        "unrealized": upnl,
                        "fees": fee_total,
                        "entry_long": pos_dict["LONG"]["entryPrice"],
                        "entry_short": pos_dict["SHORT"]["entryPrice"],
                        "size_long": ls,
                        "size_short": ss,
                        "rsi_map": rsi_vals,
                        "ladders": len(ladders_active),
                    }
                self.equity_curve.append({
                    "time": self.current_timestamp_ms,
                    "equity": equity,
                    "wallet": wallet,
                    "notional": open_notional,
                    "realized_pnl": sum(self.realized_pnl_by_symbol.values()),
                    "unrealized_pnl": total_unrealized,
                    "total_fees": getattr(self, "total_fees_paid", 0.0),
                    "bot_pnl": getattr(self, "bot_perceived_pnl", 0.0),
                    "symbols": symbol_snapshots,
                })
            
            start_bal = float(self.start_balance)
            end_bal = equity
            total_pnl = end_bal - start_bal
            total_vol = getattr(self, "total_volume", 0.0)
            stop_reason = getattr(self, "stop_reason", "completed")
            
            max_dd = 0.0
            if self.equity_curve:
                max_eq = start_bal
                for e in self.equity_curve:
                    eq_val = float(e["equity"])
                    if eq_val > max_eq: max_eq = eq_val
                    current_dd = ((max_eq - eq_val) / max_eq * 100) if max_eq > 0 else 0.0
                    if current_dd > max_dd: max_dd = current_dd

            self.record_audit(
                "audit_summary",
                {
                    "start_wallet": start_bal,
                    "final_wallet": wallet,
                    "final_equity": equity,
                    "total_pnl": total_pnl,
                    "max_drawdown": max_dd,
                    "trade_log_closes": len(self.trade_log),
                    "stop_reason": stop_reason,
                },
            )
            self.flush_audit(force=True)

            # Generate high-fidelity report
            try:
                self.generate_html_report()
            except Exception as h_err:
                print(f"❌ Error during HTML report generation: {h_err}")
            finally:
                self.flush_audit(force=True)

            # Final Console Summary
            print(f"\n{C_CYAN}{'='*65}{C_RESET}")
            print(f"{C_BOLD}  [QUANTUM ENGINE] BACKTEST SEQUENCE COMPLETE{C_RESET}")
            print(f"{C_CYAN}{'-'*65}{C_RESET}")
            print(f"  Initial Equity:  {C_WHITE}${start_bal:,.2f}{C_RESET}")
            print(f"  Final Equity:    {C_GREEN if total_pnl >= 0 else C_RED}${end_bal:,.2f}{C_RESET}")
            print(f"  Net PnL:         {C_GREEN if total_pnl >= 0 else C_RED}${total_pnl:+,.2f}{C_RESET}")
            print(f"  Total Volume:    {C_WHITE}${total_vol:,.2f}{C_RESET}")
            print(f"  Max Drawdown:    {C_RED}-{max_dd:.2f}%{C_RESET}")
            print(f"  Stop Reason:     {C_YELLOW}{stop_reason}{C_RESET}")
            print(f"  Audit Analytics: {C_YELLOW}backtest_report.html{C_RESET}")
            print(f"{C_CYAN}{'='*65}{C_RESET}\n")

        except Exception as e:
            print(f"❌ Fatal error in finalize_backtest: {e}")
            import traceback
            traceback.print_exc()

    def generate_html_report(self):
        print("📊 Constructing Quantum High-Fidelity Intelligence Audit v19.1...")
        
        start_bal = float(self.start_balance)
        wallet_bal = float(self.balance["USDT"]["total"])
        unrealized = self._update_pnl()
        end_bal = wallet_bal + unrealized
        total_pnl = end_bal - start_bal
        win_trades = [t for t in self.trade_log if t['pnl'] > 0]
        win_rate = (len(win_trades) / len(self.trade_log) * 100) if self.trade_log else 0
        total_fees = float(getattr(self, "total_fees_paid", 0.0))
        total_vol = float(getattr(self, "total_volume", 0.0))
        
        # 1. Timeline Extractor
        labels = []
        label_ts = []
        equity_vals = []
        balance_vals = []
        notional_vals = []
        drawdown_vals = []
        realized_vals = []
        unrealized_vals = []
        fee_vals = []
        drift_vals = []
        bot_pnl_vals = []
        metadata_dump = [] 
        
        sym_pnl_curves = {s: [] for s in self.symbols}
        sym_long_notional = {s: [] for s in self.symbols}
        sym_short_notional = {s: [] for s in self.symbols}
        ladder_snapshots = defaultdict(list)
        weekly_ema50_events = defaultdict(list)
        macro_switch_events = defaultdict(list)
        try:
            if os.path.exists("full_audit_log.jsonl"):
                with open("full_audit_log.jsonl", "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        has_exposure = '"event": "exposure_snapshot"' in line or '"event":"exposure_snapshot"' in line
                        has_weekly_ema = "weekly_ema50_dynamic_inverse" in line
                        has_macro_switch = "hybrid_check" in line and ('"changed": true' in line or '"changed":true' in line)
                        if not (has_exposure or has_weekly_ema or has_macro_switch):
                            continue
                        try:
                            ev = json.loads(line)
                        except Exception:
                            continue
                        pair = ev.get("pair")
                        event_name = ev.get("event")
                        if event_name == "exposure_snapshot" and pair:
                            ladder_snapshots[pair].append(ev)
                        elif event_name == "weekly_ema50_dynamic_inverse" and pair:
                            try:
                                ema50 = float(ev.get("ema50"))
                                ts_ms = int(float(ev.get("ts_ms", 0) or 0))
                                weekly_ema50_events[pair].append((ts_ms, round(ema50, 8)))
                            except Exception:
                                pass
                        elif event_name == "hybrid_check" and pair and ev.get("changed") is True:
                            try:
                                macro_switch_events[pair].append({
                                    "ts_ms": int(float(ev.get("ts_ms", 0) or 0)),
                                    "rsi": round(float(ev.get("rsi", 0.0) or 0.0), 4),
                                    "old": ev.get("old_side"),
                                    "new": ev.get("new_side"),
                                    "tf": ev.get("tf"),
                                    "dyn_os": round(float(ev.get("dyn_os", 0.0) or 0.0), 4),
                                    "dyn_ob": round(float(ev.get("dyn_ob", 0.0) or 0.0), 4),
                                })
                            except Exception:
                                pass
                for pair in ladder_snapshots:
                    ladder_snapshots[pair].sort(key=lambda x: int(x.get("ts_ms", 0) or 0))
                for pair in weekly_ema50_events:
                    weekly_ema50_events[pair].sort(key=lambda x: x[0])
                for pair in macro_switch_events:
                    macro_switch_events[pair].sort(key=lambda x: x.get("ts_ms", 0))
        except Exception as ex:
            print(f"⚠️ Could not load report audit overlays: {ex}")
        ladder_ptrs = {s: 0 for s in self.symbols}
        latest_ladder_state = {s: None for s in self.symbols}
        ladder_event_dump = {s: [] for s in self.symbols}
        ladder_event_sig = {s: None for s in self.symbols}

        def compact_report_ladder(ladder):
            return {
                "side": ladder.get("side", ""),
                "ladder_id": ladder.get("ladder_id", ""),
                "entry": round(float(ladder.get("entry", 0.0) or 0.0), 8),
                "tp": round(float(ladder.get("tp", 0.0) or 0.0), 8),
                "size": round(float(ladder.get("size", 0.0) or 0.0), 8),
                "entry_notional": round(float(ladder.get("entry_notional", 0.0) or 0.0), 4),
                "mark_notional": round(float(ladder.get("mark_notional", 0.0) or 0.0), 4),
                "profile_id": ladder.get("profile_id", ""),
                "has_tp": bool(ladder.get("has_tp", False)),
            }
        
        total_long_vals = []
        total_short_vals = []
        
        max_eq = start_bal
        max_dd = 0.0
        for e in self.equity_curve:
            ts = int(e["time"])
            eq = float(e["equity"])
            wal = float(e.get("wallet", eq))
            notional = float(e.get("notional", 0.0))
            
            if eq > max_eq: max_eq = eq
            dd_pct = ((max_eq - eq) / max_eq * 100) if max_eq > 0 else 0.0
            if dd_pct > max_dd: max_dd = dd_pct
            
            labels.append(pd.to_datetime(ts, unit='ms').strftime("%b %d %H:%M"))
            label_ts.append(ts)
            equity_vals.append(round(eq, 2))
            balance_vals.append(round(wal, 2))
            notional_vals.append(round(notional, 2))
            drawdown_vals.append(round(dd_pct, 2))
            realized_vals.append(round(float(e.get("realized_pnl", 0.0)), 2))
            unrealized_vals.append(round(float(e.get("unrealized_pnl", 0.0)), 2))
            
            fee_tot = float(e.get("total_fees", 0.0))
            fee_vals.append(round(fee_tot, 2))
            
            bot_pnl = float(e.get("bot_pnl", 0.0))
            bot_pnl_vals.append(round(bot_pnl, 2))
            drift_vals.append(round(bot_pnl - (eq - start_bal), 2))
            
            sym_snaps = e.get("symbols", {})
            for s in self.symbols:
                snaps = ladder_snapshots.get(s, [])
                ptr = ladder_ptrs.get(s, 0)
                while ptr < len(snaps) and int(snaps[ptr].get("ts_ms", 0) or 0) <= ts:
                    latest_ladder_state[s] = snaps[ptr]
                    ptr += 1
                ladder_ptrs[s] = ptr
                if latest_ladder_state.get(s) is not None:
                    state = latest_ladder_state[s]
                    ladders = [compact_report_ladder(l) for l in state.get("long_ladders", [])]
                    ladders.extend(compact_report_ladder(l) for l in state.get("short_ladders", []))
                    sig = json.dumps(ladders, separators=(",", ":"), ensure_ascii=False)
                    if ladder_event_sig.get(s) != sig:
                        ladder_event_sig[s] = sig
                        ladder_event_dump[s].append({
                            "ts_ms": int(state.get("ts_ms", 0) or 0),
                            "ladders": ladders,
                        })
                    sym_snaps.setdefault(s, {})["ladder_event_idx"] = len(ladder_event_dump[s]) - 1
            # Enrich snapshot with global metrics for JS metadata access
            enriched_snap = {
                "symbols": sym_snaps,
                "drift": round(bot_pnl - (eq - start_bal), 2),
                "unrealized": round(float(e.get("unrealized_pnl", 0.0)), 2),
                "realized": round(float(e.get("realized_pnl", 0.0)), 2),
                "fees": round(fee_tot, 2)
            }
            metadata_dump.append(enriched_snap)
            
            snap_long = 0.0
            snap_short = 0.0
            for s in self.symbols:
                s_data = sym_snaps.get(s, {})
                snap_long += float(s_data.get("long_notional", 0.0))
                snap_short += float(s_data.get("short_notional", 0.0))
                sym_pnl_curves[s].append(round(float(s_data.get("pnl", 0.0)), 2))
                sym_long_notional[s].append(round(float(s_data.get("long_notional", 0.0)), 2))
                sym_short_notional[s].append(round(float(s_data.get("short_notional", 0.0)), 2))
            
            total_long_vals.append(round(snap_long, 2))
            total_short_vals.append(round(snap_short, 2))

        weekly_ema50_overlay = {}
        for s in self.symbols:
            events = weekly_ema50_events.get(s, [])
            vals = []
            ptr = 0
            current_ema = None
            for ts in label_ts:
                while ptr < len(events) and events[ptr][0] <= ts:
                    current_ema = events[ptr][1]
                    ptr += 1
                vals.append(current_ema)
            weekly_ema50_overlay[s] = vals

        macro_switch_overlay = {}
        for s in self.symbols:
            points = []
            for sw in macro_switch_events.get(s, []):
                ts_ms = int(sw.get("ts_ms", 0) or 0)
                idx = bisect.bisect_left(label_ts, ts_ms)
                if idx >= len(label_ts):
                    idx = len(label_ts) - 1
                if idx > 0 and abs(label_ts[idx - 1] - ts_ms) <= abs(label_ts[idx] - ts_ms):
                    idx -= 1
                if 0 <= idx < len(labels):
                    points.append({
                        "idx": idx,
                        "x": labels[idx],
                        "rsi": sw.get("rsi"),
                        "old": sw.get("old"),
                        "new": sw.get("new"),
                        "tf": sw.get("tf"),
                        "dyn_os": sw.get("dyn_os"),
                        "dyn_ob": sw.get("dyn_ob"),
                    })
            macro_switch_overlay[s] = points

        # 2. Daily PnL
        daily_pnl_labels = []
        daily_pnl_values = []
        if self.trade_log:
            df_trades = pd.DataFrame(self.trade_log)
            if not df_trades.empty:
                df_trades['dt'] = pd.to_datetime(df_trades['ts'], unit='ms').dt.date
                daily = df_trades.groupby('dt')['pnl'].sum()
                daily_pnl_labels = [d.strftime("%b %d") for d in daily.index]
                daily_pnl_values = [round(float(v), 2) for v in daily.values]

        # 3. Chart Dataset Logic - Reference Visuals
        colors = ['#3fb950', '#58a6ff', '#f85149', '#d29922', '#8957e5', '#db61a2', '#a371f7', '#6e7681', '#2ea043', '#f0883e', '#79c0ff', '#ff7b72']
        def get_c(i): return colors[i % len(colors)]
        
        sym_pnl_datasets = []
        sym_pos_datasets = []
        for i, s in enumerate(self.symbols):
            col = get_c(i+2) 
            s_short = s.split('/')[0]
            sym_pnl_datasets.append({
                "label": f"{s_short} PnL",
                "data": sym_pnl_curves[s],
                "borderColor": col,
                "borderWidth": 4.0,
                "pointRadius": 0,
                "tension": 0.1
            })
            sym_pos_datasets.append({
                "label": f"{s_short} LONG",
                "data": sym_long_notional[s],
                "borderColor": col,
                "borderWidth": 4.0,
                "pointRadius": 0,
                "tension": 0.1
            })
            sym_pos_datasets.append({
                "label": f"{s_short} SHORT",
                "data": sym_short_notional[s],
                "borderColor": col,
                "borderWidth": 3.0,
                "borderDash": [4, 4],
                "pointRadius": 0,
                "tension": 0.1
            })

        sym_stats = {}
        for s in self.symbols:
            trades = [t for t in self.trade_log if t['symbol'] == s]
            total = len(trades)
            wr = (len([t for t in trades if t['pnl'] > 0]) / total * 100) if total > 0 else 0
            pnl = sum([t['pnl'] for t in trades])
            sym_stats[s] = {"ops": total, "wr": wr, "pnl": pnl}
        
        sym_rows = ""
        for s, st in sorted(sym_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
            c = "pos" if st['pnl'] >= 0 else "neg"
            sym_rows += f"<tr><td>{s.split('/')[0]}</td><td>{st['ops']}</td><td>{st['wr']:.1f}%</td><td class='{c}'>${st['pnl']:,.2f}</td></tr>"

        history_rows = ""
        for t in reversed(self.trade_log[-60:]):
            dt = pd.to_datetime(t["ts"], unit="ms").strftime("%H:%M:%S")
            c = "pos" if t["pnl"] >= 0 else "neg"
            notional = float(t.get("notional", 0.0) or 0.0)
            pnl_pct = float(t.get("pnl_pct", 0.0) or 0.0)
            history_rows += (
                f"<tr><td>{dt}</td><td>{t['symbol'].split('/')[0]}</td><td>{t['side']}</td>"
                f"<td>${notional:,.2f}</td><td>{pnl_pct:.2f}%</td><td class='{c}'>${t['pnl']:,.2f}</td></tr>"
            )

        audit_rows = ""
        for e in reversed(self.audit_buffer[-800:]):
            ts = pd.to_datetime(e["ts_ms"], unit="ms").strftime("%m/%d %H:%M:%S")
            evt = str(e["event"]).upper()
            payload = ", ".join([f"{k}: {v}" for k, v in e.items() if k not in ["ts_ms", "event"]][:4])
            if len(payload) > 90: payload = payload[:87] + "..."
            row_cls = "crit" if any(x in evt for x in ["LIQ", "MARGIN", "STOP"]) else ("warn" if any(x in evt for x in ["REJECT", "CANCEL"]) else "")
            audit_rows += f"<tr class='{row_cls}'><td>{ts}</td><td>{evt}</td><td>{payload}</td></tr>"

        html = f"""<!DOCTYPE html>
<html lang="ka">
<head>
    <meta charset="UTF-8">
    <title>ბექტიესტი V19.2 | QUANTUM DOSSIER</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;500;700&family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #000000; --panel: #0d1117; --border: #1a1e23; 
            --text: #c9d1d9; --muted: #6e7681; 
            --green: #3fb950; --blue: #58a6ff; --red: #f85149; --yellow: #d29922;
        }}
        * {{ box-sizing: border-box; }}
        body {{ 
            background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; 
            margin: 0; padding: 20px; font-size: 13px; scrollbar-width: thin;
        }}
        .brand-header {{ 
            display: flex; justify-content: space-between; align-items: flex-end; 
            padding: 0 10px 20px 10px; margin-bottom: 20px; border-bottom: 1px solid #1a1e23;
        }}
        .brand-title {{ font-size: 14px; font-weight: 800; color: #fff; letter-spacing: 0.5px; opacity: 0.8; }}
        
        .dash-grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin-bottom: 25px; }}
        .widget {{ background: #000; border: 1px solid #1a1e23; border-radius: 4px; padding: 25px; position: relative; }}
        .widget-title {{ 
            font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 700; color: #fff; 
            text-transform: uppercase; margin-bottom: 20px; text-align: center; opacity: 0.9;
        }}

        .stats-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 25px; }}
        .stat-card {{ background: #000; border: 1px solid #1a1e23; padding: 15px; border-radius: 4px; }}
        .stat-label {{ font-size: 9px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }}
        .stat-value {{ font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 600; }}

        .chart-container {{ height: 480px; width: 100%; position: relative; }}
        .chart-wide {{ height: 420px; }}

        .main-layout {{ display: grid; grid-template-columns: 1fr 380px; gap: 15px; }}
        
        table {{ width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 11px; }}
        th {{ text-align: left; padding: 10px; border-bottom: 1px solid #1a1e23; color: var(--muted); font-size: 9px; text-transform: uppercase; }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #0a0a0a; }}
        .scroll {{ max-height: 480px; overflow-y: auto; overflow-x: hidden; }}
        
        .pos {{ color: var(--green); }} .neg {{ color: var(--red); }}
        .crit {{ background: rgba(248, 81, 73, 0.1); color: #ff9a92; }}
        .warn {{ background: rgba(210, 153, 34, 0.08); color: #f1e05a; }}
        
        ::-webkit-scrollbar {{ width: 4px; }}
        ::-webkit-scrollbar-thumb {{ background: #21262d; border-radius: 2px; }}

        .legend-sidebar {{ font-size: 10px; color: #fff; opacity: 0.8; padding-top: 10px; }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 4px; }}
        .legend-color {{ width: 12px; height: 3px; margin-right: 8px; border-radius: 1px; }}
        .legend-dash {{ border-top: 1px dashed; height: 0; }}
        .chart-toolbar {{
            display: flex; justify-content: flex-end; align-items: center; gap: 10px;
            margin: -8px 0 12px 0; font-family: 'JetBrains Mono', monospace;
        }}
        .chart-toolbar label {{ color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }}
        .chart-toolbar select {{
            background: #050505; color: var(--text); border: 1px solid #30363d;
            border-radius: 4px; padding: 8px 10px; font-family: 'JetBrains Mono', monospace;
            font-size: 12px; outline: none;
        }}
        .pair-readout {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 8px; margin-top: 12px;
        }}
        .pair-readout div {{
            border: 1px solid #1a1e23; background: #050505; border-radius: 4px;
            padding: 9px 10px; font-family: 'JetBrains Mono', monospace; font-size: 11px;
            color: var(--muted);
        }}
        .pair-readout strong {{ display: block; color: var(--text); font-size: 14px; margin-top: 3px; }}
    </style>
</head>
<body>
    <div class="brand-header">
        <div class="brand-title">ბექტიესტი V19.2 - [ინტელექტუალური აუდიტი]</div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--muted);">ENGINE_BUILD_SUCCESS // SYNC_OK</div>
    </div>

    <div class="stats-row">
        <div class="stat-card"><div class="stat-label">საწყისი კაპიტალი</div><div class="stat-value">${start_bal:,.2f}</div></div>
        <div class="stat-card"><div class="stat-label">საბოლოო ექუითი</div><div class="stat-value pos">${end_bal:,.2f}</div></div>
        <div class="stat-card"><div class="stat-label">სუფთა მოგება</div><div class="stat-value {'pos' if total_pnl >= 0 else 'neg'}">${total_pnl:+,.2f}</div></div>
        <div class="stat-card"><div class="stat-label">Drawdown (Max)</div><div class="stat-value neg">-{max_dd:.2f}%</div></div>
        <div class="stat-card"><div class="stat-label">Edge Score (WR)</div><div class="stat-value">{win_rate:.1f}%</div></div>
        <div class="stat-card"><div class="stat-label">ჯამური მოცულობა</div><div class="stat-value">${total_vol:,.0f}</div></div>
    </div>

    <div class="main-layout">
        <div class="visuals">
            <div class="dash-grid">
                <div class="widget">
                    <div class="widget-title">ექუითის მრუდი: შესრულების კონტროლი ($)</div>
                    <div class="chart-container"><canvas id="equityChart"></canvas></div>
                </div>
                <div class="widget">
                    <div class="widget-title">Pair Tape: ფასი, საშუალოები, ლადერები და ექსპოზიცია</div>
                    <div class="chart-toolbar">
                        <label for="pairSelect">Pair</label>
                        <select id="pairSelect">
                            {"".join([f'<option value="{s}">{s.split("/")[0]}</option>' for s in self.symbols])}
                        </select>
                    </div>
                    <div class="chart-container"><canvas id="pairTapeChart"></canvas></div>
                    <div class="pair-readout" id="pairReadout"></div>
                    <div class="scroll" style="max-height: 260px; margin-top: 12px;">
                        <table>
                            <thead><tr><th>Side</th><th>ID</th><th>Entry</th><th>TP</th><th>Size</th><th>Entry USD</th><th>Mark USD</th><th>Profile</th><th>TP?</th></tr></thead>
                            <tbody id="pairLadderRows"></tbody>
                        </table>
                    </div>
                </div>
                <div class="widget">
                    <div class="widget-title">სიმბოლოების PnL: აქტივების დინამიკა ($)</div>
                    <div class="chart-container"><canvas id="symPnlChart"></canvas></div>
                </div>
                <div class="widget">
                    <div class="widget-title">SYNC DRIFT: ძრავისა და ბოტის სინქრონიზაცია ($)</div>
                    <div class="chart-container"><canvas id="driftChart"></canvas></div>
                </div>
                <div class="widget">
                    <div class="widget-title">აგრეგირებული სენტიმენტი (Long vs Short ექსპოზიცია)</div>
                    <div class="chart-container"><canvas id="sentimentChart"></canvas></div>
                </div>
                <div class="widget">
                    <div class="widget-title">რისკის მეტრიკა: Drawdown ვიზუალიზაცია (%)</div>
                    <div class="chart-container"><canvas id="ddChart"></canvas></div>
                </div>
                <div class="widget">
                    <div class="widget-title">ხარჯების ანალიზი: კუმულატიური საკომისიო ($)</div>
                    <div class="chart-container"><canvas id="feeChart"></canvas></div>
                </div>
            </div>

            <div class="widget" style="margin-bottom: 20px;">
                <div class="widget-title">ჯამური ნომინალური ექსპოზიცია დროსთან მიმართებაში ($)</div>
                <div class="chart-container chart-wide"><canvas id="notionalChart"></canvas></div>
            </div>

            <div class="widget" style="margin-bottom: 20px;">
                <div class="widget-title">დინამიური პოზიციის ზომა პაირებზე ($)</div>
                <div class="chart-container chart-wide"><canvas id="symSizeChart"></canvas></div>
            </div>

            <div class="widget" style="margin-bottom: 20px;">
                <div class="widget-title">დღიური PnL რეზიუმე ($)</div>
                <div class="chart-container chart-wide"><canvas id="dailyPnlChart"></canvas></div>
            </div>

            <div class="widget">
                <div class="widget-title">ინფორმაციული ნაკადის ლოგები</div>
                <div class="scroll">
                    <table>
                        <thead><tr><th width="140">ტელემეტრია_TS</th><th width="160">მოვლენის_ტიპი</th><th>მონაცემები</th></tr></thead>
                        <tbody>{audit_rows}</tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="sidebar">
            <div class="widget">
                <div class="widget-title">აქტივების შედეგები</div>
                <table>
                    <thead><tr><th>აქტივი</th><th>ოპერაცია</th><th>WR</th><th>Engine PNL</th></tr></thead>
                    <tbody>{sym_rows}</tbody>
                </table>
            </div>

            <div class="widget" style="margin-top: 20px;">
                <div class="widget-title">ბოლო აღსრულებები (Engine Avg PNL)</div>
                <div class="scroll" style="max-height: 400px;">
                    <table>
                        <thead><tr><th>დრო</th><th>აქტივი</th><th>მხარე</th><th>Mark USD</th><th>Engine PNL%</th><th>Engine PNL</th></tr></thead>
                        <tbody>{history_rows}</tbody>
                    </table>
                </div>
            </div>

            <div class="widget" style="margin-top: 20px;">
                <div class="widget-title">საბოლოო ექსპოზიცია (Notional USD)</div>
                <div class="scroll" style="max-height: 400px;">
                    <table style="width:100%; font-size: 0.8em;">
                        <thead><tr><th>Sym</th><th>Long Qty</th><th>Short Qty</th><th>Long USD</th><th>Short USD</th></tr></thead>
                        <tbody>
                            {"".join([f'<tr><td>{s.split("/")[0]}</td><td>{self.positions[s]["LONG"]["size"]:.2f}</td><td>{self.positions[s]["SHORT"]["size"]:.2f}</td><td>${ (self.positions[s]["LONG"]["size"] * (self.get_latest_price(s) or 0.0)):,.0f}</td><td>${ (self.positions[s]["SHORT"]["size"] * (self.get_latest_price(s) or 0.0)):,.0f}</td></tr>' for s in self.symbols])}
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="widget" style="margin-top: 20px;">
                <div class="widget-title">ლეგენდა</div>
                <div class="legend-sidebar">
                    <div class="legend-item"><div class="legend-color" style="background:#3fb950"></div>ექუითი (Engine)</div>
                    <div class="legend-item"><div class="legend-color legend-dash" style="border-color:#58a6ff"></div>ბალანსი (Dashed)</div>
                    <div class="legend-item"><div class="legend-color legend-dash" style="border-color:#a371f7"></div>რეალიზებული PnL (Purple)</div>
                    <div class="legend-item"><div class="legend-color legend-dash" style="border-color:#db61a2"></div>არარეალიზებული PnL (Pink)</div>
                    <div class="legend-item"><div class="legend-color" style="background:#f85149"></div>Drawdown მრუდი</div>
                    <div class="legend-item"><div class="legend-color" style="background:#d29922"></div>ნომინალური არეა</div>
                    <div class="legend-item"><div class="legend-color" style="background:#58a6ff"></div>Long ექსპოზიცია</div>
                    <div class="legend-item"><div class="legend-color legend-dash" style="border-color:#f85149"></div>Short ექსპოზიცია</div>
                    <div class="legend-item"><div class="legend-color" style="background:#8957e5"></div>Sync დრიფტის ინდექსი</div>
                    { "".join([f'<div class="legend-item"><div class="legend-color" style="background:{get_c(i+2)}"></div>{s.split("/")[0]} Metrics</div>' for i,s in enumerate(self.symbols)]) }
                </div>
            </div>
        </div>
    </div>

    <div id="missing-data-warning" style="display:none; background:#f85149; color:#fff; padding:15px; text-align:center; font-weight:bold; font-size:14px; margin-bottom:20px; border-radius:4px;">
        Warning: 'backtest_data.js' was not found or failed to load. Please make sure 'backtest_data.js' is in the same folder as this HTML file.
    </div>

    <script src="backtest_data.js"></script>
    <script>
        if (typeof window.backtest_labels === 'undefined' || typeof window.backtest_metadata === 'undefined') {{
            document.getElementById('missing-data-warning').style.display = 'block';
        }}
        const labels = window.backtest_labels || [];
        const metadata = window.backtest_metadata || [];
        const symbols = window.backtest_symbols || [];

        const chartDefs = {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{ 
                legend: {{ display: false }},
                tooltip: {{ 
                    backgroundColor: '#000', borderColor: '#30363d', borderWidth: 1,
                    titleFont: {{ family: 'Inter', size: 14, weight: 'bold' }},
                    bodyFont: {{ family: 'JetBrains Mono', size: 13 }},
                    padding: 15, cornerRadius: 4,
                    callbacks: {{
                        afterBody: function(context) {{
                            const idx = context[0].dataIndex;
                            const meta = metadata[idx];
                            if (!meta) return "";
                            
                            // Determine if we should filter symbols based on chart context
                            const datasetLabel = context[0].dataset.label || "";
                            let filterSym = null;
                            if (datasetLabel.includes(" PnL") || datasetLabel.includes(" LONG") || datasetLabel.includes(" SHORT")) {{
                                filterSym = datasetLabel.split(" ")[0];
                            }}

                            let lines = ["", "--- 📊 ინტელექტუალური დოსიე ---"];
                            lines.push(`ჯამური Realized: $${{meta.realized.toFixed(2)}}`);
                            lines.push(`ჯამური Unrealized: $${{meta.unrealized.toFixed(2)}}`);
                            lines.push(`საკომისიოები: $${{meta.fees.toFixed(2)}}`);
                            lines.push(`Ledger vs Equity gap: $${{meta.drift.toFixed(2)}}`);
                            lines.push("");
                            
                            for (const [sym, data] of Object.entries(meta.symbols)) {{
                                const symShort = sym.split('/')[0];
                                
                                // თუ ფილტრი აქტიურია (მაგ. PnL ჩარტზე), აჩვენე მხოლოდ ეს პაირი.
                                // სხვა შემთხვევაში (Equity ჩარტზე) აჩვენე ყველა აქტიური პოზიცია.
                                if (filterSym && symShort !== filterSym) continue;

                                if (data.notional > 0 || Math.abs(data.pnl) > 0.01) {{
                                    lines.push(`>> ${{symShort}} (ფასი: $${{data.price.toFixed(2)}})`);
                                    lines.push(`   PnL: $${{data.pnl.toFixed(2)}} (R: ${{data.realized.toFixed(2)}} / U: ${{data.unrealized.toFixed(2)}})`);
                                    const ladderState = data.ladder_state || null;
                                    const ladderLongAvg = ladderState ? Number(ladderState.long_avg_entry || 0) : 0;
                                    const ladderShortAvg = ladderState ? Number(ladderState.short_avg_entry || 0) : 0;
                                    if (data.size_long > 0) {{
                                        let line = `   🟢 LONG engine avg: $${{data.entry_long.toFixed(4)}} (USD: $${{data.long_notional ? data.long_notional.toFixed(2) : '0.00'}})`;
                                        if (ladderLongAvg > 0 && Math.abs(ladderLongAvg - data.entry_long) > 1e-8) line += ` | ladder avg: $${{ladderLongAvg.toFixed(4)}}`;
                                        lines.push(line);
                                    }}
                                    if (data.size_short > 0) {{
                                        let line = `   🔴 SHORT engine avg: $${{data.entry_short.toFixed(4)}} (USD: $${{data.short_notional ? data.short_notional.toFixed(2) : '0.00'}})`;
                                        if (ladderShortAvg > 0 && Math.abs(ladderShortAvg - data.entry_short) > 1e-8) line += ` | ladder avg: $${{ladderShortAvg.toFixed(4)}}`;
                                        lines.push(line);
                                    }}
                                    
                                    // რეალური RSI ტაიმფრეიმების ჩვენება
                                    let rsiStr = "   Active Rungs: " + data.ladders + " | Indicator Dossier: ";
                                    let rsiParts = [];
                                    if (data.rsi_map) {{
                                        // Sort timeframes for predictability
                                        const tfOrder = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "12h", "1d"];
                                        for (const tf of tfOrder) {{
                                            if (data.rsi_map[tf] !== undefined) {{
                                                rsiParts.push(`${{tf}}:${{data.rsi_map[tf]}}`);
                                            }}
                                        }}
                                    }}
                                    lines.push(rsiStr);
                                    if (rsiParts.length > 0) {{
                                         for (let i = 0; i < rsiParts.length; i += 3) {{
                                             lines.push("   " + rsiParts.slice(i, i + 3).join(" / "));
                                         }}
                                    }} else {{
                                        lines.push("   N/A Indicators");
                                    }}

                                    if (data.unrealized < -1000) {{
                                        lines.push(`   ⚠️ გაფრთხილება: ძალიან დიდი ღია ზარალი!`);
                                    }}
                                }}
                            }}
                            return lines;
                        }}
                    }}
                }}
            }},
            scales: {{ 
                x: {{ 
                    grid: {{ display: true, color: '#0d1117' }}, 
                    ticks: {{ color: '#6e7681', font: {{ size: 8, family: 'JetBrains Mono' }} }} 
                }}, 
                y: {{ 
                    grid: {{ color: '#0d1117', drawBorder: false }}, 
                    ticks: {{ color: '#6e7681', font: {{ size: 8, family: 'JetBrains Mono' }} }} 
                }} 
            }}
        }};

        const pairTapeOptions = {{
            ...chartDefs,
            interaction: {{ mode: 'index', intersect: false }},
            plugins: {{
                ...chartDefs.plugins,
                legend: {{ display: true, labels: {{ color: '#c9d1d9', font: {{ family: 'JetBrains Mono', size: 10 }} }} }},
                tooltip: {{
                    ...chartDefs.plugins.tooltip,
                    callbacks: {{
                        ...chartDefs.plugins.tooltip.callbacks,
                        afterBody: function(context) {{
                            const raw = context[0].raw || null;
                            let idx = context[0].dataIndex;
                            if (raw && raw.x) idx = labels.indexOf(raw.x);
                            const sym = document.getElementById('pairSelect').value;
                            const data = metadata[idx] && metadata[idx].symbols ? metadata[idx].symbols[sym] : null;
                            if (!data) return "";
                            const lines = ["", "--- Pair State ---"];
                            if (raw && raw.ladder) {{
                                const l = raw.ladder;
                                lines.push(`LADDER ${{l.side || ''}} ${{l.ladder_id || ''}}`);
                                lines.push(`Entry: $${{Number(l.entry || 0).toFixed(6)}} | TP: $${{Number(l.tp || 0).toFixed(6)}}`);
                                lines.push(`Size: ${{Number(l.size || 0).toFixed(6)}} | Entry USD: $${{Number(l.entry_notional || 0).toFixed(2)}} | Mark USD: $${{Number(l.mark_notional || 0).toFixed(2)}}`);
                                lines.push(`Profile: ${{l.profile_id || 'n/a'}} | TP Order: ${{l.has_tp ? 'YES' : 'NO'}}`);
                                lines.push("");
                            }}
                            lines.push(`Price: $${{Number(data.price || 0).toFixed(6)}}`);
                            lines.push(`LONG qty: ${{Number(data.size_long || 0).toFixed(6)}} @ $${{Number(data.entry_long || 0).toFixed(6)}}`);
                            lines.push(`SHORT qty: ${{Number(data.size_short || 0).toFixed(6)}} @ $${{Number(data.entry_short || 0).toFixed(6)}}`);
                            lines.push(`Long USD: $${{Number(data.long_notional || 0).toFixed(2)}} | Short USD: $${{Number(data.short_notional || 0).toFixed(2)}}`);
                            lines.push(`Ladders: ${{data.ladders || 0}} | UPNL: $${{Number(data.unrealized || 0).toFixed(2)}}`);
                            if (data.rsi_map) {{
                                const tfOrder = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "12h", "1d"];
                                lines.push(tfOrder.filter(tf => data.rsi_map[tf] !== undefined).map(tf => `${{tf}}:${{data.rsi_map[tf]}}`).join(" / "));
                            }}
                            return lines;
                        }}
                    }}
                }}
            }},
            scales: {{
                x: chartDefs.scales.x,
                y: {{
                    type: 'linear', position: 'left',
                    grid: {{ color: '#0d1117', drawBorder: false }},
                    ticks: {{ color: '#6e7681', font: {{ size: 8, family: 'JetBrains Mono' }} }}
                }},
                exposure: {{
                    type: 'linear', position: 'right',
                    grid: {{ drawOnChartArea: false }},
                    ticks: {{ color: '#6e7681', font: {{ size: 8, family: 'JetBrains Mono' }} }}
                }},
                pnl: {{
                    type: 'linear', position: 'right', display: false,
                    grid: {{ drawOnChartArea: false }}
                }}
            }}
        }};

        const pairTapeChart = new Chart(document.getElementById('pairTapeChart'), {{
            type: 'line',
            data: {{ labels: labels, datasets: [] }},
            options: pairTapeOptions
        }});

        function buildPairDataset(sym) {{
            const shortName = sym.split('/')[0];
            const price = metadata.map(m => m.symbols[sym] ? m.symbols[sym].price : null);
            const longEntry = metadata.map(m => {{
                const d = m.symbols[sym]; return d && d.entry_long > 0 ? d.entry_long : null;
            }});
            const shortEntry = metadata.map(m => {{
                const d = m.symbols[sym]; return d && d.entry_short > 0 ? d.entry_short : null;
            }});
            const longNotional = metadata.map(m => m.symbols[sym] ? m.symbols[sym].long_notional || 0 : 0);
            const shortNotional = metadata.map(m => m.symbols[sym] ? m.symbols[sym].short_notional || 0 : 0);
            const unrealized = metadata.map(m => m.symbols[sym] ? m.symbols[sym].unrealized || 0 : 0);
            const ladderPoints = (side, field) => {{
                const points = [];
                for (let i = 0; i < metadata.length; i++) {{
                    const d = metadata[i].symbols[sym];
                    const state = d && d.ladder_state ? d.ladder_state : null;
                    if (!state) continue;
                    const list = side === 'LONG' ? (state.long_ladders || []) : (state.short_ladders || []);
                    for (const ladder of list) {{
                        const y = Number(ladder[field] || 0);
                        if (y > 0) points.push({{ x: labels[i], y, ladder }});
                    }}
                }}
                return points;
            }};
            return [
                {{ label: `${{shortName}} Price`, data: price, yAxisID: 'y', borderColor: '#f0f6fc', borderWidth: 2.8, pointRadius: 0, tension: 0.05 }},
                {{ label: 'LONG Avg', data: longEntry, yAxisID: 'y', borderColor: '#3fb950', borderWidth: 2.0, borderDash: [6,4], pointRadius: 0, spanGaps: true, tension: 0.05 }},
                {{ label: 'SHORT Avg', data: shortEntry, yAxisID: 'y', borderColor: '#f85149', borderWidth: 2.0, borderDash: [6,4], pointRadius: 0, spanGaps: true, tension: 0.05 }},
                {{ label: 'Long USD', data: longNotional, yAxisID: 'exposure', borderColor: '#58a6ff', backgroundColor: 'rgba(88, 166, 255, 0.08)', borderWidth: 1.7, pointRadius: 0, fill: true, tension: 0.05 }},
                {{ label: 'Short USD', data: shortNotional, yAxisID: 'exposure', borderColor: '#ff7b72', backgroundColor: 'rgba(248, 81, 73, 0.08)', borderWidth: 1.7, pointRadius: 0, fill: true, tension: 0.05 }},
                {{ label: 'Unrealized PnL', data: unrealized, yAxisID: 'pnl', borderColor: '#db61a2', borderWidth: 1.5, borderDash: [2,3], pointRadius: 0, hidden: true, tension: 0.05 }},
                {{ label: 'LONG Ladder Entry', type: 'scatter', data: ladderPoints('LONG', 'entry'), yAxisID: 'y', borderColor: '#3fb950', backgroundColor: '#3fb950', pointStyle: 'triangle', pointRadius: 3.5, showLine: false }},
                {{ label: 'SHORT Ladder Entry', type: 'scatter', data: ladderPoints('SHORT', 'entry'), yAxisID: 'y', borderColor: '#f85149', backgroundColor: '#f85149', pointStyle: 'rectRot', pointRadius: 3.5, showLine: false }},
                {{ label: 'LONG TP', type: 'scatter', data: ladderPoints('LONG', 'tp'), yAxisID: 'y', borderColor: '#7ee787', backgroundColor: '#000', pointStyle: 'circle', pointRadius: 2.5, showLine: false }},
                {{ label: 'SHORT TP', type: 'scatter', data: ladderPoints('SHORT', 'tp'), yAxisID: 'y', borderColor: '#ffaaa5', backgroundColor: '#000', pointStyle: 'circle', pointRadius: 2.5, showLine: false }}
            ];
        }}

        function updatePairReadout(sym) {{
            const lastMeta = [...metadata].reverse().find(m => m.symbols && m.symbols[sym]);
            const d = lastMeta ? lastMeta.symbols[sym] : null;
            const box = document.getElementById('pairReadout');
            const rows = document.getElementById('pairLadderRows');
            if (!d) {{
                box.innerHTML = '';
                if (rows) rows.innerHTML = '';
                return;
            }}
            const state = d.ladder_state || {{}};
            const ladders = [...(state.long_ladders || []), ...(state.short_ladders || [])];
            box.innerHTML = `
                <div>Price<strong>$${{Number(d.price || 0).toFixed(6)}}</strong></div>
                <div>LONG Avg<strong>$${{Number(d.entry_long || 0).toFixed(6)}}</strong></div>
                <div>SHORT Avg<strong>$${{Number(d.entry_short || 0).toFixed(6)}}</strong></div>
                <div>Long USD<strong>$${{Number(d.long_notional || 0).toLocaleString(undefined, {{maximumFractionDigits: 0}})}}</strong></div>
                <div>Short USD<strong>$${{Number(d.short_notional || 0).toLocaleString(undefined, {{maximumFractionDigits: 0}})}}</strong></div>
                <div>UPNL<strong class="${{Number(d.unrealized || 0) >= 0 ? 'pos' : 'neg'}}">$${{Number(d.unrealized || 0).toFixed(2)}}</strong></div>
                <div>Ladders<strong>${{d.ladders || 0}}</strong></div>
                <div>Open Rungs<strong>${{ladders.length}}</strong></div>
            `;
            if (rows) {{
                rows.innerHTML = ladders.slice(-80).reverse().map(l => `
                    <tr>
                        <td class="${{l.side === 'LONG' ? 'pos' : 'neg'}}">${{l.side || ''}}</td>
                        <td>${{l.ladder_id || ''}}</td>
                        <td>$${{Number(l.entry || 0).toFixed(6)}}</td>
                        <td>$${{Number(l.tp || 0).toFixed(6)}}</td>
                        <td>${{Number(l.size || 0).toFixed(6)}}</td>
                        <td>$${{Number(l.entry_notional || 0).toFixed(2)}}</td>
                        <td>$${{Number(l.mark_notional || 0).toFixed(2)}}</td>
                        <td>${{(l.profile_id || '').slice(-8)}}</td>
                        <td>${{l.has_tp ? 'YES' : 'NO'}}</td>
                    </tr>
                `).join('');
            }}
        }}

        function updatePairTape(sym) {{
            pairTapeChart.data.datasets = buildPairDataset(sym);
            pairTapeChart.update();
            updatePairReadout(sym);
        }}

        document.getElementById('pairSelect').addEventListener('change', (event) => {{
            updatePairTape(event.target.value);
        }});
        updatePairTape(symbols[0]);

        new Chart(document.getElementById('equityChart'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{ label: 'Equity', data: {json.dumps(equity_vals)}, borderColor: '#3fb950', borderWidth: 4.5, pointRadius: 0, tension: 0.1 }},
                    {{ label: 'Balance', data: {json.dumps(balance_vals)}, borderColor: '#58a6ff', borderWidth: 2.0, borderDash: [5,5], pointRadius: 0, tension: 0.1 }},
                    {{ label: 'Realized PnL', data: {json.dumps(realized_vals)}, borderColor: '#a371f7', borderWidth: 2.5, borderDash: [2,2], pointRadius: 0, tension: 0.1, hidden: true }},
                    {{ label: 'Unrealized PnL', data: {json.dumps(unrealized_vals)}, borderColor: '#db61a2', borderWidth: 2.5, borderDash: [2,2], pointRadius: 0, tension: 0.1, hidden: true }}
                ]
            }},
            options: chartDefs
        }});

        new Chart(document.getElementById('symPnlChart'), {{
            type: 'line',
            data: {{ labels: labels, datasets: {json.dumps(sym_pnl_datasets)} }},
            options: chartDefs
        }});

        new Chart(document.getElementById('notionalChart'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [{{ label: 'Notional', data: {json.dumps(notional_vals)}, borderColor: '#d29922', borderWidth: 3, fill: true, backgroundColor: 'rgba(210, 153, 34, 0.4)', pointRadius: 0, tension: 0.1 }}]
            }},
            options: chartDefs
        }});

        new Chart(document.getElementById('ddChart'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [{{ label: 'Drawdown (%)', data: {json.dumps(drawdown_vals)}, borderColor: '#f85149', borderWidth: 3, fill: true, backgroundColor: 'rgba(248, 81, 73, 0.4)', pointRadius: 0, tension: 0.1 }}]
            }},
            options: chartDefs
        }});

        new Chart(document.getElementById('symSizeChart'), {{
            type: 'line',
            data: {{ labels: labels, datasets: {json.dumps(sym_pos_datasets)} }},
            options: chartDefs
        }});

        new Chart(document.getElementById('dailyPnlChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(daily_pnl_labels)},
                datasets: [{{
                    label: 'Daily PnL ($)',
                    data: {json.dumps(daily_pnl_values)},
                    backgroundColor: '#3fb950',
                    borderWidth: 0
                }}]
            }},
            options: chartDefs
        }});

        new Chart(document.getElementById('driftChart'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{ label: 'Drift (Bot - Engine)', data: {json.dumps(drift_vals)}, borderColor: '#8957e5', borderWidth: 4.5, fill: true, backgroundColor: 'rgba(137, 87, 229, 0.1)', pointRadius: 0, tension: 0.1 }}
                ]
            }},
            options: chartDefs
        }});

        new Chart(document.getElementById('sentimentChart'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{ label: 'Total Long Notional', data: {json.dumps(total_long_vals)}, borderColor: '#58a6ff', borderWidth: 3, pointRadius: 0, tension: 0.1 }},
                    {{ label: 'Total Short Notional', data: {json.dumps(total_short_vals)}, borderColor: '#f85149', borderWidth: 3, pointRadius: 0, tension: 0.1 }}
                ]
            }},
            options: chartDefs
        }});

        new Chart(document.getElementById('feeChart'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{ label: 'Cumulative Fees', data: {json.dumps(fee_vals)}, borderColor: '#d29922', borderWidth: 3, fill: true, backgroundColor: 'rgba(210, 153, 34, 0.1)', pointRadius: 0, tension: 0.1 }}
                ]
            }},
            options: chartDefs
        }});
    </script>
    <script src="report_adaptive_viewer.js?v=ema_macro4"></script>
</body>
</html>"""
        pair_ladder_asset_map = {}
        try:
            asset_dir = "backtest_pair_ladders"
            os.makedirs(asset_dir, exist_ok=True)
            for old_name in os.listdir(asset_dir):
                if old_name.endswith(".js"):
                    try:
                        os.remove(os.path.join(asset_dir, old_name))
                    except Exception:
                        pass
            for s in self.symbols:
                base = re.sub(r"[^A-Za-z0-9_]+", "_", s.split("/")[0])
                asset_name = f"pair_{base}.js"
                asset_path = os.path.join(asset_dir, asset_name)
                pair_ladder_asset_map[s] = asset_path.replace("\\", "/")
                payload = ladder_event_dump.get(s, [])
                with open(asset_path, "w", encoding="utf-8") as pf:
                    pf.write(
                        "window.backtest_ladder_events = window.backtest_ladder_events || {}; "
                        f"window.backtest_ladder_events[{json.dumps(s)}] = "
                        f"{json.dumps(payload, cls=NpEncoder, separators=(',', ':'), ensure_ascii=False)};\n"
                    )
        except Exception as ex:
            print(f"⚠️ Could not write pair ladder report assets: {ex}")

        # Write separate JavaScript file to avoid CORS blocks on file:// protocol
        data_js = f"""// Quantum Backtest Telemetry Dataset
window.backtest_labels = {json.dumps(labels)};
window.backtest_metadata = {json.dumps(metadata_dump, cls=NpEncoder)};
window.backtest_symbols = {json.dumps(self.symbols)};
window.backtest_equity_vals = {json.dumps(equity_vals)};
window.backtest_balance_vals = {json.dumps(balance_vals)};
window.backtest_realized_vals = {json.dumps(realized_vals)};
window.backtest_unrealized_vals = {json.dumps(unrealized_vals)};
window.backtest_notional_vals = {json.dumps(notional_vals)};
window.backtest_drawdown_vals = {json.dumps(drawdown_vals)};
window.backtest_drift_vals = {json.dumps(drift_vals)};
window.backtest_fee_vals = {json.dumps(fee_vals)};
window.backtest_total_long_vals = {json.dumps(total_long_vals)};
window.backtest_total_short_vals = {json.dumps(total_short_vals)};
window.backtest_sym_pnl_datasets = {json.dumps(sym_pnl_datasets)};
window.backtest_sym_pos_datasets = {json.dumps(sym_pos_datasets)};
window.backtest_daily_pnl_labels = {json.dumps(daily_pnl_labels)};
window.backtest_daily_pnl_values = {json.dumps(daily_pnl_values)};
window.backtest_ladder_events = {{}};
window.backtest_ladder_asset_map = {json.dumps(pair_ladder_asset_map)};
window.backtest_weekly_ema50 = {json.dumps(weekly_ema50_overlay, separators=(',', ':'), allow_nan=False)};
window.backtest_macro_switches = {json.dumps(macro_switch_overlay, separators=(',', ':'), allow_nan=False)};
"""
        with open("backtest_data.js", "w", encoding="utf-8") as f:
            f.write(data_js)
        print("✅ Telemetry Dataset generated: backtest_data.js")

        with open("backtest_report.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("✅ High-Fidelity Quantum Dossier generated: backtest_report.html")

        try:
            self.generate_inventory_report()
        except Exception as ex:
            print(f"⚠️ Could not generate inventory report: {ex}")


    def generate_inventory_report(self):
        """Write a compact diagnostics report for stuck final inventory."""
        state_path = "ladders_state_backtest.json"
        audit_path = "full_audit_log.jsonl"
        if not os.path.exists(state_path):
            print("⚠️ Inventory report skipped: ladders_state_backtest.json missing")
            return

        with open(state_path, "r", encoding="utf-8", errors="ignore") as f:
            state = json.load(f)

        pair_states = state.get("pair_states", {}) or {}
        final_ids = set()
        final_ladders = []
        for pair, ps in pair_states.items():
            for side, key in (
                ("LONG", "ladders_long"),
                ("SHORT", "ladders_short"),
                ("LONG", "full_ladders_long"),
                ("SHORT", "full_ladders_short"),
            ):
                for ladder in ps.get(key, []) or []:
                    ladder_id = str(ladder.get("ladder_id", ""))
                    if not ladder_id:
                        continue
                    final_ids.add((pair, side, ladder_id))
                    final_ladders.append((pair, side, ladder))

        origins = {}
        signals = {}
        stop_by_side = defaultdict(float)
        stop_by_pair = defaultdict(float)
        stop_rows = []
        tp_by_side = defaultdict(float)
        tp_count_by_side = defaultdict(int)
        if os.path.exists(audit_path):
            with open(audit_path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    event = ev.get("event")
                    pair = ev.get("pair")
                    side = ev.get("side")
                    try:
                        ts_ms = int(float(ev.get("ts_ms", 0) or 0))
                    except Exception:
                        ts_ms = 0

                    if event == "rsi_signal" and pair and side:
                        signals[(pair, ts_ms, side)] = ev
                    elif event == "order_fill" and pair and side:
                        ladder_id = str(ev.get("ladder_id", ""))
                        key = (pair, side, ladder_id)
                        if key in final_ids:
                            sig = signals.get((pair, ts_ms, side), {})
                            origins[key] = {
                                "ts_ms": ts_ms,
                                "phase": ev.get("weekly_ema_phase"),
                                "timeframe": sig.get("timeframe"),
                                "rsi": sig.get("rsi"),
                                "macro": sig.get("hybrid"),
                                "inverse": sig.get("inverse"),
                                "forced_inverse": sig.get("weekly_ema_forced_inverse"),
                            }
                    elif event == "weekly_ema50_stop_close" and pair and side:
                        amt = float(ev.get("realized_bot", 0.0) or 0.0)
                        stop_by_side[side] += amt
                        stop_by_pair[pair] += amt
                        stop_rows.append(ev)
                    elif event == "tp_fill" and side:
                        amt = float(ev.get("realized_bot", 0.0) or 0.0)
                        tp_by_side[side] += amt
                        tp_count_by_side[side] += 1

        now_ms = int(getattr(self, "current_timestamp_ms", 0) or 0)

        def fmt_money(v):
            return f"${v:,.2f}"

        def fmt_pct(v):
            return f"{v:+.2f}%"

        def pair_price(pair):
            try:
                px = self.get_latest_price(pair)
                return float(px) if px is not None else 0.0
            except Exception:
                return 0.0

        rows = []
        pair_summary = defaultdict(lambda: {
            "long_count": 0, "short_count": 0,
            "long_entry": 0.0, "short_entry": 0.0,
            "long_mark": 0.0, "short_mark": 0.0,
            "long_unreal": 0.0, "short_unreal": 0.0,
            "long_tp_left": 0.0, "short_tp_left": 0.0,
            "oldest_ms": None,
        })

        for pair, side, ladder in final_ladders:
            price = pair_price(pair)
            entry = float(ladder.get("entry_price", 0.0) or 0.0)
            size = float(ladder.get("size", 0.0) or 0.0)
            tp = float(ladder.get("tp", 0.0) or 0.0)
            entry_usd = float(ladder.get("usd_notional", 0.0) or (entry * size))
            mark_usd = price * size if price > 0 else 0.0
            if side == "LONG":
                unreal = (price - entry) * size if price > 0 else 0.0
                dist = ((price - entry) / entry * 100.0) if entry else 0.0
                tp_left = (tp - price) * size if price > 0 else 0.0
                bot_tp_if_hit = (tp - entry) * size
            else:
                unreal = (entry - price) * size if price > 0 else 0.0
                dist = ((entry - price) / entry * 100.0) if entry else 0.0
                tp_left = (price - tp) * size if price > 0 else 0.0
                bot_tp_if_hit = (entry - tp) * size

            key = (pair, side, str(ladder.get("ladder_id", "")))
            origin = origins.get(key, {})
            origin_ms = int(origin.get("ts_ms", 0) or 0)
            age_days = ((now_ms - origin_ms) / 86400000.0) if now_ms and origin_ms else 0.0
            opened = pd.to_datetime(origin_ms, unit="ms").strftime("%Y-%m-%d %H:%M") if origin_ms else ""

            summary = pair_summary[pair]
            prefix = "long" if side == "LONG" else "short"
            summary[f"{prefix}_count"] += 1
            summary[f"{prefix}_entry"] += entry_usd
            summary[f"{prefix}_mark"] += mark_usd
            summary[f"{prefix}_unreal"] += unreal
            summary[f"{prefix}_tp_left"] += tp_left
            if origin_ms and (summary["oldest_ms"] is None or origin_ms < summary["oldest_ms"]):
                summary["oldest_ms"] = origin_ms

            rows.append({
                "pair": pair,
                "side": side,
                "ladder_id": ladder.get("ladder_id", ""),
                "opened": opened,
                "age_days": age_days,
                "entry": entry,
                "price": price,
                "tp": tp,
                "entry_usd": entry_usd,
                "mark_usd": mark_usd,
                "unreal": unreal,
                "dist": dist,
                "tp_left": tp_left,
                "bot_tp_if_hit": bot_tp_if_hit,
                "phase": origin.get("phase", ""),
                "macro": origin.get("macro", ""),
                "tf": origin.get("timeframe", ""),
                "rsi": origin.get("rsi", ""),
                "inverse": origin.get("inverse", ""),
                "forced_inverse": origin.get("forced_inverse", ""),
            })

        rows.sort(key=lambda r: (r["unreal"], -r["age_days"]))
        worst_rows = rows[:80]

        total_long_count = sum(1 for r in rows if r["side"] == "LONG")
        total_short_count = sum(1 for r in rows if r["side"] == "SHORT")
        total_long_entry = sum(r["entry_usd"] for r in rows if r["side"] == "LONG")
        total_short_entry = sum(r["entry_usd"] for r in rows if r["side"] == "SHORT")
        total_unreal = sum(r["unreal"] for r in rows)
        long_unreal = sum(r["unreal"] for r in rows if r["side"] == "LONG")
        short_unreal = sum(r["unreal"] for r in rows if r["side"] == "SHORT")
        bot_long = float(state.get("total_profit_long", 0.0) or 0.0) + float(state.get("total_profit_long_full", 0.0) or 0.0)
        bot_short = float(state.get("total_profit_short", 0.0) or 0.0) + float(state.get("total_profit_short_full", 0.0) or 0.0)
        bot_total = bot_long + bot_short
        final_equity = float(self.equity_curve[-1]["equity"]) if getattr(self, "equity_curve", None) else 0.0
        final_wallet = float(self.equity_curve[-1].get("wallet", final_equity)) if getattr(self, "equity_curve", None) else 0.0
        max_dd = 0.0
        if getattr(self, "equity_curve", None):
            max_eq = float(self.start_balance)
            for point in self.equity_curve:
                eq = float(point.get("equity", 0.0) or 0.0)
                if eq > max_eq:
                    max_eq = eq
                dd = ((max_eq - eq) / max_eq * 100.0) if max_eq > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
        inventory_pain_ratio = (abs(total_unreal) / bot_total * 100.0) if bot_total > 0 else 0.0
        open_inventory_total = total_long_entry + total_short_entry

        def grouped_rows(group_key):
            grouped = defaultdict(lambda: {"count": 0, "entry": 0.0, "unreal": 0.0})
            for row in rows:
                key = row.get(group_key) or "UNKNOWN"
                grouped[key]["count"] += 1
                grouped[key]["entry"] += row["entry_usd"]
                grouped[key]["unreal"] += row["unreal"]
            return sorted(grouped.items(), key=lambda kv: abs(kv[1]["unreal"]), reverse=True)

        phase_breakdown = grouped_rows("phase")
        macro_breakdown = grouped_rows("macro")
        tf_breakdown = grouped_rows("tf")
        month_breakdown = defaultdict(lambda: {"count": 0, "entry": 0.0, "unreal": 0.0})
        for row in rows:
            month = row["opened"][:7] if row.get("opened") else "UNKNOWN"
            month_breakdown[month]["count"] += 1
            month_breakdown[month]["entry"] += row["entry_usd"]
            month_breakdown[month]["unreal"] += row["unreal"]
        month_breakdown = sorted(month_breakdown.items(), key=lambda kv: abs(kv[1]["unreal"]), reverse=True)

        def csv_name_for_pair(pair):
            base = str(pair).split("/")[0]
            return f"{base}_USDT_USDT_1m_1000d.csv"

        price_cache = {}

        def load_price_frame(pair):
            filename = csv_name_for_pair(pair)
            if filename not in price_cache:
                if not os.path.exists(filename):
                    price_cache[filename] = None
                else:
                    try:
                        price_cache[filename] = pd.read_csv(filename, usecols=["timestamp", "high", "low", "close"])
                    except Exception:
                        price_cache[filename] = None
            return price_cache[filename]

        def stop_ts_label(ev):
            try:
                ts = int(float(ev.get("ts_ms", 0) or 0))
                return datetime.datetime.utcfromtimestamp(ts / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return ""

        stop_recovery = defaultdict(lambda: {"count": 0, "stop": 0.0, "held_final": 0.0, "hit_avg": 0, "worse_final": 0})
        for ev in stop_rows:
            pair = ev.get("pair")
            side = ev.get("side")
            df = load_price_frame(pair)
            if df is None or not pair or not side:
                continue
            label = stop_ts_label(ev)
            if not label:
                continue
            try:
                avg = float(ev.get("avg_entry", ev.get("engine_avg_entry", 0.0)) or 0.0)
                qty = float(ev.get("qty_closed", ev.get("engine_qty_before", 0.0)) or 0.0)
                realized = float(ev.get("realized_bot", 0.0) or 0.0)
            except Exception:
                continue
            future = df[df["timestamp"] > label]
            if future.empty or avg <= 0 or qty <= 0:
                continue
            if side == "LONG":
                hit_avg = bool((future["high"] >= avg).any())
                held_final = (float(future["close"].iloc[-1]) - avg) * qty
            else:
                hit_avg = bool((future["low"] <= avg).any())
                held_final = (avg - float(future["close"].iloc[-1])) * qty
            stats = stop_recovery[(side, pair)]
            stats["count"] += 1
            stats["stop"] += realized
            stats["held_final"] += held_final
            stats["hit_avg"] += 1 if hit_avg else 0
            stats["worse_final"] += 1 if held_final < realized else 0

        stop_recovery_total = defaultdict(float)
        for stats in stop_recovery.values():
            for key, value in stats.items():
                stop_recovery_total[key] += value

        engine_inventory = []
        engine_unreal_total = 0.0
        try:
            for pair, pos_dict in self.positions.items():
                price = pair_price(pair)
                for side in ("LONG", "SHORT"):
                    pos = pos_dict.get(side, {}) or {}
                    size = float(pos.get("size", 0.0) or 0.0)
                    if size <= 0 or price <= 0:
                        continue
                    avg = float(pos.get("entryPrice", 0.0) or 0.0)
                    notional = price * size
                    unreal = (price - avg) * size if side == "LONG" else (avg - price) * size
                    engine_unreal_total += unreal
                    ladder_side = [r for r in rows if r["pair"] == pair and r["side"] == side]
                    ladder_unreal = sum(r["unreal"] for r in ladder_side)
                    ladder_entry = sum(r["entry_usd"] for r in ladder_side)
                    engine_inventory.append({
                        "pair": pair,
                        "side": side,
                        "size": size,
                        "avg": avg,
                        "price": price,
                        "notional": notional,
                        "engine_unreal": unreal,
                        "ladder_unreal": ladder_unreal,
                        "ladder_entry": ladder_entry,
                        "ladder_count": len(ladder_side),
                    })
        except Exception:
            engine_inventory = []
        engine_inventory.sort(key=lambda r: r["engine_unreal"])

        def e(value):
            return html_lib.escape(str(value))

        def inventory_grade():
            if final_equity <= 0:
                return "BROKEN"
            if inventory_pain_ratio >= 35 or max_dd >= 60:
                return "HIGH RISK"
            if inventory_pain_ratio >= 18 or max_dd >= 35:
                return "WATCH"
            return "OK"

        def table_rows_breakdown(items, label):
            parts = []
            for key, stats in items:
                parts.append(
                    "<tr>"
                    f"<td>{e(key)}</td><td>{stats['count']}</td>"
                    f"<td>{fmt_money(stats['entry'])}</td>"
                    f"<td class='{ 'bad' if stats['unreal'] < 0 else 'good' }'>{fmt_money(stats['unreal'])}</td>"
                    f"<td>{e(label)}</td>"
                    "</tr>"
                )
            return "\n".join(parts)

        def pair_verdict(pair, s):
            long_un = s["long_unreal"]
            short_un = s["short_unreal"]
            long_count = s["long_count"]
            short_count = s["short_count"]
            if long_un < short_un and long_count:
                side = "LONG"
                issue = "old/high LONG bag"
            elif short_count:
                side = "SHORT"
                issue = "old/low SHORT bag"
            else:
                side = "MIXED"
                issue = "small or balanced inventory"
            oldest = pd.to_datetime(s["oldest_ms"], unit="ms").strftime("%Y-%m-%d") if s["oldest_ms"] else ""
            total_pair_unreal = long_un + short_un
            if total_pair_unreal < -500:
                severity = "heavy"
            elif total_pair_unreal < -150:
                severity = "medium"
            elif total_pair_unreal < 0:
                severity = "light"
            else:
                severity = "ok"
            return {
                "pair": pair,
                "side": side,
                "issue": issue,
                "severity": severity,
                "oldest": oldest,
                "unreal": total_pair_unreal,
            }

        def table_rows_pair():
            parts = []
            for pair, s in sorted(pair_summary.items(), key=lambda kv: (kv[1]["long_unreal"] + kv[1]["short_unreal"])):
                oldest = pd.to_datetime(s["oldest_ms"], unit="ms").strftime("%Y-%m-%d") if s["oldest_ms"] else ""
                parts.append(
                    "<tr>"
                    f"<td>{e(pair)}</td><td>{s['long_count']}</td><td>{s['short_count']}</td>"
                    f"<td>{fmt_money(s['long_entry'])}</td><td>{fmt_money(s['short_entry'])}</td>"
                    f"<td class='{ 'bad' if s['long_unreal'] < 0 else 'good' }'>{fmt_money(s['long_unreal'])}</td>"
                    f"<td class='{ 'bad' if s['short_unreal'] < 0 else 'good' }'>{fmt_money(s['short_unreal'])}</td>"
                    f"<td>{fmt_money(s['long_tp_left'])}</td><td>{fmt_money(s['short_tp_left'])}</td>"
                    f"<td>{e(oldest)}</td>"
                    "</tr>"
                )
            return "\n".join(parts)

        def table_rows_verdict():
            parts = []
            verdicts = [pair_verdict(pair, s) for pair, s in pair_summary.items()]
            verdicts.sort(key=lambda x: x["unreal"])
            for v in verdicts:
                parts.append(
                    "<tr>"
                    f"<td>{e(v['pair'])}</td><td>{e(v['severity'])}</td><td>{e(v['side'])}</td>"
                    f"<td>{e(v['issue'])}</td><td>{e(v['oldest'])}</td>"
                    f"<td class='{ 'bad' if v['unreal'] < 0 else 'good' }'>{fmt_money(v['unreal'])}</td>"
                    "</tr>"
                )
            return "\n".join(parts)

        def table_rows_worst():
            parts = []
            for r in worst_rows:
                rsi = "" if r["rsi"] in ("", None) else f"{float(r['rsi']):.2f}"
                parts.append(
                    "<tr>"
                    f"<td>{e(r['pair'])}</td><td>{e(r['side'])}</td><td>{e(r['ladder_id'])}</td>"
                    f"<td>{e(r['opened'])}</td><td>{r['age_days']:.1f}</td>"
                    f"<td>{r['entry']:.8g}</td><td>{r['price']:.8g}</td><td>{r['tp']:.8g}</td>"
                    f"<td>{fmt_money(r['entry_usd'])}</td><td>{fmt_money(r['mark_usd'])}</td>"
                    f"<td class='{ 'bad' if r['unreal'] < 0 else 'good' }'>{fmt_money(r['unreal'])}</td>"
                    f"<td class='{ 'bad' if r['dist'] < 0 else 'good' }'>{fmt_pct(r['dist'])}</td>"
                    f"<td>{fmt_money(r['bot_tp_if_hit'])}</td><td>{fmt_money(r['tp_left'])}</td>"
                    f"<td>{e(r['phase'])}</td><td>{e(r['macro'])}</td><td>{e(r['tf'])}</td><td>{e(rsi)}</td>"
                    "</tr>"
                )
            return "\n".join(parts)

        def table_rows_stop_recovery():
            parts = []
            for (side, pair), stats in sorted(stop_recovery.items(), key=lambda kv: kv[1]["stop"]):
                delta = stats["held_final"] - stats["stop"]
                parts.append(
                    "<tr>"
                    f"<td>{e(pair)}</td><td>{e(side)}</td><td>{int(stats['count'])}</td>"
                    f"<td class='{ 'bad' if stats['stop'] < 0 else 'good' }'>{fmt_money(stats['stop'])}</td>"
                    f"<td class='{ 'bad' if stats['held_final'] < 0 else 'good' }'>{fmt_money(stats['held_final'])}</td>"
                    f"<td class='{ 'bad' if delta < 0 else 'good' }'>{fmt_money(delta)}</td>"
                    f"<td>{int(stats['hit_avg'])}/{int(stats['count'])}</td>"
                    f"<td>{int(stats['worse_final'])}/{int(stats['count'])}</td>"
                    "</tr>"
                )
            return "\n".join(parts)

        def table_rows_engine_inventory():
            parts = []
            for r in engine_inventory:
                diff = r["ladder_unreal"] - r["engine_unreal"]
                parts.append(
                    "<tr>"
                    f"<td>{e(r['pair'])}</td><td>{e(r['side'])}</td><td>{r['size']:.8g}</td>"
                    f"<td>{r['avg']:.8g}</td><td>{r['price']:.8g}</td><td>{fmt_money(r['notional'])}</td>"
                    f"<td class='{ 'bad' if r['engine_unreal'] < 0 else 'good' }'>{fmt_money(r['engine_unreal'])}</td>"
                    f"<td>{int(r['ladder_count'])}</td><td>{fmt_money(r['ladder_entry'])}</td>"
                    f"<td class='{ 'bad' if r['ladder_unreal'] < 0 else 'good' }'>{fmt_money(r['ladder_unreal'])}</td>"
                    f"<td class='{ 'bad' if diff < 0 else 'good' }'>{fmt_money(diff)}</td>"
                    "</tr>"
                )
            return "\n".join(parts)

        def table_rows_tp_stop_pair():
            all_pairs = sorted(set(list(stop_by_pair.keys()) + [r["pair"] for r in rows]))
            parts = []
            for pair in all_pairs:
                s = pair_summary.get(pair, {})
                open_unreal = float(s.get("long_unreal", 0.0) or 0.0) + float(s.get("short_unreal", 0.0) or 0.0)
                parts.append(
                    "<tr>"
                    f"<td>{e(pair)}</td>"
                    f"<td class='{ 'bad' if stop_by_pair[pair] < 0 else 'good' }'>{fmt_money(stop_by_pair[pair])}</td>"
                    f"<td class='{ 'bad' if open_unreal < 0 else 'good' }'>{fmt_money(open_unreal)}</td>"
                    f"<td>{int(s.get('long_count', 0) or 0)}</td><td>{int(s.get('short_count', 0) or 0)}</td>"
                    "</tr>"
                )
            return "\n".join(parts)

        generated_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        html = f"""<!doctype html>
<html lang="ka">
<head>
  <meta charset="utf-8">
  <title>Backtest Inventory Diagnostics</title>
  <style>
    body {{ margin: 0; background: #05070a; color: #d8e2ef; font: 13px/1.45 Consolas, monospace; }}
    header {{ padding: 22px 28px; border-bottom: 1px solid #1b2635; }}
    h1 {{ margin: 0 0 6px; font-size: 22px; }}
    main {{ padding: 20px 28px 40px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 20px; }}
    .card {{ border: 1px solid #1f2d3f; background: #0b1017; border-radius: 8px; padding: 12px; }}
    .label {{ color: #7f93aa; font-size: 11px; text-transform: uppercase; }}
    .value {{ font-size: 20px; margin-top: 4px; color: #f3f7fb; }}
    .grade {{ font-size: 24px; letter-spacing: 0; }}
    .good {{ color: #52d273; }}
    .bad {{ color: #ff6b6b; }}
    .warn {{ color: #f2c94c; }}
    section {{ margin-top: 24px; }}
    h2 {{ font-size: 16px; margin: 0 0 10px; color: #f3f7fb; }}
    table {{ width: 100%; border-collapse: collapse; background: #080d13; border: 1px solid #1f2d3f; }}
    th, td {{ padding: 7px 8px; border-bottom: 1px solid #142030; white-space: nowrap; text-align: right; }}
    th:first-child, td:first-child, td:nth-child(3), td:nth-child(4), td:nth-child(15), td:nth-child(16), td:nth-child(17) {{ text-align: left; }}
    th {{ color: #8da3bb; position: sticky; top: 0; background: #0d1420; z-index: 1; }}
    .note {{ color: #99aabe; max-width: 980px; }}
    .table-wrap {{ overflow: auto; max-height: 680px; border: 1px solid #1f2d3f; }}
  </style>
</head>
<body>
<header>
  <h1>Inventory Diagnostics</h1>
  <div class="note">Generated {e(generated_at)}. ცალკე ანგარიში final open ladders-ისთვის: age, entry distance, origin phase/macro, და ყველაზე მძიმე დარჩენილი inventory.</div>
</header>
<main>
  <div class="cards">
    <div class="card"><div class="label">Bot LONG Profit</div><div class="value good">{fmt_money(bot_long)}</div></div>
    <div class="card"><div class="label">Bot SHORT Profit</div><div class="value good">{fmt_money(bot_short)}</div></div>
    <div class="card"><div class="label">Final Equity</div><div class="value {'bad' if final_equity < float(self.start_balance) else 'good'}">{fmt_money(final_equity)}</div></div>
    <div class="card"><div class="label">Final Wallet</div><div class="value">{fmt_money(final_wallet)}</div></div>
    <div class="card"><div class="label">Max Drawdown</div><div class="value bad">{max_dd:.2f}%</div></div>
    <div class="card"><div class="label">Open LONG Ladders</div><div class="value">{total_long_count} / {fmt_money(total_long_entry)}</div></div>
    <div class="card"><div class="label">Open SHORT Ladders</div><div class="value">{total_short_count} / {fmt_money(total_short_entry)}</div></div>
    <div class="card"><div class="label">Open Inventory</div><div class="value">{fmt_money(open_inventory_total)}</div></div>
    <div class="card"><div class="label">Open Unrealized Total</div><div class="value {'bad' if total_unreal < 0 else 'good'}">{fmt_money(total_unreal)}</div></div>
    <div class="card"><div class="label">Inventory Pain</div><div class="value {'bad' if inventory_pain_ratio >= 35 else 'warn' if inventory_pain_ratio >= 18 else 'good'}">{inventory_pain_ratio:.1f}%</div></div>
    <div class="card"><div class="label">Run Grade</div><div class="value grade {'bad' if inventory_grade() in ('BROKEN', 'HIGH RISK') else 'warn' if inventory_grade() == 'WATCH' else 'good'}">{inventory_grade()}</div></div>
    <div class="card"><div class="label">LONG / SHORT Unrealized</div><div class="value"><span class="{'bad' if long_unreal < 0 else 'good'}">{fmt_money(long_unreal)}</span> / <span class="{'bad' if short_unreal < 0 else 'good'}">{fmt_money(short_unreal)}</span></div></div>
    <div class="card"><div class="label">TP Closed LONG / SHORT</div><div class="value">{fmt_money(tp_by_side['LONG'])} / {fmt_money(tp_by_side['SHORT'])}</div></div>
    <div class="card"><div class="label">Weekly Stop LONG</div><div class="value {'bad' if stop_by_side['LONG'] < 0 else 'good'}">{fmt_money(stop_by_side['LONG'])}</div></div>
    <div class="card"><div class="label">Weekly Stop SHORT</div><div class="value {'bad' if stop_by_side['SHORT'] < 0 else 'good'}">{fmt_money(stop_by_side['SHORT'])}</div></div>
    <div class="card"><div class="label">Stop Held-Final What-if</div><div class="value {'bad' if stop_recovery_total['held_final'] < 0 else 'good'}">{fmt_money(stop_recovery_total['held_final'])}</div></div>
    <div class="card"><div class="label">Stop Later Hit Avg</div><div class="value">{int(stop_recovery_total['hit_avg'])} / {int(stop_recovery_total['count'])}</div></div>
    <div class="card"><div class="label">Engine Open UPNL</div><div class="value {'bad' if engine_unreal_total < 0 else 'good'}">{fmt_money(engine_unreal_total)}</div></div>
  </div>

  <section>
    <h2>Stop Recovery / What-if</h2>
    <div class="note">Conservative check: stop-ზე დახურული პოზიცია რომ არ დახურულიყო, მერე avg entry-ს შეეხო თუ არა და ბოლომდე დატოვების შედეგი რა იქნებოდა. ეს ზუსტი TP simulation არ არის, მაგრამ stop ადრე ჭრის თუ არა სწრაფად აჩვენებს.</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Pair</th><th>Side</th><th>Stops</th><th>Stop PNL</th><th>Held To Final</th><th>Delta</th><th>Later Hit Avg</th><th>Final Worse</th></tr></thead>
        <tbody>{table_rows_stop_recovery()}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Engine Avg vs Ladder Ledger</h2>
    <div class="note">Engine UPNL არის რეალური hedge position avg-ით. Ladder ledger აჩვენებს ინდივიდუალური ლადერების ტკივილს. ეს ორი შეგნებულად ცალკეა, რომ არ აგერიოს რეალური equity და ladder inventory.</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Pair</th><th>Side</th><th>Size</th><th>Engine Avg</th><th>Price</th><th>Notional</th><th>Engine UPNL</th><th>Ladders</th><th>Ladder Entry $</th><th>Ladder UPNL</th><th>Diff</th></tr></thead>
        <tbody>{table_rows_engine_inventory()}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Stop / Open Pain By Pair</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Pair</th><th>Weekly Stop PNL</th><th>Open Ladder UPNL</th><th>Open LONG #</th><th>Open SHORT #</th></tr></thead>
        <tbody>{table_rows_tp_stop_pair()}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Worst Pair Diagnosis</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Pair</th><th>Severity</th><th>Main Side</th><th>Issue</th><th>Oldest Open</th><th>Total UPNL</th></tr></thead>
        <tbody>{table_rows_verdict()}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Origin Breakdown</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Bucket</th><th>Open #</th><th>Entry $</th><th>UPNL</th><th>Type</th></tr></thead>
        <tbody>
          {table_rows_breakdown(phase_breakdown, 'phase')}
          {table_rows_breakdown(macro_breakdown, 'macro')}
          {table_rows_breakdown(tf_breakdown, 'timeframe')}
          {table_rows_breakdown(month_breakdown, 'month')}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Pair Inventory Summary</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Pair</th><th>LONG #</th><th>SHORT #</th><th>LONG Entry $</th><th>SHORT Entry $</th><th>LONG UPNL</th><th>SHORT UPNL</th><th>LONG To TP $</th><th>SHORT To TP $</th><th>Oldest</th></tr></thead>
        <tbody>{table_rows_pair()}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Worst Open Ladders</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Pair</th><th>Side</th><th>ID</th><th>Opened</th><th>Age d</th><th>Entry</th><th>Price</th><th>TP</th><th>Entry $</th><th>Mark $</th><th>UPNL</th><th>Dist</th><th>Bot TP if Hit</th><th>Move To TP</th><th>Phase</th><th>Macro</th><th>TF</th><th>RSI</th></tr></thead>
        <tbody>{table_rows_worst()}</tbody>
      </table>
    </div>
  </section>
</main>
</body>
</html>"""

        with open("backtest_inventory_report.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("✅ Inventory Diagnostics generated: backtest_inventory_report.html")


    @property
    def current_timestamp(self):
        # Only convert if timestamp changed
        if not hasattr(self, '_cached_dt_ms') or self._cached_dt_ms != self.current_timestamp_ms:
            self._cached_dt = pd.to_datetime(self.current_timestamp_ms, unit='ms')
            self._cached_dt_ms = self.current_timestamp_ms
        return self._cached_dt

    def _update_pnl(self):
        if self.unrealized_pnl_cache is not None:
            return self.unrealized_pnl_cache
        total_pnl = 0.0
        for symbol, pos_dict in self.positions.items():
            current_price = self.get_latest_price(symbol)
            if current_price is None: continue
            for side, pos in pos_dict.items():
                if pos["size"] > 0:
                    pnl = (current_price - pos["entryPrice"]) * pos["size"] if side == "LONG" else (pos["entryPrice"] - current_price) * pos["size"]
                    pos["unrealizedPnl"] = pnl
                    total_pnl += pnl
        self.unrealized_pnl_cache = total_pnl
        return total_pnl

    def _process_open_orders(self):
        # Optimized: group active orders once per tick. The old flow rebuilt the
        # same per-symbol list by scanning all active orders for every symbol.
        orders_by_symbol = defaultdict(list)
        for order in self.active_open_orders.values():
            orders_by_symbol[order["symbol"]].append(order)
        
        for symbol, symbol_orders in orders_by_symbol.items():
            if not self.is_symbol_active(symbol):
                self._cancel_resting_orders_for_symbol(symbol)
                continue
            idx = self._get_tf_idx(symbol, '1m')
            ohlc = self._get_1m_candle_ohlc(symbol, idx)
            if ohlc is None:
                self.record_audit(
                    "order_sampling_missing_ohlc",
                    {
                        "symbol": symbol,
                        "tf": "1m",
                        "idx": idx,
                        "active_orders": len(symbol_orders),
                    },
                )
                self.flush_audit(force=True)
                raise RuntimeError(
                    f"Active orders for {symbol} were not checked because 1m OHLC is missing at idx={idx}, ts={self.current_timestamp}."
                )
            open_px, high_px, low_px = ohlc
            current_px = self.get_latest_price(symbol)

            # Sort orders: Process Limit/Entry orders (increase position) BEFORE ReduceOnly/TP orders.
            # Accurately detect both explicit reduceOnly and implicit close orders.
            def get_order_sort_key(o):
                # Process Limit/Entry (increase position) orders FIRST, and ReduceOnly/TP orders AFTER.
                ctx = self._order_position_context(symbol, o.get("side"), o.get("params", {}) or {})
                return 1 if ctx["is_reduce"] else 0

            symbol_orders.sort(key=get_order_sort_key)
            
            for order in symbol_orders:
                trigger = False
                prev_checked_idx = order.get("_last_checked_1m_idx")
                if prev_checked_idx is not None and idx > int(prev_checked_idx) + 1:
                    self.record_audit(
                        "order_sampling_gap",
                        {
                            "symbol": symbol,
                            "order_id": order.get("id"),
                            "side": order.get("side"),
                            "price": order.get("price"),
                            "prev_idx": int(prev_checked_idx),
                            "idx": int(idx),
                            "missed_candles": int(idx) - int(prev_checked_idx) - 1,
                        },
                    )
                    self.flush_audit(force=True)
                    raise RuntimeError(
                        f"Active order {order.get('id')} for {symbol} skipped 1m checks: prev_idx={prev_checked_idx}, idx={idx}."
                    )
                order["_last_checked_1m_idx"] = int(idx)
                order["_last_checked_ts_ms"] = self.current_timestamp_ms
                try:
                    limit_price = float(order["price"])
                except (TypeError, ValueError, KeyError):
                    continue
                exec_price = limit_price
                order_side_lower = str(order.get("side", "")).lower()
                if order_side_lower == "buy":
                    if low_px <= limit_price:
                        trigger = True
                        exec_price = limit_price
                elif order_side_lower == "sell":
                    if high_px >= limit_price:
                        trigger = True
                        exec_price = limit_price
                
                if trigger:
                    # Remove from active first to release reserved margin check for executions
                    if order["id"] in self.active_open_orders:
                        del self.active_open_orders[order["id"]]
                        
                    try:
                        exec_qty = self._execute_order(symbol, order["side"], order["amount"], exec_price, order["params"], order_type="limit", is_resting_trigger=True)
                        if exec_qty > 0:
                            order["filled"] += exec_qty
                            self._finalize_order_fill_status(order, symbol, exec_price)
                            if order["status"] == "open":
                                self.active_open_orders[order["id"]] = order
                        else:
                            order["status"] = "canceled"
                            order["remaining"] = order["amount"]
                    except ccxt.InsufficientFunds as e:
                        print(f"⚠️ Limit order execution failed (Insufficient Margin). Order ID: {order['id']}, Symbol: {symbol}. Keeping order open.")
                        # Restore order status to open and re-add to active_open_orders instead of canceling
                        order["status"] = "open"
                        self.active_open_orders[order["id"]] = order
                    except Exception as e:
                        print(f"⚠️ Limit order execution unexpected error: {e}. Order ID: {order['id']}, Symbol: {symbol}. Canceling order.")
                        order["status"] = "canceled"
                        
                    self.needs_sync = True

    def _execute_order_leg(self, symbol, side, pos_key, is_closing, amount, price, params, order_type="market", reduce_only=False, pos_idx=0):
        pos = self.positions[symbol][pos_key]
        closes_position = (pos_key == "LONG" and str(side).lower() == "sell") or (pos_key == "SHORT" and str(side).lower() == "buy")

        # A malformed reduceOnly order must never open/increase exposure or run margin checks.
        if reduce_only and not closes_position:
            self.record_audit(
                "reduce_only_noop",
                {
                    "symbol": symbol,
                    "side": pos_key,
                    "order_side": str(side).lower(),
                    "amount": amount,
                    "price": price,
                    "order_type": order_type,
                    "positionIdx": pos_idx,
                },
            )
            return 0.0
        
        if not is_closing:
            mark_px = self._mark_price(symbol) or price
            incremental_margin_needed = self._incremental_im_for_order(symbol, pos_key, amount, mark_px)
            free_margin = self._get_free_margin()
            if incremental_margin_needed > 0 and free_margin < incremental_margin_needed:
                margin_breakdown = self._margin_breakdown()
                wallet = float(self.balance["USDT"]["total"])
                unrealized = float(self._update_pnl())
                self.record_audit(
                    "order_reject",
                    {
                        "symbol": symbol,
                        "side": pos_key,
                        "reason": "insufficient_margin",
                        "required_margin": incremental_margin_needed,
                        "free_margin": free_margin,
                        "wallet": wallet,
                        "unrealized": unrealized,
                        "equity": wallet + unrealized,
                        "used_margin": margin_breakdown.get("used_margin"),
                        "position_margin": margin_breakdown.get("position_margin"),
                        "open_order_margin": margin_breakdown.get("open_order_margin"),
                        "top_margin_symbols": margin_breakdown.get("top_symbols"),
                        "order_side": str(side).lower(),
                        "amount": amount,
                        "price": price,
                        "order_type": order_type
                    },
                )
                raise ccxt.InsufficientFunds(
                    f"Bybit error: Insufficient available balance. Required margin: {incremental_margin_needed:.2f}, Free margin: {free_margin:.2f}"
                )
        
        # Calculate effectively executed quantity
        executed_qty = amount if not is_closing else min(amount, pos["size"])
        if is_closing and executed_qty > 0:
            remaining_pos = max(0.0, float(pos["size"]) - float(executed_qty))
            if remaining_pos <= self._amount_dust_threshold(symbol):
                executed_qty = float(pos["size"])
        if executed_qty <= 0 and is_closing:
            return 0.0 # Nothing to close
            
        fee_rate = self.MAKER_FEE_RATE if str(order_type).lower() == "limit" else self.TAKER_FEE_RATE
        fee = (executed_qty * price) * fee_rate
        self.balance["USDT"]["total"] -= fee
        self.total_fees_paid += fee
        self.symbol_fees[symbol] += fee  # ტრეკინგი კონკრეტულ პაირზე
        self.total_volume += (executed_qty * price)
        
        if not is_closing:
            new_size = pos["size"] + amount
            pos["entryPrice"] = ((pos["entryPrice"] * pos["size"]) + (amount * price)) / new_size if new_size > 0 else 0
            pos["size"] = new_size
            # Subtract opening fee from realized PnL stats to match wallet exactly
            self.realized_pnl_by_symbol[symbol] -= fee
            
            if getattr(self, "verbose_trade_console", False):
                dt_str = str(pd.to_datetime(self.current_timestamp_ms, unit='ms')) if hasattr(self, 'current_timestamp_ms') else ""
                print(
                    f"🪜 [FILL-ENTRY] {dt_str} | {symbol} | {pos_key} | {order_type.upper()} "
                    f"filled Qty: {amount:.4f} @ ${price:.4f} | "
                    f"New Size: {new_size:.4f} | New Avg Entry: ${pos['entryPrice']:.4f} | "
                    f"Fee: ${fee:.4f}",
                    flush=True
                )
            
            self.record_audit(
                "position_open",
                {
                    "symbol": symbol,
                    "side": pos_key,
                    "qty": amount,
                    "price": price,
                    "new_size": new_size,
                    "entry_price": pos["entryPrice"],
                    "fee": fee,
                    "wallet": self.balance["USDT"]["total"],
                    "order_type": order_type,
                    "positionIdx": pos_idx,
                },
            )
        else:
            close_qty = executed_qty
            avg_entry = float(pos["entryPrice"] or 0.0)
            old_size = float(pos["size"] or 0.0)
            # Exchange accounting must stay Bybit-like: a reduce/TP fill realizes
            # against the exchange position average. The bot may track per-ladder
            # profit separately, but that metadata must not mutate wallet equity
            # or the exchange-side average entry used for unrealized PnL/margin.
            pnl_position_avg = (price - avg_entry) * close_qty if pos_key == "LONG" else (avg_entry - price) * close_qty
            pnl = pnl_position_avg
            pnl_basis = "position_avg"
            pnl_rung = None
            rung_entry = None
            if params:
                try:
                    rung_entry = float(params.get("ladder_entry_price") or 0) or None
                except (TypeError, ValueError):
                    rung_entry = None
            if rung_entry and rung_entry > 0:
                pnl_rung = (
                    (price - rung_entry) * close_qty
                    if pos_key == "LONG"
                    else (rung_entry - price) * close_qty
                )
            pos["size"] -= close_qty
            self.balance["USDT"]["total"] += pnl
            unrealized_after = self._update_pnl()
            equity_after = self.balance["USDT"]["total"] + unrealized_after
            
            pnl_info = f"PosAvg={avg_entry:.4f} | RungEntry={rung_entry if rung_entry else 'None'} | Exit={price:.4f}"
            if pnl > 0:
                audit_msg = f"{C_GREEN}✔ TRADE_WIN{C_RESET} | {symbol} | Gain: ${pnl:.2f} ({pnl_info})"
            else:
                audit_msg = f"{C_RED}✘ TRADE_EXIT{C_RESET} | {symbol} | Loss: ${pnl:.2f} ({pnl_info})"
            
            self.realized_pnl_by_symbol[symbol] += (pnl - fee)
            self.gross_pnl_by_symbol[symbol] += pnl
            if getattr(self, "verbose_trade_console", False):
                print(audit_msg, flush=True)
                
            self.record_audit(
                "position_close",
                {
                    "symbol": symbol,
                    "side": pos_key,
                    "qty_requested": amount,
                    "qty_closed": close_qty,
                    "price": price,
                    "entry_price": avg_entry,
                    "entry_price_after": pos["entryPrice"],
                    "ladder_entry_price": rung_entry,
                    "pnl": pnl,
                    "pnl_net": pnl - fee,
                    "pnl_basis": pnl_basis,
                    "pnl_position_avg": pnl_position_avg,
                    "pnl_rung": pnl_rung,
                    "fee": fee,
                    "wallet": self.balance["USDT"]["total"],
                    "unrealized": unrealized_after,
                    "equity": equity_after,
                    "order_type": order_type,
                    "reduce_only": reduce_only,
                    "positionIdx": pos_idx,
                },
            )
            
            # Reset entry price and size to 0 if position is fully closed
            if pos["size"] <= self._amount_dust_threshold(symbol):
                pos["size"] = 0.0
                pos["entryPrice"] = 0.0
                pos["unrealizedPnl"] = 0.0
                
            # Store realized trade
            if close_qty > 0:
                self.trade_log.append({
                    "symbol": symbol,
                    "side": pos_key,
                    "pnl": pnl - fee,
                    "qty": close_qty,
                    "exit_price": price,
                    "entry_price": avg_entry,
                    "notional": close_qty * price,
                    "pnl_pct": ((price - avg_entry) / avg_entry * 100.0 if pos_key == "LONG" else (avg_entry - price) / avg_entry * 100.0) if avg_entry > 0 else 0.0,
                    "ts": self.current_timestamp_ms,
                })

        return executed_qty

    def _execute_order(self, symbol, side, amount, price, params, order_type="market", is_resting_trigger=False):
        symbol = self._normalize_to_ccxt(symbol)
        side_lower = str(side).lower()
        
        # Apply slippage for market orders to simulate bid-ask spread and execution slippage
        slippage_rate = self.config.get("slippage_rate", 0.0002)
        if str(order_type).lower() == "market" and slippage_rate > 0:
            if side_lower == "buy":
                price = price * (1.0 + slippage_rate)
            else:
                price = price * (1.0 - slippage_rate)

        # Immediate cache invalidation to prevent stale balance/PnL reads
        self.balance_cache = None
        self.unrealized_pnl_cache = None
        
        if not self.is_hedge_mode:
            # One-Way Mode Netting Logic
            params = params or {}
            reduce_only = bool(params.get("reduceOnly", False))
            
            long_sz = self.positions[symbol]["LONG"]["size"]
            short_sz = self.positions[symbol]["SHORT"]["size"]
            
            if side_lower == "buy":
                if short_sz > 0:
                    close_qty = min(amount, short_sz)
                    rem_qty = amount - close_qty
                    
                    exec_close = self._execute_order_leg(
                        symbol, side, "SHORT", True, close_qty, price, params, order_type, reduce_only, 0
                    )
                    
                    exec_open = 0.0
                    if rem_qty > 0 and not reduce_only:
                        exec_open = self._execute_order_leg(
                            symbol, side, "LONG", False, rem_qty, price, params, order_type, reduce_only, 0
                        )
                    return exec_close + exec_open
                else:
                    if reduce_only:
                        return 0.0
                    return self._execute_order_leg(
                        symbol, side, "LONG", False, amount, price, params, order_type, reduce_only, 0
                    )
            else: # sell
                if long_sz > 0:
                    close_qty = min(amount, long_sz)
                    rem_qty = amount - close_qty
                    
                    exec_close = self._execute_order_leg(
                        symbol, side, "LONG", True, close_qty, price, params, order_type, reduce_only, 0
                    )
                    
                    exec_open = 0.0
                    if rem_qty > 0 and not reduce_only:
                        exec_open = self._execute_order_leg(
                            symbol, side, "SHORT", False, rem_qty, price, params, order_type, reduce_only, 0
                        )
                    return exec_close + exec_open
                else:
                    if reduce_only:
                        return 0.0
                    return self._execute_order_leg(
                        symbol, side, "SHORT", False, amount, price, params, order_type, reduce_only, 0
                    )
        else:
            # Original Hedge Mode Logic
            ctx = self._order_position_context(symbol, side, params)
            pos_idx = ctx["pos_idx"]
            reduce_only = ctx["reduce_only"]
            pos_key = ctx["pos_key"]
            is_closing = ctx["is_reduce"]
            
            return self._execute_order_leg(
                symbol, side, pos_key, is_closing, amount, price, params, order_type, reduce_only, pos_idx
            )

    def get_latest_price(self, symbol):
        symbol = self._normalize_to_ccxt(symbol)
        if symbol in self.price_cache:
            return self.price_cache[symbol]
        try:
            if symbol not in self.fast_ohlcv_arrays or "1m" not in self.fast_ohlcv_arrays[symbol]:
                return None

            arr = self.fast_ohlcv_arrays[symbol]["1m"]['c']
            if len(arr) == 0:
                return None

            idx = self._get_tf_idx(symbol, '1m')
            if idx < 0:
                p = float(self.fast_ohlcv_arrays[symbol]["1m"]['o'][0])
            elif idx >= len(arr):
                p = float(arr[-1])
            else:
                p = float(arr[idx])
            if p > 0 and not np.isnan(p):
                self.price_cache[symbol] = p
                return p
                
            # If NaN or 0, fallback to last valid price by searching backwards from idx
            for i in range(idx, -1, -1):
                val = float(arr[i])
                if val > 0 and not np.isnan(val):
                    self.price_cache[symbol] = val
                    return val
            
            # Failing that, search forward from idx
            for i in range(idx, len(arr)):
                val = float(arr[i])
                if val > 0 and not np.isnan(val):
                    self.price_cache[symbol] = val
                    return val
            
            return None
        except Exception:
            return None

    def fetch_positions(self, symbols=None, params={}):
        self._update_pnl() # Ensure unrealized PnL is perfectly up-to-date
        res = []
        normalized_targets = [self._normalize_to_ccxt(s) for s in symbols] if symbols else None
        for symbol, pos_dict in self.positions.items():
            norm_sym = self._normalize_to_ccxt(symbol)
            if normalized_targets and norm_sym not in normalized_targets: continue
            for side, pos in pos_dict.items():
                if pos["size"] > 0:
                    if not self.is_hedge_mode:
                        res.append({
                            "symbol": symbol,
                            "side": "long" if side == "LONG" else "short",
                            "contracts": pos["size"],
                            "entryPrice": pos["entryPrice"],
                            "unrealizedPnl": pos["unrealizedPnl"],
                            "info": {
                                "size": str(pos["size"]),
                                "positionIdx": 0,
                                "side": "Buy" if side == "LONG" else "Sell"
                            }
                        })
                    else:
                        res.append({
                            "symbol": symbol, "side": side.lower(), "contracts": pos["size"],
                            "entryPrice": pos["entryPrice"], "unrealizedPnl": pos["unrealizedPnl"],
                            "info": {"size": str(pos["size"]), "positionIdx": 1 if side=="LONG" else 2}
                        })
        return res

    def fetch_position(self, symbol, params={}):
        positions = self.fetch_positions([symbol])
        return positions[0] if positions else None

    @property
    def has(self):
        return {"fetchPositions": True, "fetchPosition": True, "createOrder": True}

    def fetch_ticker(self, symbol, params={}):
        price = self.get_latest_price(symbol)
        if price is None:
            price = 0.0
        return {"last": price, "bid": price, "ask": price, "close": price, "symbol": symbol}

    def fetch_balance(self, params={}):
        if self.balance_cache is not None:
            return self.balance_cache
        total_unrealized = self._update_pnl()
        wallet_balance = self.balance["USDT"]["total"]
        equity = wallet_balance + total_unrealized
        free_margin = self._get_free_margin()
        used_margin = max(0.0, equity - free_margin)
        # Match live Bybit unified account: bot sizing reads equity, not wallet-only
        bal = {
            "USDT": {
                "free": free_margin,
                "used": used_margin,
                "total": wallet_balance
            },
            "timestamp": self.current_timestamp_ms,
            "datetime": str(self.current_timestamp),
            "free": {"USDT": free_margin},
            "used": {"USDT": used_margin},
            "total": {"USDT": wallet_balance},
            "info": {
                "result": {
                    "list": [{
                        "coin": [
                            {
                                "coin": "USDT",
                                "equity": str(equity),
                                "walletBalance": str(wallet_balance),
                            }
                        ]
                    }]
                }
            }
        }
        self.balance_cache = bal
        return bal

    def fetch_ohlcv(self, symbol, timeframe='1m', limit=100, params={}):
        symbol = self._normalize_to_ccxt(symbol)
        if not self.is_symbol_active(symbol):
            return []

        tf_idx = self._get_tf_idx(symbol, timeframe)
        cache_key = (symbol, timeframe, limit, tf_idx)
        if cache_key in self.ohlcv_cache:
            return self.ohlcv_cache[cache_key]

        try:
            t_arr = self.fast_times[symbol][timeframe]
            arr = self.fast_ohlcv_arrays[symbol][timeframe]
        except KeyError:
            return []

        idx = tf_idx + 1
        if idx <= 0:
            return []
        idx = min(idx, len(t_arr))
        start = max(0, idx - int(limit))

        # Build only the requested window from compact numpy arrays.
        # This preserves the CCXT OHLCV shape without keeping a full Python
        # list-of-lists for every symbol/timeframe in RAM.
        res = [
            [
                int(t_arr[i]),
                float(arr['o'][i]),
                float(arr['h'][i]),
                float(arr['l'][i]),
                float(arr['c'][i]),
                float(arr['v'][i]),
            ]
            for i in range(start, idx)
        ]

        # Prevent look-ahead bias: append the CURRENT INCOMPLETE higher-timeframe candle.
        if timeframe != "1m" and idx < len(t_arr):
            incomplete_candle = [
                int(t_arr[idx]),
                float(arr['o'][idx]),
                float(arr['h'][idx]),
                float(arr['l'][idx]),
                float(arr['c'][idx]),
                float(arr['v'][idx]),
            ]
            open_time = incomplete_candle[0]
            current_price = self.get_latest_price(symbol)
            incomplete_candle[4] = current_price

            try:
                t_1m = self.fast_times[symbol]["1m"]
                arr_1m = self.fast_ohlcv_arrays[symbol]["1m"]
                high_1m = arr_1m['h']
                low_1m = arr_1m['l']
                volume_1m = arr_1m.get('v')

                start_1m_idx = int(np.searchsorted(t_1m, open_time, side='left'))
                end_1m_idx = self._get_tf_idx(symbol, "1m")

                if start_1m_idx <= end_1m_idx and start_1m_idx < len(t_1m):
                    end_1m_idx = min(end_1m_idx, len(t_1m) - 1)
                    seg_highs = high_1m[start_1m_idx : end_1m_idx + 1]
                    seg_lows = low_1m[start_1m_idx : end_1m_idx + 1]
                    seg_high = float(np.max(seg_highs)) if len(seg_highs) else current_price
                    seg_low = float(np.min(seg_lows)) if len(seg_lows) else current_price
                    seg_vol = float(np.sum(volume_1m[start_1m_idx : end_1m_idx + 1])) if volume_1m is not None else 0.0
                    candle_open = float(arr_1m['o'][start_1m_idx])
                else:
                    seg_high = current_price
                    seg_low = current_price
                    seg_vol = 0.0
                    candle_open = current_price

                incomplete_candle[1] = candle_open
                incomplete_candle[2] = max(seg_high, current_price)
                incomplete_candle[3] = min(seg_low, current_price)
                incomplete_candle[5] = seg_vol
            except Exception:
                incomplete_candle[1] = current_price
                incomplete_candle[2] = current_price
                incomplete_candle[3] = current_price
                incomplete_candle[5] = 0.0

            res.append(incomplete_candle)

        self.ohlcv_cache[cache_key] = res
        return res

    def get_indicator(self, symbol, timeframe, name):
        symbol = self._normalize_to_ccxt(symbol)
        if not self.is_symbol_active(symbol):
            return None
        cache_key = (symbol, timeframe, name)
        if cache_key in self.indicator_cache:
            return self.indicator_cache[cache_key]
        # სწრაფი წვდომა ინდიკატორებზე
        try:
            idx = self._get_tf_idx(symbol, timeframe)
            
            if idx < 0:
                self.indicator_cache[cache_key] = float('nan')
                return float('nan')

            tf_durations = {
                "1m": 60000, "3m": 180000, "5m": 300000, "15m": 900000,
                "30m": 1800000, "1h": 3600000, "2h": 7200000, "4h": 14400000,
                "12h": 43200000, "1d": 86400000,
            }
            duration_ms = tf_durations.get(timeframe, 60000)
            candle_open_ms = int(self.fast_times[symbol][timeframe][idx])
            candle_close_ms = candle_open_ms + duration_ms

            # At the exact close of the selected candle, its final indicator is
            # already available. Do not append the same close again as a fake
            # live candle; that dampens RSI/EMA and can suppress signals.
            if self.current_timestamp_ms <= candle_close_ms:
                val = self.fast_indicators[symbol][timeframe][name][idx]
                self.indicator_cache[cache_key] = val
                return val
                
            current_price = self.get_latest_price(symbol)
            prev_idx = idx
            prev_close = self.fast_ohlcv_arrays[symbol][timeframe]['c'][prev_idx]
            
            if name == 'rsi':
                prev_gain = self.fast_indicators[symbol][timeframe]['rsi_gain'][prev_idx]
                prev_loss = self.fast_indicators[symbol][timeframe]['rsi_loss'][prev_idx]
                
                period = 14
                delta = current_price - prev_close
                gain = delta if delta > 0 else 0.0
                loss = -delta if delta < 0 else 0.0
                
                live_gain = prev_gain * ((period - 1) / period) + gain * (1 / period)
                live_loss = prev_loss * ((period - 1) / period) + loss * (1 / period)
                
                if live_loss == 0:
                    rs = float('inf')
                else:
                    rs = live_gain / live_loss
                    
                val = 100.0 - (100.0 / (1.0 + rs))
                self.indicator_cache[cache_key] = val
                return val
                
            elif name == 'ema':
                prev_ema = self.fast_indicators[symbol][timeframe]['ema'][prev_idx]
                ema_len = 50
                alpha = 2.0 / (ema_len + 1)
                val = prev_ema * (1 - alpha) + current_price * alpha
                self.indicator_cache[cache_key] = val
                return val

            # Fallback for non-live indicators.
            val = self.fast_indicators[symbol][timeframe][name][idx]
            self.indicator_cache[cache_key] = val
            return val
        except:
            self.indicator_cache[cache_key] = None
            return None

    def create_order(self, symbol, type, side, amount, price=None, params={}):
        symbol = self._normalize_to_ccxt(symbol)
        if not self.is_symbol_active(symbol):
            raise ccxt.ExchangeError(f"Bybit error: Symbol {symbol} is currently inactive (outside its available timeline at {self.current_timestamp}). Orders are disabled.")
        current_px = self.get_latest_price(symbol)
        idx = self._get_tf_idx(symbol, "1m")
        ohlc = self._get_1m_candle_ohlc(symbol, idx)
        amount = float(amount)
        price = float(price) if price is not None else current_px
        
        # Ensure positionIdx in params is always an integer if present
        if params is not None and "positionIdx" in params:
            try:
                params["positionIdx"] = int(params["positionIdx"])
            except (ValueError, TypeError):
                pass
        
        self.order_counter += 1
        o_id = f"m_{self.order_counter}"
        
        # Determine clientOrderId (orderLinkId)
        client_order_id = params.get("orderLinkId", "") if params else ""
        
        order = {
            "id": o_id,
            "clientOrderId": client_order_id,
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "price": price,
            "status": "open",
            "params": params,
            "filled": 0,
            "remaining": amount,
            "average": None,
            "lastTradeTimestamp": None,
            "timestamp": self.current_timestamp_ms,
            "datetime": str(self.current_timestamp)
        }
        
        self.open_orders[o_id] = order
        self.orders_by_symbol[symbol].append(order)
        self.needs_sync = False

        # Limit fill if candle range touches price (same as _process_open_orders)
        is_crossed = False
        type_lower = str(type).lower()
        side_lower = str(side).lower()
        if type_lower == "limit":
            if side_lower == "buy" and current_px <= price:
                is_crossed = True
            elif side_lower == "sell" and current_px >= price:
                is_crossed = True

        if type_lower == "market" or is_crossed:
            self.needs_sync = True
            exec_price = current_px
            try:
                exec_qty = self._execute_order(symbol, side, amount, exec_price, params, order_type=type_lower)
            except Exception:
                order["status"] = "rejected"
                order["remaining"] = order["amount"]
                order["filled"] = 0.0
                self.active_open_orders.pop(o_id, None)
                raise
            if exec_qty > 0:
                order["filled"] += exec_qty
                self._finalize_order_fill_status(order, symbol, exec_price)
                if order["status"] == "open":
                    self.active_open_orders[o_id] = order
            else:
                order["status"] = "canceled"
                order["remaining"] = order["amount"]
        else:
            # Check margin BEFORE placing order (if it increases position)
            ctx = self._order_position_context(symbol, side, params)

            if not ctx["is_reduce"]:
                # Calculate margin we would need if we place this
                # Use delta logic to allow hedging orders if they don't consume MORE margin
                old_free_margin = self._get_free_margin()
                self.active_open_orders[o_id] = order
                new_free_margin = self._get_free_margin()
                
                if new_free_margin < 0 and new_free_margin < old_free_margin - 1e-6:
                    del self.active_open_orders[o_id]
                    order["status"] = "canceled"
                    raise ccxt.InsufficientFunds(
                        f"Bybit error: Insufficient available balance (Hedge constraint). Free margin: {new_free_margin:.2f}"
                    )
            else:
                self.active_open_orders[o_id] = order
            
        return order

    def cancel_order(self, id, symbol=None, params={}):
        if id not in self.active_open_orders:
            if id in self.open_orders:
                if self.open_orders[id]["status"] == "closed":
                    raise ccxt.OrderNotFound(f"Bybit error: orderLinkId or orderId {id} has already been filled or is inactive")
                return self.open_orders[id]
            raise ccxt.OrderNotFound(f"Bybit error: orderLinkId or orderId {id} not found")

        # It's an active order, so we can cancel it
        self.open_orders[id]["status"] = "canceled"
        self.open_orders[id]["remaining"] = self.open_orders[id]["amount"]
        del self.active_open_orders[id]
        return self.open_orders[id]

    def fetch_open_orders(self, symbol=None, params={}):
        orders = []
        if symbol:
            target_ccxt = self._normalize_to_ccxt(symbol)
            orders = [o for o in self.active_open_orders.values() if self._normalize_to_ccxt(o["symbol"]) == target_ccxt]
        else:
            orders = list(self.active_open_orders.values())
            
        # Support for orderId filtering
        order_id = params.get("orderId")
        if order_id:
            orders = [o for o in orders if o["id"] == order_id]
            
        return orders

    def fetch_order(self, id, symbol=None, params={}):
        if id not in self.open_orders:
            raise ccxt.OrderNotFound(f"Order {id} not found in backtest engine.")
        return self.open_orders.get(id)

    def load_markets(self, reload=False):
        return {s: {"symbol": s} for s in self.symbols}

    @property
    def symbols(self):
        return self._symbols_list

    def set_leverage(self, leverage, symbol=None, params={}):
        self.leverage = float(leverage)
        return {"status": "ok"}

    def set_position_mode(self, hedged, symbol=None, params={}):
        self.is_hedge_mode = bool(hedged)
        return {"status": "ok"}

    def amount_to_precision(self, symbol, amount):
        # truncate safely instead of round(), to prevent rounding up past available balances
        res = f"{amount:.8f}".rstrip('0').rstrip('.')
        return res if res != "" else "0"

    def price_to_precision(self, symbol, price):
        return str(round(price, 5))

    def market(self, symbol):
        return {"symbol": symbol, "precision": {"amount": 0.0001, "price": 0.00001}, "limits": {"amount": {"min": 0.001}, "cost": {"min": 1.0}}}

    @property
    def markets(self):
        return {s: self.market(s) for s in self._symbols_list}

    def parse_order(self, order_info, market=None):
        # Bybit V5 to CCXT format
        raw_status = str(order_info.get("orderStatus") or "")
        if raw_status in ["New", "PartiallyFilled"]:
            status = "open"
        elif raw_status in ["Filled"]:
            status = "closed"
        elif raw_status in ["Cancelled", "Canceled", "Deactivated"]:
            status = "canceled"
        elif raw_status in ["Rejected"]:
            status = "rejected"
        else:
            status = raw_status.lower() or "closed"
        
        avg_str = order_info.get("avgPrice")
        price_str = order_info.get("price")
        avg_price = 0.0
        if avg_str and avg_str != "" and avg_str != "0":
            avg_price = float(avg_str)
        elif price_str and price_str != "":
            avg_price = float(price_str)
            
        qty = float(order_info.get("qty", 0) or 0.0)
        cum_exec = float(order_info.get("cumExecQty", 0) or 0.0)
        
        return {
            "id": order_info.get("orderId"),
            "clientOrderId": order_info.get("orderLinkId"),
            "symbol": order_info.get("symbol"),
            "side": order_info.get("side", "").lower(),
            "price": float(price_str) if price_str and price_str != "" else 0.0,
            "amount": qty,
            "status": status,
            "average": avg_price if avg_price > 0 else (float(price_str) if price_str and price_str != "" else 0.0),
            "filled": cum_exec,
            "remaining": max(0.0, qty - cum_exec),
            "info": order_info
        }

    def __getattr__(self, name):
        # Handle Bybit V5 Realtime Order Query (Crucial for TP Rescue logic)
        if name.startswith("private_get_v5_order_realtime"):
            def mock_realtime(params={}):
                symbol = params.get("symbol")
                target_pos_idx = params.get("positionIdx")
                
                orders = list(self.active_open_orders.values())
                if symbol:
                    target_ccxt = self._normalize_to_ccxt(symbol)
                    orders = [o for o in orders if self._normalize_to_ccxt(o["symbol"]) == target_ccxt]
                
                if target_pos_idx is not None:
                    orders = [o for o in orders if o["params"].get("positionIdx") == int(target_pos_idx)]

                v5_list = []
                for o in orders:
                    side = "Buy" if str(o.get("side", "")).lower() == "buy" else "Sell"
                    p = o.get("params") or {}
                    amount_qty = float(o.get("amount", 0.0) or 0.0)
                    filled_qty = float(o.get("filled", 0.0) or 0.0)
                    remaining_qty = float(o.get("remaining", max(0.0, amount_qty - filled_qty)) or 0.0)
                    v5_list.append({
                        "orderId": o["id"],
                        "orderLinkId": o.get("clientOrderId") or p.get("orderLinkId", ""),
                        "symbol": self._to_v5_symbol(o["symbol"]),
                        "side": side,
                        "orderType": "Limit",
                        "price": str(o["price"]),
                        "qty": str(amount_qty),
                        "orderStatus": "PartiallyFilled" if filled_qty > 0 else "New",
                        "leavesQty": str(remaining_qty),
                        "cumExecQty": str(filled_qty),
                        "positionIdx": p.get("positionIdx", 0),
                        "createdTime": str(int(self.current_timestamp_ms)),
                        "updatedTime": str(int(self.current_timestamp_ms))
                    })
                return {"result": {"list": v5_list, "nextPageCursor": ""}, "retCode": 0, "retMsg": "OK"}
            return mock_realtime

        if name.startswith("private_get_v5_order_history"):
            def mock_history(params={}):
                symbol = params.get("symbol")
                target_pos_idx = params.get("positionIdx")
                order_id = params.get("orderId")
                
                closed_orders = [o for o in self.open_orders.values() if o["status"] == "closed"]
                
                if order_id:
                    closed_orders = [o for o in closed_orders if o["id"] == order_id]
                if symbol:
                    target_ccxt = self._normalize_to_ccxt(symbol)
                    closed_orders = [o for o in closed_orders if self._normalize_to_ccxt(o["symbol"]) == target_ccxt]
                    
                if target_pos_idx is not None:
                    closed_orders = [o for o in closed_orders if str((o.get("params") or {}).get("positionIdx")) == str(target_pos_idx)]
                    
                v5_list = []
                for o in closed_orders:
                    side = "Buy" if str(o.get("side", "")).lower() == "buy" else "Sell"
                    avg_px = o.get("average") if o.get("average") is not None else o.get("price")
                    if avg_px is None: avg_px = 0.0
                    p = o.get("params") or {}
                    filled_qty = float(o.get("filled", 0.0) or 0.0)
                    amount_qty = float(o.get("amount", 0.0) or 0.0)
                    v5_list.append({
                        "orderId": o["id"],
                        "orderLinkId": o.get("clientOrderId") or p.get("orderLinkId", ""),
                        "symbol": self._to_v5_symbol(o["symbol"]),
                        "side": side,
                        "orderType": o.get("type", "limit").capitalize(),
                        "price": str(o["price"]),
                        "qty": str(amount_qty),
                        "orderStatus": "Filled",
                        "leavesQty": "0",
                        "cumExecQty": str(filled_qty),
                        "avgPrice": str(avg_px),
                        "positionIdx": p.get("positionIdx", 0),
                        "createdTime": str(int(o.get("timestamp", self.current_timestamp_ms))),
                        "updatedTime": str(int(o.get("lastTradeTimestamp", self.current_timestamp_ms) or self.current_timestamp_ms))
                    })
                return {"result": {"list": v5_list, "nextPageCursor": ""}, "retCode": 0, "retMsg": "OK"}
            return mock_history

        if name.startswith("private_get"):
            def mock_call(params={}):
                return {"result": {"list": [], "nextPageCursor": ""}}
            return mock_call
        raise AttributeError(f"MockCCXTBybit has no attribute '{name}'")

    def generate_report(self):
        print("\n" + "="*50)
        print("🏁 ბექტიესტი დასრულდა!")
        print(f"💰 საწყისი ბალანსი: $10000.00")
        print(f"📈 საბოლოო Equity: ${self.balance['USDT']['total'] + self._update_pnl():.2f}")
        print(f"📊 ჯამური PnL: ${self.balance['USDT']['total'] + self._update_pnl() - 10000.00:.2f}")
        print(f"🔄 სულ შესრულდა {len(self.trade_log)} ტრანზაქცია")
        print("="*50 + "\n")
        print("💡 შენიშვნა: ვინაიდან ბოტი იყენებს DCA სტრატეგიას და ასაშუალოებს ფასს (Average Entry),")
        print("ხშირად, ბოლო ლადერის (რუნგის) მოგებით დახურვა ბირჟის მიერ ფიქსირდება როგორც 'რეალიზებული ზარალი'")
        print("(რადგან ის იხურება საშუალო ფასზე უარეს ნიშნულზე). ეს სრულებით ნორმალურია და რეალურ Equity-ს არ აზარალებს!\n")
        
        # Save results to a file as well
        # Convert trade log timestamps to string only at the end
        serialized_trades = []
        for t in self.trade_log:
            t_copy = t.copy()
            if 'ts' in t_copy:
                t_copy['time'] = str(pd.to_datetime(t_copy.pop('ts'), unit='ms'))
            # Convert any potential numpy types to avoid serialization issues
            for k, v in t_copy.items():
                if isinstance(v, (np.integer, np.int64, np.int32)):
                    t_copy[k] = int(v)
                elif isinstance(v, (np.floating, np.float64, np.float32)):
                    t_copy[k] = float(v)
            serialized_trades.append(t_copy)

        report = {
            "final_equity": float(self.balance['USDT']['total'] + self._update_pnl()),
            "trades": serialized_trades,
            "equity_curve": [{"time": int(e["time"]), "equity": float(e["equity"])} for e in self.equity_curve]
        }
        with open("backtest_report.json", "w") as f:
            json.dump(report, f, indent=2, cls=NpEncoder)
