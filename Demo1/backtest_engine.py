"""
Backtest engine for validated FVG trading strategy.
Entry: Wait for price to touch validation level on C3 (day after FVG formation)
SL: Lowest low (bullish) / Highest high (bearish) from validation candle to FVG formation
TP: 3x SL distance
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import mt5_core as mt5_core


def _get_pip_size(symbol: str) -> Optional[float]:
    """Best-effort pip size from MT5 symbol info.

    Heuristic:
    - For 5-digit and 3-digit symbols, 1 pip is typically 10 * point.
    - Otherwise, assume 1 pip == point.
    """
    try:
        info = mt5_core.mt5.symbol_info(symbol)
    except Exception:
        info = None
    if info is None or not getattr(info, 'point', None):
        return None

    point = float(info.point)
    digits = int(getattr(info, 'digits', 0) or 0)
    if digits in (3, 5):
        return point * 10
    return point


def _price_diff_to_pips(symbol: str, price_diff: float) -> Optional[float]:
    pip_size = _get_pip_size(symbol)
    if not pip_size:
        return None
    return price_diff / pip_size


def _fetch_next_trading_day_m5(
    symbol: str,
    start_day: object,
    lookahead_days: int = 10,
) -> Tuple[Optional[pd.DataFrame], Optional[datetime], Optional[datetime]]:
    """Fetch M5 candles for the first day on/after start_day that actually has data.

    This handles holidays/weekends where the "calendar next day" has no candles.
    Returns (m5_df, used_day_start, used_day_end). If none found, returns (None, None, None).
    """
    # Important: do NOT force midnight here.
    # MT5 daily candles often start at a broker-specific hour (not 00:00 UTC/local).
    # Using the original time-of-day keeps our 24h window aligned with MT5's session.
    start_dt = pd.to_datetime(start_day).to_pydatetime().replace(microsecond=0)

    for i in range(max(0, int(lookahead_days)) + 1):
        day_start = start_dt + timedelta(days=i)
        day_end = day_start + timedelta(days=1)
        m5 = mt5_core.fetch_m5_candles(symbol, day_start, day_end)
        if m5 is not None and len(m5) > 0:
            # Filter out any candles that are before our search start time (avoid C2 overlap)
            m5_filtered = m5[m5['time'] > start_dt]
            if len(m5_filtered) > 0:
                return m5_filtered.reset_index(drop=True), day_start, day_end

    return None, None, None


def get_c1_midpoint(symbol: str, c2_date: pd.Timestamp) -> Optional[float]:
    """
    Get C1 daily candle (day before C2) and calculate its midpoint.
    Midpoint = (C1.high + C1.low) / 2
    """
    print(f"\n[DEBUG get_c1_midpoint] Symbol: {symbol}, C2 date: {c2_date}")
    
    # Fetch daily candles from a range that includes C2 and C1
    c2_dt = pd.to_datetime(c2_date).to_pydatetime()
    
    # Fetch from 5 days before C2 to ensure we get C1 (accounting for weekends)
    start_date = c2_dt - timedelta(days=5)
    end_date = c2_dt + timedelta(days=1)
    
    daily_candles = mt5_core.mt5.copy_rates_range(symbol, mt5_core.mt5.TIMEFRAME_D1, start_date, end_date)
    
    if daily_candles is None or len(daily_candles) < 2:
        print(f"[DEBUG get_c1_midpoint] Failed: Got {len(daily_candles) if daily_candles is not None else 0} daily candles")
        return None
    
    df = pd.DataFrame(daily_candles)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    print(f"[DEBUG get_c1_midpoint] Fetched {len(df)} daily candles:")
    for idx, row in df.iterrows():
        print(f"  [{idx}] {row['time']} - O:{row['open']:.5f} H:{row['high']:.5f} L:{row['low']:.5f} C:{row['close']:.5f}")
    
    # Find C2 in the data
    c2_candles = df[df['time'].dt.date == c2_dt.date()]
    print(f"[DEBUG get_c1_midpoint] Looking for C2 date: {c2_dt.date()}, found {len(c2_candles)} matches")
    
    if len(c2_candles) == 0:
        print(f"[DEBUG get_c1_midpoint] Failed: C2 date not found in daily candles")
        return None
    
    c2_idx = c2_candles.index[0]
    print(f"[DEBUG get_c1_midpoint] C2 index: {c2_idx}")
    
    # C1 is the candle before C2
    if c2_idx == 0:
        print(f"[DEBUG get_c1_midpoint] Failed: C2 is at index 0, no C1 available")
        return None
    
    c1 = df.iloc[c2_idx - 1]
    midpoint = (float(c1['high']) + float(c1['low'])) / 2.0
    
    print(f"[DEBUG get_c1_midpoint] C1: {c1['time']} - High:{c1['high']:.5f}, Low:{c1['low']:.5f}, Midpoint:{midpoint:.5f}")
    
    return midpoint


def calculate_sl_level(
    m5_df: pd.DataFrame,
    validation_time: pd.Timestamp,
    fvg_formation_time: pd.Timestamp,
    fvg_type: str
) -> Optional[float]:
    """
    Calculate stop loss level based on price action from validation candle to FVG formation.
    
    For Bullish FVG: Find lowest low in the range (SL below entry)
    For Bearish FVG: Find highest high in the range (SL above entry)
    
    Note: validation_time typically comes BEFORE fvg_formation_time
    """
    # Per strategy definition, validation should occur strictly BEFORE FVG formation.
    if validation_time >= fvg_formation_time:
        return None

    # Get candles from validation to FVG formation (inclusive)
    range_candles = m5_df[
        (m5_df['time'] >= validation_time) &
        (m5_df['time'] <= fvg_formation_time)
    ]
    
    if len(range_candles) == 0:
        return None
    
    if fvg_type == 'Bullish':
        return range_candles['low'].min()
    else:  # Bearish
        return range_candles['high'].max()


def check_entry_triggered(
    c3_candles: pd.DataFrame,
    validation_level: float,
    fvg_type: str
) -> Optional[Tuple[pd.Timestamp, float, int]]:
    """
    Check if price touched validation level during C3.
    Returns (entry_time, entry_price, entry_candle_index) or None if not triggered.
    
    For Bullish FVG: Check if any candle's low <= validation_level (price comes down to entry)
    For Bearish FVG: Check if any candle's high >= validation_level (price comes up to entry)
    """
    for idx, (_, candle) in enumerate(c3_candles.iterrows()):
        if fvg_type == 'Bullish':
            # For bullish, we wait for price to come DOWN to validation level
            if candle['low'] <= validation_level:
                return (candle['time'], validation_level, idx)
        else:  # Bearish
            # For bearish, we wait for price to come UP to validation level
            if candle['high'] >= validation_level:
                return (candle['time'], validation_level, idx)
    
    return None


def check_sl_tp_hit_on_candle(
    candle: pd.Series,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    fvg_type: str
) -> Optional[str]:
    """
    Check if SL or TP was hit on a single candle.
    Returns 'win', 'loss', or None if neither hit.
    
    If both SL and TP could be hit within the candle's range, the trade is discarded
    as ambiguous (no reliable order from OHLC data).
    """
    if fvg_type == 'Bullish':
        # SL is below entry, TP is above entry
        sl_hit = candle['low'] <= stop_loss
        tp_hit = candle['high'] >= take_profit
        
        if sl_hit and tp_hit:
            return 'discard'
        elif sl_hit:
            return 'loss'
        elif tp_hit:
            return 'win'
    else:  # Bearish
        # SL is above entry, TP is below entry
        sl_hit = candle['high'] >= stop_loss
        tp_hit = candle['low'] <= take_profit
        
        if sl_hit and tp_hit:
            return 'discard'
        elif sl_hit:
            return 'loss'
        elif tp_hit:
            return 'win'
    
    return None


def monitor_trade_on_c3(
    symbol: str,
    c3_candles: pd.DataFrame,
    entry_candle_idx: int,
    entry_time: pd.Timestamp,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    fvg_type: str,
    aggressive_mode: bool = False
) -> Dict:
    """
    Monitor trade on C3 M5 candles starting from entry candle.
    Check each subsequent candle to see if SL or TP is hit first.
    
    Conservative mode (default):
    - At 3R: Take 50% profit, move SL to breakeven
    - Let remaining 50% run to TP or new SL
    
    Aggressive mode:
    - No partial profit taking
    - Hold full position until TP or SL
    
    Returns trade result with all details.
    """
    sl_distance = abs(entry_price - stop_loss)
    partial_taken = False
    partial_r = 0.0
    current_sl = stop_loss
    
    result = {
        'entry_time': entry_time,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'exit_time': None,
        'exit_price': None,
        'outcome': 'pending',  # Not closed by end of C3
        'pips': 0.0,
        'r_multiple': 0.0,
        'days_held': 0.0,
        'last_checked_time': c3_candles.iloc[-1]['time'] if len(c3_candles) else entry_time,
        'reason': None,
        'partial_taken': False,
        'partial_r': 0.0,
    }
    
    # Start checking from the ENTRY candle (trade can resolve immediately).
    for idx in range(entry_candle_idx, len(c3_candles)):
        candle = c3_candles.iloc[idx]
        
        # Check for partial profit taking at 3R (before checking SL/TP)
        # Skip in aggressive mode
        if not aggressive_mode and not partial_taken:
            if fvg_type == 'Bullish':
                current_profit = candle['high'] - entry_price
            else:  # Bearish
                current_profit = entry_price - candle['low']
            
            current_r = current_profit / sl_distance if sl_distance > 0 else 0
            
            if current_r >= 3.0:
                # Take 50% profit at 3R
                partial_taken = True
                partial_r = 3.0 * 0.5  # 50% of 3R = 1.5R
                current_sl = entry_price  # Move SL to breakeven
                result['partial_taken'] = True
                result['partial_r'] = partial_r
        
        # Check SL/TP with current SL level
        outcome = check_sl_tp_hit_on_candle(candle, entry_price, current_sl, take_profit, fvg_type)
        
        if outcome == 'discard':
            result['exit_time'] = candle['time']
            result['exit_price'] = None
            result['outcome'] = 'discarded'
            result['pips'] = 0.0
            result['r_multiple'] = 0.0
            result['reason'] = 'SL and TP both within the same candle range'
            break
        if outcome == 'loss':
            result['exit_time'] = candle['time']
            result['exit_price'] = current_sl
            
            if partial_taken:
                # Hit breakeven after taking partial
                result['outcome'] = 'breakeven_after_partial'
                result['r_multiple'] = partial_r  # Only keep the partial profit
                price_diff = 0.0  # Breakeven on remaining position
            else:
                # Full loss
                result['outcome'] = 'loss'
                result['r_multiple'] = -1.0
                price_diff = (current_sl - entry_price) if fvg_type == 'Bullish' else (entry_price - current_sl)
            
            pips = _price_diff_to_pips(symbol, price_diff)
            result['pips'] = float(pips) if pips is not None else float(price_diff)
            
            # Store potential R if TP had been hit
            tp_distance = abs(take_profit - entry_price)
            result['potential_r'] = tp_distance / sl_distance if sl_distance > 0 else 0.0
            break
        elif outcome == 'win':
            result['exit_time'] = candle['time']
            result['exit_price'] = take_profit
            
            # Calculate R for remaining 50%
            tp_distance = abs(take_profit - entry_price)
            remaining_r = tp_distance / sl_distance if sl_distance > 0 else 0.0
            
            if partial_taken:
                result['outcome'] = 'full_win_with_partial'
                result['r_multiple'] = partial_r + (remaining_r * 0.5)  # 1R from partial + 50% of full R
            else:
                result['outcome'] = 'win'
                result['r_multiple'] = remaining_r
            
            price_diff = (take_profit - entry_price) if fvg_type == 'Bullish' else (entry_price - take_profit)
            pips = _price_diff_to_pips(symbol, price_diff)
            result['pips'] = float(pips) if pips is not None else float(price_diff)
            break
    
    # Calculate hours held if trade closed
    if result['exit_time']:
        result['hours_held'] = (result['exit_time'] - entry_time).total_seconds() / 3600
    
    return result


def monitor_trade_extended(
    symbol: str,
    entry_time: pd.Timestamp,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    fvg_type: str,
    after_time: Optional[pd.Timestamp] = None,
    max_days: Optional[int] = None,
    aggressive_mode: bool = False
) -> Dict:
    """
    Extended monitoring if trade didn't close on C3.
    Fetch additional M5 data and monitor until TP/SL hit or max days reached.
    
    Conservative mode (default):
    - At 3R: Take 50% profit, move SL to breakeven
    - Let remaining 50% run to TP or new SL
    
    Aggressive mode:
    - No partial profit taking
    - Hold full position until TP or SL
    """
    sl_distance = abs(entry_price - stop_loss)
    partial_taken = False
    partial_r = 0.0
    current_sl = stop_loss
    
    result = {
        'entry_time': entry_time,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'exit_time': None,
        'exit_price': None,
        'outcome': 'open',
        'pips': 0.0,
        'r_multiple': 0.0,
        'hours_held': 0.0,
        'last_checked_time': None,
        'reason': None,
        'partial_taken': False,
        'partial_r': 0.0,
    }
    
    # Fetch M5 data for extended monitoring period
    start_time = entry_time
    if max_days is not None:
        end_time = entry_time + timedelta(days=max_days)
    else:
        end_time = datetime.now()
    
    m5_data = mt5_core.fetch_m5_candles(symbol, start_time, end_time)
    
    if m5_data is None or len(m5_data) == 0:
        result['outcome'] = 'open'
        result['last_checked_time'] = after_time or entry_time
        return result
    
    # Monitor each candle after entry
    for _, candle in m5_data.iterrows():
        if candle['time'] <= entry_time:
            continue
        if after_time is not None and candle['time'] <= after_time:
            continue
        
        # Check for partial profit taking at 3R (before checking SL/TP)
        # Skip in aggressive mode
        if not aggressive_mode and not partial_taken:
            if fvg_type == 'Bullish':
                current_profit = candle['high'] - entry_price
            else:  # Bearish
                current_profit = entry_price - candle['low']
            
            current_r = current_profit / sl_distance if sl_distance > 0 else 0
            
            if current_r >= 3.0:
                # Take 50% profit at 3R
                partial_taken = True
                partial_r = 3.0 * 0.5  # 50% of 3R = 1.5R
                current_sl = entry_price  # Move SL to breakeven
                result['partial_taken'] = True
                result['partial_r'] = partial_r
        
        # Check SL/TP with current SL level
        outcome = check_sl_tp_hit_on_candle(candle, entry_price, current_sl, take_profit, fvg_type)
        
        if outcome == 'discard':
            result['exit_time'] = candle['time']
            result['exit_price'] = None
            result['outcome'] = 'discarded'
            result['pips'] = 0.0
            result['r_multiple'] = 0.0
            result['reason'] = 'SL and TP both within the same candle range'
            break
        if outcome == 'loss':
            result['exit_time'] = candle['time']
            result['exit_price'] = current_sl
            
            if partial_taken:
                # Hit breakeven after taking partial
                result['outcome'] = 'breakeven_after_partial'
                result['r_multiple'] = partial_r  # Only keep the partial profit
                price_diff = 0.0  # Breakeven on remaining position
            else:
                # Full loss
                result['outcome'] = 'loss'
                result['r_multiple'] = -1.0
                price_diff = (current_sl - entry_price) if fvg_type == 'Bullish' else (entry_price - current_sl)
            
            pips = _price_diff_to_pips(symbol, price_diff)
            result['pips'] = float(pips) if pips is not None else float(price_diff)
            
            # Store potential R if TP had been hit
            tp_distance = abs(take_profit - entry_price)
            result['potential_r'] = tp_distance / sl_distance if sl_distance > 0 else 0.0
            break
        elif outcome == 'win':
            result['exit_time'] = candle['time']
            result['exit_price'] = take_profit
            
            # Calculate R for remaining 50%
            tp_distance = abs(take_profit - entry_price)
            remaining_r = tp_distance / sl_distance if sl_distance > 0 else 0.0
            
            if partial_taken:
                result['outcome'] = 'full_win_with_partial'
                result['r_multiple'] = partial_r + (remaining_r * 0.5)  # 1R from partial + 50% of full R
            else:
                result['outcome'] = 'win'
                result['r_multiple'] = remaining_r
            
            price_diff = (take_profit - entry_price) if fvg_type == 'Bullish' else (entry_price - take_profit)
            pips = _price_diff_to_pips(symbol, price_diff)
            result['pips'] = float(pips) if pips is not None else float(price_diff)
            break
    
    # Calculate hours held if trade closed
    last_time = m5_data.iloc[-1]['time']
    result['last_checked_time'] = last_time

    if result['exit_time']:
        result['hours_held'] = (result['exit_time'] - entry_time).total_seconds() / 3600
    else:
        # Still open as of last available candle
        result['hours_held'] = (last_time - entry_time).total_seconds() / 3600
    
    return result


def _select_fvg_farthest_from_c2_close(validated_fvgs: List[Dict], c2_close: float) -> Optional[Dict]:
    """Pick the validated FVG whose validation level is farthest from C2 close."""
    if not validated_fvgs:
        return None

    candidates: List[Dict] = []
    for fvg in validated_fvgs:
        if not fvg or not fvg.get('is_validated'):
            continue
        levels = fvg.get('validation_levels') or []
        if not levels:
            continue
        validation_time = levels[0].get('time')
        formation_time = fvg.get('start_time')
        if validation_time is None or formation_time is None:
            continue
        # Validation must happen strictly before formation.
        if pd.to_datetime(validation_time) >= pd.to_datetime(formation_time):
            continue
        candidates.append(fvg)

    if not candidates:
        return None

    def score(f: Dict) -> float:
        lvl = f['validation_levels'][0]['level']
        return abs(float(lvl) - float(c2_close))

    return max(candidates, key=score)


def backtest_validated_fvg(
    symbol: str,
    fvg: Dict,
    c2_date: pd.Timestamp,
    pattern_type: str,
    m5_c2: Optional[pd.DataFrame] = None,
    aggressive_mode: bool = False
) -> Optional[Dict]:
    """
    Backtest a single validated FVG.
    
    Process:
    1. Get validation level and calculate SL/TP
    2. Check if entry triggered on C3
    3. If triggered, monitor trade until completion
    4. Return trade result
    
    Args:
        aggressive_mode: If True, skip partial profit taking at 3R
    """
    if not fvg.get('is_validated') or not fvg.get('validation_levels'):
        return {
            'symbol': symbol,
            'fvg_date': c2_date,
            'pattern_type': pattern_type,
            'entry_triggered': False,
            'outcome': 'no_valid_fvg',
            'reason': 'FVG is not validated or missing validation levels',
        }
    
    validation_level = fvg['validation_levels'][0]['level']
    validation_time = fvg['validation_levels'][0]['time']
    fvg_formation_time = fvg['start_time']
    fvg_type = fvg['type']

    # Validation must be strictly before formation (per strategy definition)
    if pd.to_datetime(validation_time) >= pd.to_datetime(fvg_formation_time):
        return {
            'symbol': symbol,
            'fvg_date': c2_date,
            'pattern_type': pattern_type,
            'fvg_type': fvg_type,
            'validation_level': validation_level,
            'entry_triggered': False,
            'outcome': 'no_valid_fvg',
            'reason': 'Validation time must be before FVG formation time',
        }
    
    # Get M5 data for the C2 session.
    # Use the D1 candle timestamp as the 24h anchor (do not force midnight).
    c2_start = pd.to_datetime(c2_date).to_pydatetime().replace(microsecond=0)
    c2_end = c2_start + timedelta(days=1)
    
    if m5_c2 is None:
        m5_c2 = mt5_core.fetch_m5_candles(symbol, c2_start, c2_end)
    
    if m5_c2 is None or len(m5_c2) == 0:
        return {
            'symbol': symbol,
            'fvg_date': c2_date,
            'pattern_type': pattern_type,
            'fvg_type': fvg_type,
            'validation_level': validation_level,
            'entry_triggered': False,
            'outcome': 'no_c2_data',
            'reason': 'No C2 M5 data',
        }
    
    # Calculate SL
    sl_level = calculate_sl_level(m5_c2, validation_time, fvg_formation_time, fvg_type)
    
    if sl_level is None:
        return {
            'symbol': symbol,
            'fvg_date': c2_date,
            'pattern_type': pattern_type,
            'fvg_type': fvg_type,
            'validation_level': validation_level,
            'entry_triggered': False,
            'outcome': 'no_sl',
            'reason': 'Could not calculate SL (validation must be before formation)',
        }
    
    # Calculate TP as C1 midpoint
    tp_level = get_c1_midpoint(symbol, c2_date)
    
    if tp_level is None:
        return {
            'symbol': symbol,
            'fvg_date': c2_date,
            'pattern_type': pattern_type,
            'fvg_type': fvg_type,
            'validation_level': validation_level,
            'stop_loss': sl_level,
            'entry_triggered': False,
            'outcome': 'no_tp',
            'reason': 'Could not calculate TP (C1 midpoint unavailable)',
        }
    
    # Validate entry vs TP logic
    # For Bullish: entry must be < TP (price moves UP to target)
    # For Bearish: entry must be > TP (price moves DOWN to target)
    if fvg_type == 'Bullish':
        if validation_level >= tp_level:
            return {
                'symbol': symbol,
                'fvg_date': c2_date,
                'pattern_type': pattern_type,
                'fvg_type': fvg_type,
                'validation_level': validation_level,
                'stop_loss': sl_level,
                'take_profit': tp_level,
                'entry_triggered': False,
                'outcome': 'invalid_tp',
                'reason': f'Bullish entry ({validation_level:.5f}) >= TP ({tp_level:.5f})',
            }
    else:  # Bearish
        if validation_level <= tp_level:
            return {
                'symbol': symbol,
                'fvg_date': c2_date,
                'pattern_type': pattern_type,
                'fvg_type': fvg_type,
                'validation_level': validation_level,
                'stop_loss': sl_level,
                'take_profit': tp_level,
                'entry_triggered': False,
                'outcome': 'invalid_tp',
                'reason': f'Bearish entry ({validation_level:.5f}) <= TP ({tp_level:.5f})',
            }
    
    # Calculate distances and pips
    sl_distance = abs(validation_level - sl_level)
    tp_distance = abs(tp_level - validation_level)
    sl_distance_pips = _price_diff_to_pips(symbol, float(sl_distance))
    tp_distance_pips = _price_diff_to_pips(symbol, float(tp_distance))
    
    # Calculate risk-reward ratio dynamically
    risk_reward = tp_distance / sl_distance if sl_distance > 0 else 0
    
    # Get C3 candles (next trading day after C2; skip holidays/weekends)
    c3_calendar_start = c2_end
    m5_c3, c3_start, c3_end = _fetch_next_trading_day_m5(symbol, c3_calendar_start, lookahead_days=10)

    if m5_c3 is None or len(m5_c3) == 0 or c3_start is None:
        return {
            'symbol': symbol,
            'fvg_date': c2_date,
            'pattern_type': pattern_type,
            'fvg_type': fvg_type,
            'validation_level': validation_level,
            'stop_loss': sl_level,
            'take_profit': tp_level,
            'entry_triggered': False,
            'outcome': 'no_entry',
            'reason': 'No C3 data (no trading day found in lookahead window)'
        }
    
    # Check if entry triggered on C3
    entry_result = check_entry_triggered(m5_c3, validation_level, fvg_type)
    
    if entry_result is None:
        return {
            'symbol': symbol,
            'fvg_date': c2_date,
            'pattern_type': pattern_type,
            'fvg_type': fvg_type,
            'validation_level': validation_level,
            'stop_loss': sl_level,
            'take_profit': tp_level,
            'c3_date_used': c3_start,
            'entry_triggered': False,
            'outcome': 'no_entry',
            'reason': 'Price did not touch validation level on C3'
        }
    
    # Entry triggered - first monitor trade on C3 candles
    entry_time, entry_price, entry_candle_idx = entry_result
    
    # Monitor on remaining C3 candles first
    trade_result = monitor_trade_on_c3(
        symbol,
        m5_c3,
        entry_candle_idx,
        entry_time,
        entry_price,
        sl_level,
        tp_level,
        fvg_type,
        aggressive_mode
    )
    
    # If trade didn't close on C3, continue monitoring on subsequent days
    if trade_result['outcome'] == 'pending':
        trade_result = monitor_trade_extended(
            symbol,
            entry_time,
            entry_price,
            sl_level,
            tp_level,
            fvg_type,
            after_time=m5_c3.iloc[-1]['time'] if len(m5_c3) else entry_time,
            max_days=30,
            aggressive_mode=aggressive_mode
        )
    
    # Combine all info
    full_result = {
        'symbol': symbol,
        'fvg_date': c2_date,
        'pattern_type': pattern_type,
        'fvg_type': fvg_type,
        'validation_level': validation_level,
        'stop_loss': sl_level,
        'take_profit': tp_level,
        'c3_date_used': c3_start,
        'entry_triggered': True,
        'sl_distance_pips': float(sl_distance_pips) if sl_distance_pips is not None else float(sl_distance),
        'tp_distance_pips': float(tp_distance_pips) if tp_distance_pips is not None else float(tp_distance),
        'risk_reward': float(risk_reward),
        **trade_result
    }
    
    return full_result


def calculate_backtest_statistics(results: List[Dict]) -> Dict:
    """
    Calculate comprehensive statistics from backtest results.
    """
    if not results:
        return {}
    
    # Filter only trades that were entered
    entered_trades = [r for r in results if r.get('entry_triggered', False)]
    
    if not entered_trades:
        return {
            'total_patterns': len(results),
            'entries_triggered': 0,
            'entry_rate': 0.0
        }
    
    # Filter completed trades (all outcomes including new partial outcomes)
    all_outcomes = ['win', 'loss', 'full_win_with_partial', 'breakeven_after_partial']
    completed_trades = [r for r in entered_trades if r.get('outcome') in all_outcomes]
    
    # Categorize outcomes
    full_wins = [r for r in completed_trades if r['outcome'] == 'win']
    full_losses = [r for r in completed_trades if r['outcome'] == 'loss']
    partial_wins = [r for r in completed_trades if r['outcome'] == 'full_win_with_partial']
    breakevens = [r for r in completed_trades if r['outcome'] == 'breakeven_after_partial']
    
    # For statistics, count wins = full_wins + partial_wins + breakevens (positive outcomes)
    total_wins = len(full_wins) + len(partial_wins) + len(breakevens)
    total_losses = len(full_losses)
    
    discarded = [r for r in entered_trades if r.get('outcome') == 'discarded']
    open_trades = [r for r in entered_trades if r.get('outcome') == 'open']
    pending_trades = [r for r in entered_trades if r.get('outcome') == 'pending']
    
    total_pips = sum(r.get('pips', 0) for r in completed_trades)
    total_r = sum(r.get('r_multiple', 0) for r in completed_trades)
    
    # Partial profit statistics
    trades_with_partial = [r for r in completed_trades if r.get('partial_taken', False)]
    
    stats = {
        'total_patterns': len(results),
        'entries_triggered': len(entered_trades),
        'entry_rate': len(entered_trades) / len(results) * 100,
        
        'total_trades': len(completed_trades),
        'wins': total_wins,
        'losses': total_losses,
        'full_wins': len(full_wins),
        'partial_wins': len(partial_wins),
        'breakevens': len(breakevens),
        'full_losses': len(full_losses),
        'trades_with_partial': len(trades_with_partial),
        'discarded': len(discarded),
        'open_trades': len(open_trades),
        'pending_trades': len(pending_trades),
        
        'win_rate': total_wins / len(completed_trades) * 100 if completed_trades else 0,
        
        'total_pips': total_pips,
        'average_pips': total_pips / len(completed_trades) if completed_trades else 0,
        
        'total_r': total_r,
        'average_r': total_r / len(completed_trades) if completed_trades else 0,
        
        'avg_hours_held': sum(r.get('hours_held', 0) for r in completed_trades) / len(completed_trades) if completed_trades else 0,
    }
    
    # Best and worst trades
    if completed_trades:
        stats['best_trade'] = max(completed_trades, key=lambda x: x.get('pips', 0))
        stats['worst_trade'] = min(completed_trades, key=lambda x: x.get('pips', 0))
    
    return stats


def run_backtest_on_patterns(patterns: List[Dict], progress_callback=None, aggressive_mode: bool = False) -> Tuple[List[Dict], Dict]:
    """
    Run backtest on a list of detected patterns.
    Returns (individual_results, statistics)
    
    Args:
        aggressive_mode: If True, skip partial profit taking at 3R
    """
    results = []
    
    if not mt5_core.initialize_connection():
        return results, {}
    
    try:
        total = len(patterns)
        
        for idx, pattern in enumerate(patterns, 1):
            if progress_callback:
                progress_callback(f"Backtesting {idx}/{total}: {pattern['symbol']}")
            
            symbol = pattern['symbol']
            c2_date = pattern['date']
            pattern_type = pattern.get('pattern_type', '')

            # Get validated FVGs - support both 'validated_fvgs' (list) and 'fvg' (single dict)
            validated_fvgs = pattern.get('validated_fvgs', [])
            if not validated_fvgs:
                single_fvg = pattern.get('fvg', {})
                if single_fvg:
                    validated_fvgs = [single_fvg]

            # Fetch C2 M5 once (used for SL and for choosing the farthest validation from C2 close)
            c2_start = pd.to_datetime(c2_date).replace(hour=0, minute=0, second=0, microsecond=0)
            c2_end = c2_start + timedelta(days=1)
            m5_c2 = mt5_core.fetch_m5_candles(symbol, c2_start, c2_end)

            if m5_c2 is None or len(m5_c2) == 0:
                results.append({
                    'symbol': symbol,
                    'fvg_date': c2_date,
                    'pattern_type': pattern_type,
                    'entry_triggered': False,
                    'outcome': 'no_c2_data',
                    'reason': 'No C2 M5 data',
                })
                continue

            c2_close = float(m5_c2.iloc[-1]['close'])
            chosen_fvg = _select_fvg_farthest_from_c2_close(validated_fvgs, c2_close)

            if not chosen_fvg:
                results.append({
                    'symbol': symbol,
                    'fvg_date': c2_date,
                    'pattern_type': pattern_type,
                    'entry_triggered': False,
                    'outcome': 'no_valid_fvg',
                    'reason': 'No validated FVGs (or validation occurs after formation)',
                })
                continue

            result = backtest_validated_fvg(
                symbol,
                chosen_fvg,
                c2_date,
                pattern_type,
                m5_c2=m5_c2,
                aggressive_mode=aggressive_mode
            )
            if result:
                results.append(result)
        
        stats = calculate_backtest_statistics(results)
        
        # Add mode to stats for reference
        stats['mode'] = 'aggressive' if aggressive_mode else 'conservative'
        
        if progress_callback:
            progress_callback(f"Backtest complete: {len(results)} results")
        
        return results, stats
    
    finally:
        mt5_core.terminate_connection()