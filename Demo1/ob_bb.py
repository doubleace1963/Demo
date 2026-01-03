"""
Tkinter GUI for the MT5 exhaustion pattern scanner.
Uses `mt5_core.scan_all` for scanning work and updates UI via thread-safe callbacks.
Double-click any pattern to view M5 candles chart with FVG zones.
"""
import threading
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import List, Dict, Tuple
from datetime import timedelta

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

import mt5_core as mt5_core



DEFAULT_MIN_CANDLE_PIPS = 50


def find_extreme_candle_index(m5_df: pd.DataFrame, pattern_type: str) -> int:
    """Find index of M5 candle with lowest low (TB Bullish) or highest high (TB Bearish)."""
    if pattern_type == 'TB Bullish':
        return m5_df['low'].idxmin()
    else:  # TB Bearish
        return m5_df['high'].idxmax()


def detect_bullish_fvg(c1, c2, c3) -> Tuple[bool, float, float]:
    """
    Detect bullish FVG: candle1.high < candle3.low
    Returns (is_fvg, gap_bottom, gap_top)
    """
    if c1['high'] < c3['low']:
        return True, c1['high'], c3['low']
    return False, 0, 0


def detect_bearish_fvg(c1, c2, c3) -> Tuple[bool, float, float]:
    """
    Detect bearish FVG: candle1.low > candle3.high
    Returns (is_fvg, gap_top, gap_bottom)
    """
    if c1['low'] > c3['high']:
        return True, c1['low'], c3['high']
    return False, 0, 0


def is_fvg_filled(fvg_top: float, fvg_bottom: float, subsequent_candles: pd.DataFrame) -> bool:
    """Check if FVG has been filled by subsequent price action."""
    for _, candle in subsequent_candles.iterrows():
        if candle['low'] <= fvg_top and candle['high'] >= fvg_bottom:
            return True
    return False

def find_unfilled_fvgs_structural(
    m5_df: pd.DataFrame,
    start_idx: int,
    pattern_type: str
) -> List[Dict]:
    """
    PASS 1:
    Detect unfilled FVGs only (no validation).
    """
    fvgs = []

    range_df = m5_df.loc[start_idx:].reset_index(drop=True)
    if len(range_df) < 3:
        return fvgs

    for i in range(len(range_df) - 2):
        c1 = range_df.iloc[i]
        c3 = range_df.iloc[i + 2]

        if pattern_type == "TB Bullish":
            if c1.high >= c3.low:
                continue
            bottom, top = c1.high, c3.low
            fvg_type = "Bullish"
        else:
            if c1.low <= c3.high:
                continue
            top, bottom = c1.low, c3.high
            fvg_type = "Bearish"

        # Check if filled AFTER formation
        subsequent = range_df.iloc[i + 3:]
        if is_fvg_filled(top, bottom, subsequent):
            continue

        fvgs.append({
            "start_time": c1.time,
            "end_time": range_df.iloc[-1].time,
            "top": top,
            "bottom": bottom,
            "type": fvg_type,
            "validation_levels": [],
            "is_validated": False
        })

    return fvgs


def validate_fvgs_by_price_projection(
    all_candles: pd.DataFrame,
    fvgs: List[Dict],
    lookahead: int = 12
) -> None:
    """
    PASS 2:
    Validate FVGs using price-only projection logic.
    Modifies fvgs in-place.
    """

    for fvg in fvgs:
        candles = all_candles[all_candles["time"] <= fvg["start_time"]]
        bottom = fvg["bottom"]
        top = fvg["top"]
        fvg_type = fvg["type"]

        validations = []

        for i in range(len(all_candles) - 2):
            c1 = all_candles.iloc[i]
            c2 = all_candles.iloc[i + 1]

            if fvg_type == "Bullish":
                reaction_ok = (
                    c1.close > c1.open and
                    bottom <= c1.close <= top and
                    c2.close < c2.open
                )
                displacement_ok = lambda c: c.close < bottom
            else:
                reaction_ok = (
                    c1.close < c1.open and
                    bottom <= c1.close <= top and
                    c2.close > c2.open
                )
                displacement_ok = lambda c: c.close > top

            if not reaction_ok:
                continue

                        # bounded displacement
            for j in range(i + 2, min(i + 2 + lookahead, len(candles))):
                disp_candle = candles.iloc[j]

                if displacement_ok(disp_candle):

                    reaction_level = c2.open
                    fvg_time = fvg["start_time"]
                    violated = False

                    # CONTINUOUS CHECK until FVG forms
                    for k in range(j + 1, len(all_candles)):
                        future_candle = all_candles.iloc[k]

                        if future_candle.time >= fvg_time:
                            break

                        # Bullish FVG invalidation
                        if fvg_type == "Bullish" and future_candle.high > reaction_level:
                            violated = True
                            break

                        # Bearish FVG invalidation
                        if fvg_type == "Bearish" and future_candle.low < reaction_level:
                            violated = True
                            break

                    if violated:
                        break

                    # ✅ VALIDATION CONFIRMED
                    validations.append({
                        "level": reaction_level,
                        "time": c2.time
                    })
                    break

                

        if validations:
            fvg["validation_levels"] = validations[-1:]
            fvg["is_validated"] = True
            break

def show_m5_chart(symbol: str, c2_time, pattern_type: str):
    """Display M5 candles for C2 day with FVG zones in a matplotlib chart."""
    if not mt5_core.initialize_connection():
        messagebox.showerror('Error', 'Failed to connect to MT5')
        return
    
    try:
        if isinstance(c2_time, pd.Timestamp):
            c2_date = c2_time.to_pydatetime()
        else:
            c2_date = pd.to_datetime(c2_time).to_pydatetime()
        
        start_time = c2_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=1)
        
        m5_df = mt5_core.fetch_m5_candles(symbol, start_time, end_time)
        
        if m5_df is None or len(m5_df) == 0:
            messagebox.showinfo('No Data', f'No M5 data available for {symbol} on {c2_date.strftime("%Y-%m-%d")}')
            return
        
        # Find extreme candle and detect FVGs
        extreme_idx = find_extreme_candle_index(m5_df, pattern_type)
        unfilled_fvgs = find_unfilled_fvgs_structural(
                            m5_df,
                            extreme_idx,
                            pattern_type
                        )

        # PASS 2 — validate using price projection
        validate_fvgs_by_price_projection(
                            m5_df,
                            unfilled_fvgs
                        )

        validated_count = sum(1 for fvg in unfilled_fvgs if fvg['is_validated'])
        
        chart_window = tk.Toplevel()
        chart_window.title(f'{symbol} - {pattern_type} - {c2_date.strftime("%Y-%m-%d")} - {len(unfilled_fvgs)} FVGs ({validated_count} validated)')
        chart_window.geometry('1200x700')
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot candlesticks
        for idx, row in m5_df.iterrows():
            color = 'green' if row['close'] > row['open'] else 'red'
            
            ax.plot([row['time'], row['time']], [row['open'], row['close']], 
                   color=color, linewidth=2, solid_capstyle='round')
            
            ax.plot([row['time'], row['time']], [row['low'], row['high']], 
                   color=color, linewidth=0.5)
        
        # Highlight extreme candle
        extreme_candle = m5_df.iloc[extreme_idx]
        extreme_label = 'Lowest Low' if pattern_type == 'TB Bullish' else 'Highest High'
        ax.scatter(extreme_candle['time'], 
                  extreme_candle['low'] if pattern_type == 'TB Bullish' else extreme_candle['high'],
                  color='blue', s=100, zorder=5, marker='v' if pattern_type == 'TB Bullish' else '^',
                  label=extreme_label)
        
        # Get the last time for horizontal lines
        last_time = m5_df.iloc[-1]['time']
        
        # Plot FVG zones and validation lines
        for fvg in unfilled_fvgs:

            fvg_color = 'lightgreen' if fvg['type'] == 'Bullish' else 'lightcoral'
            edge_color = 'darkgreen' if fvg['type'] == 'Bullish' else 'darkred'
            
            # Make validated FVGs more prominent
            alpha = 0.4 if fvg['is_validated'] else 0.2
            edge_width = 2 if fvg['is_validated'] else 1
            
            # Convert times to matplotlib format
            start_time_mpl = mdates.date2num(fvg['start_time'])
            end_time_mpl = mdates.date2num(fvg['end_time'])
            width = end_time_mpl - start_time_mpl
            height = fvg['top'] - fvg['bottom']
            
            rect = Rectangle((start_time_mpl, fvg['bottom']), width, height,
                           facecolor=fvg_color, alpha=alpha, edgecolor=edge_color, 
                           linewidth=edge_width, label=f"{fvg['type']} FVG")
            ax.add_patch(rect)
            
            # Draw validation horizontal lines
            if fvg['validation_levels']:
                line_color = 'darkblue' if fvg['type'] == 'Bullish' else 'purple'
                for val in fvg['validation_levels']:
                    # Line from validation time to current price (end of chart)
                    ax.plot([val['time'], last_time], [val['level'], val['level']], 
                           color=line_color, linestyle='--', linewidth=1.5, 
                           alpha=0.7, label='Validation Level')
        
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Price', fontsize=10)
        
        title = f'{symbol} - M5 Candles - {c2_date.strftime("%Y-%m-%d")} - {pattern_type}'
        if unfilled_fvgs:
            title += f' - {len(unfilled_fvgs)} FVG(s) ({validated_count} validated)'
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.xticks(rotation=45)
        
        # Add legend (avoid duplicate labels)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=chart_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Print FVG info to console
        if unfilled_fvgs:
            print(f"\n{'='*80}")
            print(f"UNFILLED FVGs for {symbol} on {c2_date.strftime('%Y-%m-%d')} ({pattern_type})")
            print(f"{'='*80}")
            for i, fvg in enumerate(unfilled_fvgs, 1):
                validation_status = "✓ VALIDATED" if fvg['is_validated'] else "✗ Not Validated"
                print(f"\nFVG #{i} ({fvg['type']}) - {validation_status}:")
                print(f"  Start Time: {fvg['start_time'].strftime('%Y-%m-%d %H:%M')}")
                print(f"  Top: {fvg['top']:.5f}")
                print(f"  Bottom: {fvg['bottom']:.5f}")
                print(f"  Size: {(fvg['top'] - fvg['bottom']):.5f}")
                if fvg['validation_levels']:
                    print(f"  Validation Levels:")
                    for j, val in enumerate(fvg['validation_levels'], 1):
                        print(f"    #{j}: {val['level']:.5f} at {val['time'].strftime('%H:%M')}")
            print(f"\nTotal: {len(unfilled_fvgs)} FVGs ({validated_count} validated)")
            print(f"{'='*80}\n")
        
    except Exception as e:
        messagebox.showerror('Error', f'Failed to fetch M5 data: {str(e)}')
    finally:
        mt5_core.terminate_connection()


def create_gui() -> tk.Tk:
    root = tk.Tk()
    root.title('MT5 Exhaustion Pattern Scanner')
    root.geometry('700x450')

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill='x')

    ttk.Label(frm, text='Min candle size (pips):').grid(row=0, column=0, sticky='w')
    min_pips_var = tk.StringVar(value=str(DEFAULT_MIN_CANDLE_PIPS))
    min_entry = ttk.Entry(frm, textvariable=min_pips_var, width=10)
    min_entry.grid(row=0, column=1, padx=6)

    status_var = tk.StringVar(value='Idle')
    status_lbl = ttk.Label(frm, textvariable=status_var)
    status_lbl.grid(row=0, column=2, padx=10)

    btn_frame = ttk.Frame(frm)
    btn_frame.grid(row=0, column=3, padx=6)

    start_btn = ttk.Button(btn_frame, text='Start Scan')
    stop_btn = ttk.Button(btn_frame, text='Stop', state='disabled')
    save_btn = ttk.Button(btn_frame, text='Save CSV', state='disabled')

    start_btn.grid(row=0, column=0, padx=2)
    stop_btn.grid(row=0, column=1, padx=2)
    save_btn.grid(row=0, column=2, padx=2)

    list_frame = ttk.Frame(root, padding=(10, 6))
    list_frame.pack(fill='both', expand=True)

    columns = ('symbol', 'date', 'type')
    tree = ttk.Treeview(list_frame, columns=columns, show='headings')
    tree.heading('symbol', text='Symbol')
    tree.heading('date', text='Date')
    tree.heading('type', text='Pattern')
    
    tree.column('symbol', width=100)
    tree.column('date', width=100)
    tree.column('type', width=80)

    vsb = ttk.Scrollbar(list_frame, orient='vertical', command=tree.yview)
    tree.configure(yscroll=vsb.set)
    tree.pack(side='left', fill='both', expand=True)
    vsb.pack(side='right', fill='y')

    stop_event = threading.Event()
    scan_thread = None
    found_patterns: List[Dict] = []

    def on_progress(text: str):
        root.after(0, status_var.set, text)

    def on_found(patt: Dict):
        def _insert():
            date_str = patt['date'].strftime('%Y-%m-%d')
            
            tree.insert('', 'end', values=(
                patt['symbol'], 
                date_str, 
                patt['pattern_type'],
            ))
            found_patterns.append(patt)
            save_btn.config(state='normal')

        root.after(0, _insert)

    def on_tree_double_click(event):
        selection = tree.selection()
        if not selection:
            return
        
        item = tree.item(selection[0])
        values = item['values']
        
        if len(values) >= 3:
            symbol = values[0]
            date_str = values[1]
            pattern_type = values[2]
            
            for patt in found_patterns:
                if (patt['symbol'] == symbol and 
                    patt['date'].strftime('%Y-%m-%d') == date_str and
                    patt['pattern_type'] == pattern_type):
                    show_m5_chart(symbol, patt['date'], pattern_type)
                    break

    tree.bind('<Double-Button-1>', on_tree_double_click)

    def start_scan():
        nonlocal scan_thread, stop_event, found_patterns
        try:
            min_pips = int(min_pips_var.get())
        except ValueError:
            messagebox.showerror('Input error', 'Min pips must be an integer')
            return

        for i in tree.get_children():
            tree.delete(i)
        found_patterns = []
        stop_event.clear()

        start_btn.config(state='disabled')
        stop_btn.config(state='normal')
        save_btn.config(state='disabled')
        status_var.set('Initializing...')

        def worker():
            mt5_core.scan_all(min_pips, on_progress, on_found, stop_event)
            root.after(0, lambda: start_btn.config(state='normal'))
            root.after(0, lambda: stop_btn.config(state='disabled'))

        scan_thread = threading.Thread(target=worker, daemon=True)
        scan_thread.start()

    def stop_scan():
        stop_event.set()
        status_var.set('Stopping...')

    def save_csv():
        if not found_patterns:
            messagebox.showinfo('No data', 'No patterns to save')
            return
        fp = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv')])
        if not fp:
            return
        
        df = pd.DataFrame(found_patterns)
        df.to_csv(fp, index=False)
        messagebox.showinfo('Saved', f'Saved {len(found_patterns)} patterns to {fp}')

    start_btn.config(command=start_scan)
    stop_btn.config(command=stop_scan)
    save_btn.config(command=save_csv)

    root.protocol('WM_DELETE_WINDOW', root.destroy)
    return root