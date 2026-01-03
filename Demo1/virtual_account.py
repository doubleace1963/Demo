"""
Virtual Account Simulator
Simulates account balance progression based on backtest results with risk management.
"""

import pandas as pd
from datetime import datetime


def simulate_virtual_account(trades, initial_balance=5000, risk_per_trade_pct=1.0):
    """
    Simulate virtual account balance progression from backtest trades.
    
    Args:
        trades: List of trade dictionaries with keys:
                - entry_time: timestamp when trade entered
                - exit_time: timestamp when trade exited
                - r_multiple: Risk/Reward ratio (positive for wins, -1 for losses)
                - result: 'won' or 'lost'
        initial_balance: Starting account balance (default $5000)
        risk_per_trade_pct: Risk percentage per trade (default 1.0%)
    
    Returns:
        equity_curve: List of (timestamp, balance) tuples showing balance progression
        trade_results: List of trade result dictionaries with balance info
    """
    if not trades:
        return [(datetime.now(), initial_balance)], []
    
    # Filter only won and lost trades
    valid_trades = [t for t in trades if t.get('result') in ['won', 'lost']]
    
    if not valid_trades:
        return [(datetime.now(), initial_balance)], []
    
    # Sort trades by entry time
    sorted_trades = sorted(valid_trades, key=lambda x: x['entry_time'])
    
    # Initialize tracking
    current_balance = initial_balance
    equity_curve = [(sorted_trades[0]['entry_time'], initial_balance)]
    trade_results = []
    
    # Process each trade
    for trade in sorted_trades:
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        r_multiple = trade['r_multiple']
        result = trade['result']
        
        # Calculate risk amount at entry (1% of current balance)
        risk_amount = current_balance * (risk_per_trade_pct / 100)
        
        # Calculate P&L based on R multiple
        if result == 'won':
            # Win: gain = risk_amount * R multiple
            pnl = risk_amount * r_multiple
        else:
            # Loss: lose 1R (risk_amount)
            pnl = -risk_amount
        
        # Update balance at exit
        new_balance = current_balance + pnl
        
        # Record equity point at exit time
        equity_curve.append((exit_time, new_balance))
        
        # Store trade result
        trade_results.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'balance_before': current_balance,
            'risk_amount': risk_amount,
            'r_multiple': r_multiple,
            'pnl': pnl,
            'balance_after': new_balance,
            'result': result,
            'symbol': trade.get('symbol', 'N/A')
        })
        
        # Update current balance
        current_balance = new_balance
    
    return equity_curve, trade_results


def calculate_account_statistics(equity_curve, trade_results, initial_balance=5000):
    """
    Calculate statistics from virtual account simulation.
    
    Args:
        equity_curve: List of (timestamp, balance) tuples
        trade_results: List of trade result dictionaries
        initial_balance: Starting balance
    
    Returns:
        Dictionary with account statistics
    """
    if not equity_curve or not trade_results:
        return {}
    
    final_balance = equity_curve[-1][1]
    total_return = ((final_balance - initial_balance) / initial_balance) * 100
    
    # Calculate drawdown
    peak_balance = initial_balance
    max_drawdown = 0
    max_drawdown_pct = 0
    
    for timestamp, balance in equity_curve:
        if balance > peak_balance:
            peak_balance = balance
        
        drawdown = peak_balance - balance
        drawdown_pct = (drawdown / peak_balance) * 100 if peak_balance > 0 else 0
        
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_pct = drawdown_pct
    
    # Calculate win/loss stats
    total_trades = len(trade_results)
    winning_trades = [t for t in trade_results if t['result'] == 'won']
    losing_trades = [t for t in trade_results if t['result'] == 'lost']
    
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
    
    # Average P&L
    avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
    
    # Profit factor
    total_wins = sum(t['pnl'] for t in winning_trades)
    total_losses = abs(sum(t['pnl'] for t in losing_trades))
    profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
    
    return {
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_return': total_return,
        'total_return_dollars': final_balance - initial_balance,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'total_trades': total_trades,
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }
