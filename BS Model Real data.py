import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from datetime import datetime

r = 0.035
initial_cash = 10000
cash = initial_cash
position = 0.0

buy_signals = []
sell_signals = []
trade_log = []
portfolio_value = []

# =============================================================================
#  -- Black Scholes Model --
# =============================================================================
def Black_Scholes_Call(S0, K, sigma, r, t):
    d1 = (np.log(S0/K) + (r + (sigma**2/2))*t) / (sigma*np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    C = S0*norm.cdf(d1) - K*np.exp(-r*t)*norm.cdf(d2)
    return C

def BS_delta(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

# Use real data
data = yf.download("^STOXX50E", start="2024-01-01", end=datetime.today(), auto_adjust=False)
dataSPX = yf.download("^GSPC", start="2024-01-01", end=datetime.today(), auto_adjust=False)

# Define expiry beyond the end of your data so T never goes permanently negative
expiry = datetime(2026, 12, 19)

lookback = 252
K = None
rebalance_every = 5  # trading days between rebalances

for i in range(lookback, len(data)):
    S = float(data['Close'].iloc[i])

    # Volatility using only past data up to day i (no look-ahead)
    past_prices = data['Close'].iloc[i-lookback:i]
    past_returns = np.log(past_prices / past_prices.shift(1)).dropna()
    sigma_i = past_returns.std() * np.sqrt(252)

    # Re-strike on a fixed schedule only (not on position changes)
    if K is None or (i % 63 == 0):
        K = round(S)

    T_i = (expiry - data.index[i]).days / 365
    if T_i <= 0:
        continue

    delta = BS_delta(S, K, r, sigma_i, T_i)

    if i % rebalance_every == 0:
        total_value = cash + position * S
        target_position_value = delta * total_value
        target_position = target_position_value / S  # fractional, not floor-divided

        trade_size = target_position - position
        cost = trade_size * S
        cash -= cost

        if trade_size > 0:
            buy_signals.append((data.index[i], S))
        elif trade_size < 0:
            sell_signals.append((data.index[i], S))

        trade_log.append((data.index[i], 'REBALANCE', S, trade_size))
        position = target_position

    total_value = cash + position * S
    portfolio_value.append((data.index[i], total_value))

# Rebase S&P 500 to start at the same date and value as the backtest
backtest_start_date = data.index[lookback]
sp_start_price = dataSPX.loc[dataSPX.index >= backtest_start_date, 'Close'].iloc[0]
dataSPX['Value'] = (dataSPX['Close'] / sp_start_price) * initial_cash

pv_df = pd.DataFrame(portfolio_value, columns=['Date', 'Value']).set_index('Date')
comparison_df = pv_df.join(dataSPX['Value'], how='inner', lsuffix='_Portfolio', rsuffix='_SP500')

# =============================================================================
#  -- Plotting --
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.subplots_adjust(hspace=0.4)

ax1.plot(data.index, data['Close'])

if buy_signals:
    buy_dates, buy_prices = zip(*buy_signals)
    ax1.scatter(buy_dates, buy_prices, marker='^', color='green', label='Buy Signal')

if sell_signals:
    sell_dates, sell_prices = zip(*sell_signals)
    ax1.scatter(sell_dates, sell_prices, marker='v', color='red', label='Sell Signal')

ax1.set_title("Historical price of Euro Stoxx 50")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.legend()
ax1.grid(True)

ax2.plot(comparison_df.index, comparison_df['Value_Portfolio'], label='My Portfolio', color='purple', linewidth=2)
ax2.plot(comparison_df.index, comparison_df['Value_SP500'], label='S&P 500 (Buy & Hold)', color='orange', linestyle='--')
ax2.axhline(initial_cash, color='gray', linestyle='--', label='Starting Capital')
ax2.set_title("Portfolio Value Over Time")
ax2.set_xlabel("Date")
ax2.set_ylabel("Value (Euros)")
ax2.legend()
ax2.grid(True)

plt.show()
