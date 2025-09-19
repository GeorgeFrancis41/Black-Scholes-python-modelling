import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from datetime import datetime

r = 0.035
initial_cash = 10000
cash = initial_cash
position = 0

portfolio_value = []
trade_log = []

entry_price = None
pl_points=[]
# =============================================================================
#  -- Black Scholes Model --
# =============================================================================
def Black_Scholes_Call(S0, K, sigma, r,t):
    """
    Parameters
    ----------
    S : Spot price of an asset
        Price on the day of maturity.
    K : Strike price
        Pre-set asset price.
    sigma : Volatility of the asset
        Frequency of asset fluctuation.
    r : Risk-free interest rate
        Hypothetical situation where there lies zero risk of finacial losses.
    t : Time to maturity
        Time remaining until the contract expires.

    Returns
    -------
    C : Call option price
        Estimated price of the call option after time 't' has passed.

    """
    d1 = (np.log(S0/K) + (r  + (sigma**2/2))*t)/(sigma*np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    C = S0*norm.cdf(d1) - K*np.exp(-r*t)*norm.cdf(d2)
    return C


#Use real data
data = yf.download("^STOXX50E", start="2024-01-01", end=datetime.today(), auto_adjust=False)
dataSPX = yf.download("^GSPC",start="2024-01-01", end=datetime.today(), auto_adjust=False)

dataSPX['Value'] = (dataSPX['Close'] / dataSPX['Close'].iloc[0]) * initial_cash

#Calculate returns
returns = np.log(data['Close']/data['Close'].shift(1)).dropna()

#Anualised Volatility
sigma_real = returns.std() * np.sqrt(252)

#Latest price
S0_real = data['Close'].iloc[-1]
K = round(S0_real)



#Time to maturity
expiry = datetime(2025,12,19)
today = datetime.today()
T_real = (expiry - today).days / 365

def BS_delta(S,K,r,sigma,T):
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return norm.cdf(d1)

Real_BS_Price = Black_Scholes_Call(S0_real, S0_real, sigma_real, 0.035, T_real)

buy_signals = []
sell_signals = []

for i in range(len(data)):
    S = float(data['Close'].iloc[i])
    T_i = (data.index[-1] - data.index[i]).days/365
    
    if T_i <= 0:
        continue
    delta = BS_delta(S, K, 0.035, sigma_real, T_i)
    
    if delta > 0.4 and position > 0:
        cash += position * S
        sell_signals.append((data.index[i], S))  
        PL = (S - entry_price)*position
        pl_points.append((data.index[i], S, PL))
        trade_log.append((data.index[i], 'SELL', S, position))
        position = 0


    
    elif delta < 0.425 and position == 0:
        position = cash // S
        entry_price = S
        cash -= position*S
        buy_signals.append((data.index[i], S))  
        trade_log.append((data.index[i],'BUY',S,position))

        
    total_value = cash + position * S
    portfolio_value.append((data.index[i], total_value))
    pv_df = pd.DataFrame(portfolio_value, columns = ['Date', 'Value']).set_index('Date')
    comparison_df = pv_df.join(dataSPX['Value'], how = 'inner',lsuffix='_Portfolio', rsuffix='_SP500')




fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.subplots_adjust(hspace=0.4)

ax1.plot(data.index, data['Close'])

if buy_signals:
    buy_dates, buy_prices = zip(*buy_signals)
    ax1.scatter(buy_dates, buy_prices, marker='^', color='green', label='Buy Signal')

if sell_signals:
    sell_dates, sell_prices = zip(*sell_signals)
    ax1.scatter(sell_dates, sell_prices, marker='v', color='red', label='Sell Signal')


for date, price, pl in pl_points:
    color = 'green' if pl>= 0 else 'red'
    ax1.annotate(f"{pl:.0f}", xy = (date,price), xytext = (0,10), textcoords = 'offset points', ha = 'center', color = color, fontsize = 8)
    
ax1.set_title("Historical price of Euro Stoxx 50")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")

ax2.plot(comparison_df.index, comparison_df['Value_Portfolio'], label='My Portfolio', color='purple', linewidth=2)
ax2.plot(comparison_df.index, comparison_df['Value_SP500'], label='S&P 500 (Buy & Hold)', color='orange', linestyle='--')

ax2.axhline(initial_cash, color='gray', linestyle='--', label='Starting Capital')
ax2.set_title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Value (Euros)")

plt.legend()
plt.grid(True)

plt.legend()
plt.grid(True)
plt.show()







