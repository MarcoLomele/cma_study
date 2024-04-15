#%% Import libraries
import numpy as np
import cma
import yfinance as yf
from datetime import datetime
import warnings
from matplotlib import pyplot as plt

np.set_printoptions(suppress=True)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#% Define functions
def calc_returns(prices):
    return np.log(prices/prices.shift(1)).dropna()

def expected_returns(weights, mean_returns):
    return np.dot(weights, mean_returns)

def portfolio_std(weights, returns_covmat):
    return np.sqrt(np.dot(weights.T, np.dot(returns_covmat, weights)))

def sharpe_ratio(weights, mean_returns, returns_covmat):
    weights /= np.sum(weights)
    R = expected_returns(weights, mean_returns)
    sigma_p = portfolio_std(weights, returns_covmat)
    return (R) / sigma_p

def objective_function(weights, mean_returns, returns_covat, alpha_penalty):
    fitness = sharpe_ratio(weights, mean_returns, returns_covat) 
    fitness += alpha_penalty * min(0, weights.min())
    return -1*fitness

#% Import data
start_date = datetime(2009, 3, 1)
end_date = datetime(2024,3,1)

# stocks = 'AAPL AMZN GLD GOOGL GS META NFLX PFE SIEGY TM'
stocks = '''AAPL ABT ADBE AMD AMZN AXP BA BABA BAC CSCO CVS DHR DIS F GLD GOOGL 
            GS HD HON IBM INTC JNJ KO LLY LVMUY MA MCD META MMM MSFT NFLX NKE 
            NVDA ORCL PFE PG PYPL QCOM RIO RTX SAP SBUX SIEGY T TM TMO TSLA 
            UNH V WMT'''

prices = yf.Tickers(stocks).history(start = start_date, end = end_date)['Close']

daily_returns = calc_returns(prices)
pre_agg_returns = daily_returns.reset_index().drop(columns='Date')
weekly_returns = pre_agg_returns.groupby(pre_agg_returns.index // 5).sum()
monthly_returns = pre_agg_returns.groupby(pre_agg_returns.index // 22).sum()

returns = daily_returns
mean_returns = returns.mean()
returns_covmat = returns.cov()

#%% Optimisation
# CMA-ES
num_assets = returns.shape[1]
initial_weights = np.ones(num_assets) / num_assets  
initial_sigma = 1
alpha_penalty = 100

es = cma.CMAEvolutionStrategy(initial_weights, initial_sigma)

mean_history = []

while not es.stop():
    candidate_solutions = es.ask()
    objective_values = [
        objective_function(s, mean_returns, returns_covmat, alpha_penalty)
        for s in candidate_solutions]
    es.tell(candidate_solutions, objective_values)
    es.disp()
    
    mean_history.append(es.mean)

cma_output = np.round(es.result.xbest, 5)
cma_solution = list(zip(stocks.split(), cma_output))
cma_sharpe = -es.result.fbest

# Portfolio Opt. (Benchmark)
from pypfopt.efficient_frontier import EfficientFrontier
ef = EfficientFrontier(mean_returns, returns_covmat)
weights = ef.max_sharpe(risk_free_rate = 0.0)
benchmark_solution = list(ef.clean_weights().items())
benchmark_sharpe = ef.portfolio_performance(risk_free_rate = 0.0)[2]

#%% Results
# Plotting the parameter history of top v parameters. 
# v = 2
# top_index = np.argsort(cma_output)[-v:][::-1]
# mean_history = np.array(mean_history)
# colors = ('r', 'b', 'g', 'o')

# plt.figure(figsize=(10, 5))
# for i in range(len(top_index)):
#     j = top_index[i]
#     stock = np.array(stocks.split(), dtype="str")[j].item()
#     plt.plot(mean_history[:, j], color = colors[i], label=f'Weight for {stock}')
#     plt.axhline(y=cma_output[j], color = colors[i] ,linestyle = '--', label=f'Best weight for {stock}')

# plt.xlabel('Generation')
# plt.ylabel('Weight value')
# plt.title(f'Evolution of Top {v} Weights')
# plt.legend()
# plt.show()

print("CMA-ES Solution:", cma_solution)
print("CMA-ES Sharpe Ratio:", cma_sharpe)
print("Benchmark Solution:", benchmark_solution)
print("Benchmark Sharpe Ratio:", benchmark_sharpe)

data = [cma_solution, benchmark_solution]
tickers = [d[0] for d in data[0]]  # Get the tickers from the first data entry
values1 = [d[1] for d in data[0]]  # Values from the first dataset
values2 = [d[1] for d in data[1]]  # Values from the second dataset

# Create a new figure
fig, ax = plt.subplots(figsize=(10, 4))

# Number of bars
n = len(tickers)
index = range(n)

bar_width = 0.35

bars1 = ax.bar(index, values1, bar_width, label='CMA-ES Weights', color = 'r')
bars2 = ax.bar([p + bar_width for p in index], values2, bar_width, label='Benchmark Weights', color = 'b')

ax.set_xlabel('Stock Ticker')
ax.set_ylabel('Weight')
ax.set_title('CMA-ES vs Benchmark – Medium Term Strategy')
ax.set_xticks([p + bar_width / 2 for p in index])
ax.set_xticklabels(tickers)
ax.legend()

plt.show()

# %%