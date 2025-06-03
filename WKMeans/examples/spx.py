import yfinance as yf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors as mcolors
from WKMeans.src.WKMean import WKMeans as wk
from WKMeans.src.WKUtils import reconstruct, window_lift, cum_rets

data = yf.download('^GSPC', multi_level_index=False,start='2001-01-01', end='2025-05-30')
data['returns'] = np.log(data.Close) - np.log(data.Close.shift(1))
data = data.dropna()

model = wk(k=2, max_iter=20, tol=1e-5, gamma=0.6, mmd_pairs=7)
distributions, idxs = window_lift(data.returns, h1=30, h2=3)
model.fit(distributions)
labels = model.predict(distributions)
labels = [int(l) for l in labels]
df = reconstruct(data.returns, idxs, labels)
cumulative_ret_regime = cum_rets(df, 'returns', 'cluster')
df = df[df['cluster'].notna()].copy()  # Drop NaN clusters

df['cluster'] = df['cluster'].astype(int)

# Calculate cumulative returns
df['cum_return_'] = (1 + df['returns']).fillna(0).cumprod()

# Compute rolling volatility per row
df['volatility'] =df['returns'].rolling(window=10).std().fillna(method='bfill')

# Compute mean volatility per cluster
regime_vols = df.groupby('cluster')['volatility'].mean()

# Normalize volatilities for colormap scaling
norm = mcolors.Normalize(vmin=regime_vols.min(), vmax=regime_vols.max())
cmap = plt.cm.Reds  # Use red colormap for volatility

# Setup plots
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# Plot cumulative return
ax[0].plot(df.index, df['cum_return_'], color='black', label='Cumulative Return')

# Regime-colored spans based on volatility
last_label = None
start_idx = None
for i in range(len(df)):
    label = df['cluster'].iloc[i]
    if label != last_label:
        if start_idx is not None:
            start = df.index[start_idx]
            end = df.index[i]
            color = cmap(norm(regime_vols[last_label]))
            ax[0].axvspan(start, end, color=color, alpha=0.3)
        start_idx = i
        last_label = label

# Final span
if start_idx is not None:
    start = df.index[start_idx]
    end = df.index[-1]
    color = cmap(norm(regime_vols[last_label]))
    ax[0].axvspan(start, end, color=color, alpha=0.3)

ax[0].set_title('Cumulative Return with Volatility-Colored Regimes')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Cumulative Market Returns')
ax[0].legend()
ax[0].grid(True)

# Plot cumulative return by cluster (assumes cum_rets returns a DataFrame)
cum_return_by_reg = cum_rets(df)
for regime in cum_return_by_reg.columns:
    ax[1].plot(cum_return_by_reg.index, cum_return_by_reg[regime], label=f'Cluster {regime}')

ax[1].set_title('Cumulative Return by Cluster')
ax[1].set_xlabel('Time Index')
ax[1].set_ylabel('Cumulative Return')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()


