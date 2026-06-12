# Data Dictionary

Every column the feature pipeline can produce, with its definition and the window
it uses. All features at row `t` use only data available **at or before** the close
of bar `t` (no look-ahead). `C` = adjusted close, `V` = volume.

## Price / return features (`features/returns.py`)
| Feature | Definition |
| --- | --- |
| `ret_1` | Simple return: `C[t]/C[t-1] - 1` |
| `log_ret_1` | Log return: `ln(C[t]/C[t-1])` |
| `ret_5`, `ret_10`, `ret_21` | N-day simple returns (≈ week, 2 weeks, month) |
| `lag_ret_1..lag_ret_5` | Previous-day returns shifted 1..5 days |
| `mom_10`, `mom_21`, `mom_63` | Momentum: `C[t]/C[t-n] - 1` |

## Trend features (`features/technical.py`)
| Feature | Definition |
| --- | --- |
| `sma_10`, `sma_20`, `sma_50`, `sma_200` | Simple moving average of `C` |
| `ema_12`, `ema_26` | Exponential moving average of `C` |
| `price_to_sma_50` | `C[t] / sma_50 - 1` (distance from trend) |
| `sma_20_50_cross` | `sma_20 - sma_50` normalized by `C` |

## Oscillators / bands (`features/technical.py`)
| Feature | Definition |
| --- | --- |
| `rsi_14` | Relative Strength Index, 14-day (Wilder smoothing) |
| `macd` | `ema_12 - ema_26` |
| `macd_signal` | 9-day EMA of `macd` |
| `macd_hist` | `macd - macd_signal` |
| `bb_mid_20` | 20-day SMA (Bollinger middle band) |
| `bb_upper_20`, `bb_lower_20` | `bb_mid ± 2·rolling_std(C, 20)` |
| `bb_pctb` | `%B`: position of `C` within the bands [0,1+] |
| `bb_width` | `(upper - lower) / mid` (volatility proxy) |

## Volatility / volume (`features/volatility.py`)
| Feature | Definition |
| --- | --- |
| `vol_10`, `vol_21` | Rolling std of `ret_1` over n days (realized vol) |
| `vol_ratio` | `vol_10 / vol_21` (short vs. long vol) |
| `atr_14` | Average True Range, 14-day |
| `vol_chg_5` | `V[t] / rolling_mean(V, 5) - 1` |
| `dollar_vol` | `C[t] * V[t]` (liquidity proxy), log-scaled |

## Market / relative features (`features/market.py`)
| Feature | Definition |
| --- | --- |
| `mkt_ret_1_SPY` | SPY 1-day return |
| `mkt_ret_1_QQQ` | QQQ 1-day return |
| `excess_ret_1` | `ret_1 - mkt_ret_1_SPY` (stock minus market) |
| `beta_63` | Rolling 63-day beta of stock returns vs. SPY |
| `rel_strength_SPY` | Ratio of stock to SPY cumulative return, normalized |
| `sector_excess_ret_1` | `ret_1 - sector_ETF_ret_1` (if sector mapped) |

## Labels (`features/pipeline.py`)
| Label | Task | Definition |
| --- | --- | --- |
| `y_dir_1` | classification | `1` if `C[t+1] > C[t]` else `0` (**default target**) |
| `y_dir_5` | classification | `1` if `C[t+5] > C[t]` else `0` |
| `y_fwd_ret_1` | regression | `C[t+1]/C[t] - 1` |
| `y_fwd_vol_5` | regression | realized vol of next 5 days' returns |

> The forward-looking label columns are dropped for rows near the end of the series
> where the future is unavailable. **Only label columns may reference the future.**
