# ⚠️ Warnings — Read This Before Trusting Any Result

Most amateur quant projects "work" only because they are quietly cheating. This
page lists the ways that happens and how this repo defends against each. **If you
remember one thing: a backtest is a hypothesis, not a promise.**

---

## 1. Look-ahead bias
**What it is:** using information at time `t` that would not have been known until
after `t`. Example: computing a feature with `close[t+1]`, or normalizing the whole
series (mean/std over all time) before splitting.

**Defenses here:**
- Features only use rolling windows ending at `t`.
- Scalers/encoders are `fit` on the **training slice only**, then applied to test.
- The label is the *only* thing allowed to look forward, and labelled rows at the
  end of the series (where the future is unknown) are dropped.

## 2. Data leakage
**What it is:** test information bleeding into training — e.g. shuffling time-series
rows, computing global statistics, or tuning hyperparameters on the test set.

**Defenses here:** time-ordered splits, walk-forward folds, and a held-out test set
that is touched **once**, at the very end.

## 3. Overfitting
**What it is:** a model that memorizes noise. Symptom: great train/validation score,
poor live/test score; metrics that collapse when you change the date range slightly.

**Defenses here:**
- Start with **Logistic Regression** as a hard-to-beat baseline.
- Regularization, shallow trees, early stopping for boosters.
- Judge models on **walk-forward out-of-sample** performance, not in-sample fit.
- Be suspicious of any accuracy far above ~56% on daily direction.

## 4. Survivorship bias
**What it is:** testing only on companies that still exist today. Delisted/bankrupt
tickers vanish from most free data, inflating returns.

**Mitigation:** be explicit that yfinance gives a *survivorship-biased* universe.
For serious work, use a point-in-time, delisting-inclusive dataset. Treat
single-name results as illustrative, not as a tradable universe.

## 5. Unrealistic backtests
Common ways backtests lie:
- **No costs.** Real trading pays commissions, spreads, and slippage. We model
  `TRANSACTION_COST_BPS` + `SLIPPAGE_BPS` on every position change.
- **Trading on the signal bar's close using that bar's data.** We act on the *next*
  bar to avoid same-bar leakage.
- **Ignoring liquidity/capacity.** Small-cap edges often can't be traded at size.
- **Cherry-picked date ranges.** Always report the full sample and sub-periods.

## 6. Multiple-testing / p-hacking
Try 200 feature/model combos and one will look great by luck. Keep an experiment
log (`model_runs` table), pre-register what "good" means, and validate the winner
on a date range it never influenced.

## 7. Prediction confidence ≠ probability of profit
A classifier's `predict_proba` is a model-internal score, not a calibrated,
risk-adjusted edge. Calibrate it (Phase 5), and remember a 60% "confidence" can
still lose money after costs if the winners are small and losers are large.

## 8. Non-stationarity / regime change
Markets change. A model trained on 2015–2019 may fail in 2020 or 2022. Re-fit
regularly (walk-forward already simulates this) and monitor live performance.

---

## The bottom line
This repository is a **research and education platform**. It is **not** financial
advice, and it makes **no guarantee of profit**. Past performance — especially
backtested performance — does not predict future results. Do not trade real money
based on its output without independent validation and professional risk management.
