# NFL Prediction Model

This project applies machine learning to NFL play-by-play and schedule data through two standalone pipelines. The **Weekly Game Predictor** estimates win probabilities and final scores for every game, and uses them to project the 2026 season. The **Play Call Predictor** estimates whether the next play will be a run or a pass from the game situation. Both pipelines are validated on strictly out-of-time data, and Monte Carlo simulation turns model outputs into probability distributions.

---

## Objectives

**Weekly Game Predictor** — predict the winner of every NFL game and project final scores for the remaining 2026 schedule. Each team is rated by opponent-adjusted offensive and defensive efficiency (EPA per play), pace (plays per game), and rest days. A blend of a logistic classifier and a margin regression produces win probabilities, and a Monte Carlo simulation produces expected scores. Every model is benchmarked against a naive home-team baseline and the Vegas point spread.

**Play Call Predictor** — predict whether the next play will be a run or a pass, given down, distance, field position, score, time remaining, and each team's recent tendencies. A Monte Carlo simulation then turns the model's probability into a distribution of likely outcomes for a given situation.

---

## Project Structure

```
NFL-Game-Predictions-And-Play-Calling/
├── README.md
├── requirements_1.txt
├── sim_results_2026.csv           # 2026 game-by-game simulation output
├── Game_Predictions/
│   ├── data_cleaning.py           # loads pbp/schedule, builds opponent-adjusted rolling EPA features
│   ├── model.py                   # logistic regression win-probability model
│   ├── evaluation.py              # time-based scorecard (baseline, logistic, margin model, Vegas)
│   ├── window_experiment.py       # raw rolling EPA vs. ridge-adjusted ratings experiment
│   ├── simulation.py              # current ratings + 2026 season Monte Carlo simulation
│   ├── main.py                    # runs the full pipeline end to end
│   └── sim_results_2026_updated.csv
└── Play_Calling/
    ├── data_cleaning.py           # loads pbp, builds trailing pass-rate feature
    ├── model.py                   # XGBoost run/pass classifier
    ├── simulation.py              # situation-based Monte Carlo simulation
    └── main.py                    # runs the full pipeline end to end
```

| File | Purpose |
|---|---|
| `data_cleaning.py` | Pulls raw data via `nflreadpy`, cleans it with SQL (`pandasql`), and engineers the features each model uses |
| `model.py` | Trains and evaluates the prediction model and defines the `FEATURES` list shared with `simulation.py` and `evaluation.py` |
| `evaluation.py` | *(Game Predictions)* Scores every model on the same time-based holdouts so each change is comparable |
| `window_experiment.py` | *(Game Predictions)* Compares raw rolling EPA against opponent-adjusted ridge ratings |
| `simulation.py` | Runs the Monte Carlo simulation on top of the trained models, either projecting the remaining 2026 season (`Game_Predictions`) or a specific game situation (`Play_Calling`) |
| `main.py` | Entry point that runs the full pipeline (clean → train → evaluate → simulate) in order |

---

## Results & Analysis

### Weekly Game Predictor

**Data.** Play-by-play from 2014–2026 and schedules from 2015–2026. After filtering to regular-season run and pass plays with win probability between 5% and 95%, **341,899 of 596,385 raw plays** were kept. Of 3,110 games, **2,853** had complete rolling features. Tied games were dropped, so the target is a clean home win (1) or away win (0).

#### 1. Logistic regression (final time-ordered fold)

Trained with `TimeSeriesSplit(n_splits=5)`, scoring on the most recent fold (475 games), so the model never sees the future.

| Metric | Value |
|---|---|
| Accuracy | **0.62** |
| Precision / Recall / F1 (away win) | 0.61 / 0.52 / 0.56 |
| Precision / Recall / F1 (home win) | 0.63 / 0.70 / 0.66 |
| Brier score | 0.2269 |
| Log loss | 0.6441 |
| Naive "always pick home team" baseline | 0.533 |

Confusion matrix (rows = actual, columns = predicted; 0 = away win, 1 = home win):

```
[[116 106]
 [ 75 178]]
```

#### 2. Model scorecard (all models, same test games)

Each model is fit only on earlier seasons, then scored on later games. "±" is the 95% confidence half-width on accuracy.

**2024–25 holdout** (fit on seasons ≤ 2023, n = 543)

| Model | Accuracy | ± | Precision | Recall | F1 | Brier | Log loss |
|---|---|---|---|---|---|---|---|
| Always home | 0.536 | 0.042 | 0.536 | 1.000 | 0.698 | 0.249 | 0.691 |
| Logistic | 0.637 | 0.040 | 0.647 | 0.711 | 0.678 | 0.223 | 0.637 |
| Margin → P(win) | 0.637 | 0.040 | 0.645 | 0.718 | 0.680 | 0.222 | 0.635 |
| Vegas spread | **0.683** | 0.039 | 0.687 | 0.753 | 0.718 | **0.207** | **0.601** |

**2026 so far** (fit on seasons ≤ 2025, n = 92)

| Model | Accuracy | ± | Precision | Recall | F1 | Brier | Log loss |
|---|---|---|---|---|---|---|---|
| Always home | 0.511 | 0.102 | 0.511 | 1.000 | 0.676 | 0.251 | 0.695 |
| Logistic | 0.565 | 0.101 | 0.561 | 0.681 | 0.615 | 0.248 | 0.690 |
| Margin → P(win) | 0.565 | 0.101 | 0.559 | 0.702 | 0.623 | 0.246 | 0.685 |
| Vegas spread | **0.630** | 0.099 | 0.618 | 0.723 | 0.667 | **0.232** | **0.659** |

**Takeaways**

- On the 2024–25 holdout, both models beat the home-field baseline by about **10 points of accuracy** (63.7% vs. 53.6%) and improve Brier score and log loss, which means the probabilities themselves are better calibrated, not just the picks.
- The margin-regression model and the logistic model perform almost identically, so blending them (see Methodology) mainly adds stability.
- The Vegas spread remains the strongest benchmark (68.3% accuracy, Brier 0.207). The model closes part of the gap to the market but does not beat it.
- The 2026 sample is small (92 games, roughly ±10 points of accuracy), so those results are directional rather than conclusive.

#### 3. 2026 season projection

Of 272 scheduled games, 179 were still unplayed, and all 179 had complete team ratings. Out-of-sample residual spreads from the linear models are:

| Quantity | Residual STD |
|---|---|
| Point margin | **13.12 points** |
| Total points | **13.24 points** |

These are used as the noise terms in the per-game Monte Carlo simulation. Results for every remaining game (win probability, predicted score, predicted margin and total) are saved to `sim_results_2026_updated.csv`.

---

### Play Call Predictor (XGBoost)

Trained on 2015–2025 plays (444,027 of 446,069 rows kept after cleaning),
70/30 train/test split:

| Metric | Value |
|---|---|
| Accuracy | 0.73 |
| Precision / Recall / F1 (run) | 0.69 / 0.63 / 0.66 |
| Precision / Recall / F1 (pass) | 0.75 / 0.80 / 0.77 |
| Naive "always pick pass" baseline | 0.585 |

Confusion matrix (0 = run, 1 = pass):

```
[[35109 20207]
 [15878 62015]]
```

A ~15-point lift over the naive baseline, with the model doing
noticeably better on identifying passes than runs.

**Example situation simulation** (3rd & 4, own 35, down 3, 2 min left):
model predicted an 85.5% pass probability, and a 10,000-trial Monte Carlo
draw landed at 85.8% simulated pass rate / 14.2% run rate — consistent
with the model's probability, as expected.

---

## Methodology & Key Concepts

### Weekly Game Predictor

**Tools:** Python, `nflreadpy`, `pandasql` (SQL), `pandas`, `NumPy`, `SciPy`, `scikit-learn`.

#### Feature engineering

- **Expected Points Added (EPA)** measures how much each play changes a team's expected scoring. Only run and pass plays in the regular season with a pre-snap win probability between 0.05 and 0.95 are used, which removes garbage-time plays.
- **Per team, per week:** offensive EPA per play, defensive EPA per play allowed, and plays per game (pace).
- **Opponent adjustment (ridge regression).** Raw EPA ignores schedule strength, so each team's rating is fit with a ridge regression over the previous **17 league-weeks**:

$$\text{off\_epa}_{g} = \beta_0 + \text{off}[\text{team}] + \text{def}[\text{opp}] + \varepsilon$$

  Each row is one offense in one game. The ridge penalty $\alpha = 10$ acts like 10 "pseudo-games" of league-average prior, which keeps ratings stable early in the season:

$$\min_{\beta}\ \sum_i \left(y_i - \hat{y}_i\right)^2 + \alpha \lVert \beta \rVert_2^2$$

- **No data leakage.** Ratings for a given week are fit only on weeks strictly before it. Rest days (`home_rest`, `away_rest`) are added as context features.
- **Model features (8):** home/away offensive EPA, defensive EPA, plays per game, and rest days.

#### Models

| Model | Role |
|---|---|
| **Logistic regression** (standardized features) | Predicts $P(\text{home win})$ directly |
| **Linear regression (margin)** | Predicts expected home point margin |
| **Linear regression (total)** | Predicts expected combined points |

The logistic model is
$$P(\text{home win}) = \frac{1}{1 + e^{-(\beta_0 + \boldsymbol{\beta}^\top \mathbf{x})}}$$

The margin model's probability assumes normally distributed errors with residual standard deviation $\sigma_m$:

$$P_{\text{margin}} = \Phi\!\left(\frac{\hat{m}}{\sigma_m}\right)$$

The final win probability is an equal blend of the two:

$$p = 0.5\,P_{\text{logistic}} + 0.5\,P_{\text{margin}}$$

#### Monte Carlo score simulation

1. Convert the blended probability back to a margin centre: $\mu_m = \sigma_m\,\Phi^{-1}(p)$.
2. For each game, draw 1,000 simulations (fixed seed) of margin and total: $M \sim \mathcal{N}(\mu_m, \sigma_m^2)$ and $T \sim \mathcal{N}(\hat{T}, \sigma_T^2)$.
3. Convert each draw to team scores:

$$\text{Home} = \frac{T + M}{2}, \qquad \text{Away} = \frac{T - M}{2}$$

4. Average the simulated scores and snap them to a realistic NFL score (any score making up at least 0.4% of historical team scores, which removes values like 1, 2, 4 and 5). The projected favourite is always given the higher score.

#### Evaluation

- **Time-based validation only.** Models train on the past and test on the future, never on a random split, because neighbouring games share rolling-window information.
- **Accuracy**, **precision / recall / F1** (for the home-win class), and a 95% accuracy interval: $\pm 1.96\sqrt{\hat{a}(1-\hat{a})/n}$.
- **Brier score** (mean squared error of the probabilities, lower is better): $\frac{1}{n}\sum_i (p_i - y_i)^2$
- **Log loss** (penalizes confident wrong predictions, lower is better): $-\frac{1}{n}\sum_i \left[y_i \ln p_i + (1-y_i)\ln(1-p_i)\right]$
- **Benchmarks:** an "always pick home" baseline and the **Vegas spread**, converted to a probability with $\Phi(\text{spread}/\sigma_{\text{line}})$.
- **Ratings experiment** (`window_experiment.py`): raw rolling EPA vs. ridge-adjusted ratings at different lookbacks and penalty strengths, trained on seasons ≤ 2023 and tested on 2024–25.

### Play Call Predictor

**Tools:** Python, `nflreadpy`, `pandasql`, `XGBoost`, `scikit-learn`. A gradient-boosted tree classifier predicts run vs. pass from situational features and a trailing team pass-rate feature. Monte Carlo sampling of the predicted probability produces a distribution of outcomes for a given situation.

---

## Setup

```bash
pip install -r requirements_1.txt
```

Each pipeline's `main.py` can be run directly:

```bash
cd Game_Predictions && python main.py
cd Play_Calling && python main.py
```

---

## Notes

- Each pipeline pulls live data via `nflreadpy` at runtime, so a run always reflects the latest available games. Adjust the `SEASONS` constants in each `data_cleaning.py` to pin specific years. Because the 2026 season is in progress, results above reflect the data at the time of the run and will shift as more games are played.
- Team ratings roll forward automatically on each re-run, so the simulation only covers games that have not yet been played.
