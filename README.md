# NFL Prediction Models

Two machine learning pipelines built on NFL play-by-play and schedule data,
split from a single Jupyter notebook into standalone Python packages so
they're readable directly on GitHub.

## Objective

**Weekly Game Predictor** — predict the winner of every NFL game, and
project final scores for the upcoming season, using each team's rolling
offensive/defensive efficiency (EPA) and pace (plays per game).

**Play Call Predictor** — predict whether the next play will be a run or a
pass, given down, distance, field position, score, time remaining, and
each team's recent tendencies. A Monte Carlo simulation then turns the
model's probability into a distribution of likely outcomes for a given
situation.

## Analysis / Results

### Weekly Game Predictor (Logistic Regression)

Trained on 2015–2025 games (2,893 of 3,018 rows kept after dropping
incomplete rolling-feature rows), 70/30 train/test split:

| Metric | Value |
|---|---|
| Accuracy | 0.60 |
| Precision / Recall / F1 (away win) | 0.57 / 0.45 / 0.50 |
| Precision / Recall / F1 (home win) | 0.61 / 0.72 / 0.66 |
| Naive "always pick home team" baseline | 0.547 |

Confusion matrix (rows = actual, columns = predicted; 0 = away win, 1 = home win):

```
[[177 216]
 [134 341]]
```

The model beats the home-field baseline by about 5 points of accuracy —
a modest but real edge from the rolling EPA/pace features.

**2026 season projection:** all 272 scheduled games had complete ratings
to simulate. Linear regression models for point margin and total points
gave an out-of-sample residual spread of **13.86 points (margin)** and
**14.20 points (total)**, which is used as the noise term in the
per-game Monte Carlo simulation of final scores and win probabilities.

> Note: `totalResidSTD` is computed from `marginModel`'s residuals rather
> than `totalModel`'s in the original notebook — carried over as-is (see
> the comment in `game_prediction/simulation.py`), so treat the total-points
> spread as approximate.

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

## Project Structure

```
nflProject/
├── README.md
├── requirements.txt
├── game_prediction/
│   ├── data_cleaning.py   # loads pbp/schedule, builds rolling EPA features
│   ├── model.py            # logistic regression win-probability model
│   ├── simulation.py       # frozen ratings + 2026 season Monte Carlo sim
│   └── main.py              # runs the full pipeline end to end
└── play_calling/
    ├── data_cleaning.py   # loads pbp, builds trailing pass-rate feature
    ├── model.py            # XGBoost run/pass classifier
    ├── simulation.py       # situation-based Monte Carlo simulation
    └── main.py              # runs the full pipeline end to end
```

- **`data_cleaning.py`** — pulls raw data via `nflreadpy`, cleans it with
  SQL (via `pandasql`), and engineers the rolling features each model uses
- **`model.py`** — trains and evaluates the prediction model, defines the
  `FEATURES` list used by both `model.py` and `simulation.py`
- **`simulation.py`** — runs the Monte Carlo simulation on top of the
  trained model, either projecting a full season (`game_prediction`) or a
  handful of specific situations (`play_calling`)
- **`main.py`** — entry point that runs the full pipeline (clean → train →
  simulate) in order

## Setup

```
pip install -r requirements.txt
```

Each `main.py` can be run directly, e.g.:

```
cd game_prediction && python main.py
cd play_calling && python main.py
```

## Notes

- Each pipeline pulls live data via `nflreadpy` at runtime — no data files
  are checked into the repo, so a run always reflects the latest available
  seasons (adjust the `SEASONS` constant in each `data_cleaning.py` if you
  want to pin specific years). This also means metrics above were captured
  at the time the notebook was originally run and may shift slightly on a
  fresh pull as more games are played.
- `game_prediction/simulation.py` carries over a small bug from the
  original notebook: `totalResidSTD` is computed from `marginModel`'s
  predictions instead of `totalModel`'s. It's flagged with a comment in
  the code — worth fixing if you want that number to be meaningful.
