# NFL Prediction Models

Split from a single Jupyter notebook into two standalone pipelines so
they're readable directly on GitHub.

## `game\_prediction/`

Predicts the winner of every NFL game (and simulates final scores for the
upcoming season) from each team's rolling offensive/defensive EPA and pace.

* `data\_cleaning.py` — loads play-by-play + schedule data and builds the
per-team rolling EPA features, joined onto each historical game
* `model.py` — trains/evaluates a logistic regression win-probability model
* `simulation.py` — freezes end-of-season team ratings, projects next
season's schedule, and runs a Monte Carlo simulation for win% and
expected scores
* `main.py` — runs the full pipeline end to end

## `play\_calling/`

Predicts whether the next play will be a run or a pass from down,
distance, field position, game state, and each team's tendencies.

* `data\_cleaning.py` — loads play-by-play data and builds the trailing
pass-rate feature
* `model.py` — trains/evaluates an XGBoost run/pass classifier
* `simulation.py` — simulates play calls for one or more specific game
situations
* `main.py` — runs the full pipeline end to end

## Setup

```
pip install -r requirements.txt
```

Each `main.py` can be run directly, e.g.:

```
cd game\_prediction \&\& python main.py
cd play\_calling \&\& python main.py
```

## Notes

* Each pipeline pulls live data via `nflreadpy` at runtime — no data files
are checked into the repo, so a run always reflects the latest available
seasons (adjust the `SEASONS` constant in each `data\_cleaning.py` if you
want to pin specific years).
* `game\_prediction/simulation.py` carries over a small bug from the
original notebook: `totalResidSTD` is computed from `marginModel`'s
predictions instead of `totalModel`'s. It's flagged with a comment in
the code — worth fixing if you want that number to be meaningful.

