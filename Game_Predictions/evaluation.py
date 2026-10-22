"""
Step 0: one time-based scorecard used for EVERY model change.

* Train strictly on the past, test strictly on the future (no random split,
  because neighbouring games share rolling-window information).
* Score the model the simulation actually uses (margin regression -> win
  probability) next to the logistic regression and simple baselines.
* Report precision / recall / F1 for "home win" (class 1) AND the
  probability metrics (Brier, log loss) that the win percentages depend on.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, log_loss)
from model import FEATURES, fit_win_model

# name -> (which games to score, fit models on seasons <= this)
TEST_SETS = {
    "2024-25 holdout": (lambda g: g["season"].between(2024, 2025) & (g["week"] <= 18), 2023),
    "2026 so far":     (lambda g: g["season"] == 2026, 2025),   # fit through 2025, like the sim does
}


def _score(label, y, p):
    """Metrics for one model. y = 1 if home won; p = predicted P(home win)."""
    y = np.asarray(y)
    p = np.asarray(p)
    pred = (p > 0.5).astype(int)
    acc = accuracy_score(y, pred)
    return {
        "model": label,
        "n": len(y),
        "acc": acc,
        "+/-": 1.96 * np.sqrt(acc * (1 - acc) / len(y)),   # ~95% interval half-width
        "prec": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
    }


def _fit_all(g, train_through):
    """Fit every model on seasons <= train_through; return what scoring needs."""
    train = g[g["season"] <= train_through]
    logit, scaler = fit_win_model(g, train_through)

    reg = LinearRegression().fit(train[FEATURES], train["margin"])
    margin_sd = (train["margin"] - reg.predict(train[FEATURES])).std()   # noise used for P(win)

    line_sd = None                       # Vegas benchmark (positive spread = home favoured)
    if "spread_line" in g.columns:
        tl = train.dropna(subset=["spread_line"])
        line_sd = (tl["margin"] - tl["spread_line"]).std()
    return scaler, logit, reg, margin_sd, train["result"].mean(), line_sd


def evaluate_models(games, verbose=True):
    """Score every model on each test set, always fitting on strictly earlier seasons."""
    need = FEATURES + ["home_score", "away_score", "result"]
    g = games.dropna(subset=need).copy()
    g["margin"] = g["home_score"] - g["away_score"]

    tables = {}
    for name, (mask_fn, train_through) in TEST_SETS.items():
        test = g[mask_fn(g)]
        if len(test) == 0:
            continue
        scaler, logit, reg, margin_sd, home_rate, line_sd = _fit_all(g, train_through)
        y = test["result"]
        rows = [
            _score("always home", y, np.full(len(test), home_rate)),
            _score("logistic", y, logit.predict_proba(scaler.transform(test[FEATURES]))[:, 1]),
            _score("margin -> P(win)", y, norm.cdf(reg.predict(test[FEATURES]) / margin_sd)),
        ]
        if line_sd is not None:
            tv = test.dropna(subset=["spread_line"])
            rows.append(_score("Vegas spread", tv["result"], norm.cdf(tv["spread_line"] / line_sd)))
        tables[name] = pd.DataFrame(rows).set_index("model")
        if verbose:
            print(f"\n=== {name} (fit on seasons <= {train_through}) ===")
            print(tables[name].round(3).to_string())
    return tables


if __name__ == "__main__":
    from data_cleaning import load_and_clean_games
    games, _ = load_and_clean_games()
    evaluate_models(games)