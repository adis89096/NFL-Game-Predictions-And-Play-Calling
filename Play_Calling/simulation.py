"""
Monte Carlo simulation of run/pass calls for one or more specific game
situations, using the trained XGBoost play predictor's probabilities.
"""
import numpy as np
import pandas as pd

from model import FEATURES

NSIMS = 10000


def simulate_situation(model, situation, nsims=NSIMS, seed=42):
    """
    Simulate play calls for a single situation (a dict of feature values).

    Returns
    -------
    pass_prob : float — the model's predicted probability of a pass
    sim_pass_pct, sim_run_pct : float — simulated pass/run rates over `nsims` draws
    """
    situation_df = pd.DataFrame([situation])[FEATURES]

    pass_prob = model.predict_proba(situation_df)[0, 1]

    rng = np.random.default_rng(seed)
    draws = rng.random(nsims) < pass_prob
    sim_pass_pct = draws.mean()
    sim_run_pct = 1 - sim_pass_pct

    return pass_prob, sim_pass_pct, sim_run_pct


def simulate_situations(model, situations, nsims=NSIMS, seed=42):
    """
    Same as `simulate_situation`, but for a list of situation dicts at once.

    Returns a DataFrame with one row per situation.
    """
    situations_df = pd.DataFrame(situations)[FEATURES]
    pass_probs = model.predict_proba(situations_df)[:, 1]

    rng = np.random.default_rng(seed)
    results = []
    for i, p in enumerate(pass_probs):
        draws = rng.random(nsims) < p
        results.append({
            "situation": i,
            "model_pass_prob": p,
            "sim_pass_pct": draws.mean(),
            "sim_run_pct": 1 - draws.mean()
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    from data_cleaning import load_and_clean_plays
    from model import train_play_predictor

    pbp = load_and_clean_plays()
    model = train_play_predictor(pbp)

    print("features:", FEATURES)

    situation = {
        "down": 3, "ydstogo": 4, "yardline_100": 35, "score_differential": -3,
        "game_seconds_remaining": 120, "qtr": 4, "shotgun": 1, "no_huddle": 0,
        "pass_rate_roll": 0.58
    }
    pass_prob, sim_pass_pct, sim_run_pct = simulate_situation(model, situation)
    print(f"Model's pass probability: {pass_prob:.3f}")
    print(f"Simulated pass rate over {NSIMS} trials: {sim_pass_pct:.3f}")
    print(f"Simulated run rate over {NSIMS} trials:  {sim_run_pct:.3f}")

    situations = [
        {"down": 1, "ydstogo": 10, "yardline_100": 75, "score_differential": 0,
         "game_seconds_remaining": 3500, "qtr": 1, "shotgun": 0, "no_huddle": 0, "pass_rate_roll": 0.55},
        {"down": 3, "ydstogo": 8, "yardline_100": 40, "score_differential": -7,
         "game_seconds_remaining": 600, "qtr": 4, "shotgun": 1, "no_huddle": 1, "pass_rate_roll": 0.62},
        {"down": 2, "ydstogo": 2, "yardline_100": 5, "score_differential": 3,
         "game_seconds_remaining": 200, "qtr": 4, "shotgun": 0, "no_huddle": 0, "pass_rate_roll": 0.40}
    ]
    print(simulate_situations(model, situations))
