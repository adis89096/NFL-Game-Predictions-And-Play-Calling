"""
Entry point for the Play Call Predictor: cleans the data, trains the
run/pass model, and simulates a few example game situations.
"""
from data_cleaning import load_and_clean_plays
from model import train_play_predictor
from simulation import simulate_situation, simulate_situations, NSIMS

if __name__ == "__main__":
    pbp = load_and_clean_plays()

    print("\n=== Training play-call model ===")
    model = train_play_predictor(pbp)

    print("\n=== Simulating example situations ===")
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
