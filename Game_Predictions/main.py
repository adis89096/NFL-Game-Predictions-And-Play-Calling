"""
Entry point for the Weekly Game Predictor: cleans the data, trains the
win-probability model, and runs the 2026 season Monte Carlo simulation.
"""
import pandas as pd
from pathlib import Path
from data_cleaning import load_and_clean_games
from model import train_game_predictor, fit_win_model
from evaluation import evaluate_models
from simulation import (build_current_ratings, build_2026_matchups, train_margin_total_models, 
                        predict_2026, simulate_scores, build_allowed_scores)

if __name__ == "__main__":
    games, teamWeek = load_and_clean_games()

    print("\n=== Training win-probability model ===")
    train_game_predictor(games)

    print("\n=== Original Evaluation Scorecard ===")
    evaluate_models(games)

    print("\n=== Building 2026 season simulation ===")
    frozen_ratings = build_current_ratings(teamWeek)
    simGames = build_2026_matchups(frozen_ratings)

    marginModelFinal, totalModelFinal, marginResidSTD, totalResidSTD = train_margin_total_models(games)
    winModel, winScaler = fit_win_model(games)
    simFeatures = predict_2026(simGames, marginModelFinal, totalModelFinal, winModel, winScaler)

    pd.set_option("display.max_rows", None)
    allowedScores = build_allowed_scores(games)
    print("Allowed scores: ", allowedScores)

    simResults = simulate_scores(simFeatures, marginResidSTD, totalResidSTD, allowed=allowedScores)
    simResults = simResults.merge(simFeatures[["game_id", "pred_margin", "pred_total"]], on="game_id")
    simResults = simResults.sort_values(["week", "game_id"]).round(3)

    outPath = Path(__file__).parent / "sim_results_2026_updated.csv"
    simResults.to_csv(outPath, index=True)
    print(f"Saved {len(simResults)} game -> {outPath}")
