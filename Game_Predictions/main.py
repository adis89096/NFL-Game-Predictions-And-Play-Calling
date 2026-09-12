"""
Entry point for the Weekly Game Predictor: cleans the data, trains the
win-probability model, and runs the 2026 season Monte Carlo simulation.
"""
import pandas as pd

from data_cleaning import load_and_clean_games
from model import train_game_predictor
from simulation import (
    build_frozen_ratings, build_2026_matchups,
    train_margin_total_models, predict_2026, simulate_scores
)

if __name__ == "__main__":
    games, teamWeek = load_and_clean_games()

    print("\n=== Training win-probability model ===")
    train_game_predictor(games)

    print("\n=== Building 2026 season simulation ===")
    frozen_ratings = build_frozen_ratings(teamWeek)
    simGames = build_2026_matchups(frozen_ratings)

    marginModelFinal, totalModelFinal, marginResidSTD, totalResidSTD = train_margin_total_models(games)
    simFeatures = predict_2026(simGames, marginModelFinal, totalModelFinal)

    pd.set_option("display.max_rows", None)
    simResults = simulate_scores(simFeatures, marginResidSTD, totalResidSTD)
    print(simResults)
