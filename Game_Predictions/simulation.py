"""
Projects the 2026 season using "frozen" end-of-2025 team ratings, then runs
a Monte Carlo simulation over predicted margin/total points to get win
probabilities and expected scores for every game.
"""
import numpy as np
import pandas as pd
import nflreadpy as nfl
from pandasql import sqldf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from model import FEATURES

NSIMS = 1000


def build_frozen_ratings(teamWeek):
    """Freeze each team's rating as the average of its last 3 played weeks."""
    return sqldf("""
        WITH ranked AS (
            SELECT team, season, week, off_epa, def_epa, plays,
                ROW_NUMBER() OVER (PARTITION BY team ORDER BY season DESC, week DESC) AS rn
            FROM teamWeek
        )
        SELECT team,
                AVG(off_epa) AS off_epa_frozen,
                AVG(def_epa) AS def_epa_frozen,
                AVG(plays) AS plays_frozen
        FROM ranked
        WHERE rn <= 3
        GROUP BY team
    """, locals())


def build_2026_matchups(frozen_ratings):
    """Attach frozen home/away ratings onto the 2026 schedule."""
    schedule2026 = nfl.load_schedules([2026]).to_pandas()

    home = frozen_ratings.rename(columns={
        "team": "home_team", "off_epa_frozen": "home_off_epa",
        "def_epa_frozen": "home_def_epa", "plays_frozen": "home_plays"
    })
    away = frozen_ratings.rename(columns={
        "team": "away_team", "off_epa_frozen": "away_off_epa",
        "def_epa_frozen": "away_def_epa", "plays_frozen": "away_plays"
    })

    simGames = sqldf("""
        SELECT s.game_id, s.season, s.week, s.home_team, s.away_team,
               h.home_off_epa, h.home_def_epa, h.home_plays,
               a.away_off_epa, a.away_def_epa, a.away_plays
        FROM schedule2026 s
        LEFT JOIN home h ON s.home_team = h.home_team
        LEFT JOIN away a ON s.away_team = a.away_team
    """, locals())

    print(len(schedule2026), "->", simGames["home_off_epa"].notna().sum(), "games with complete ratings")
    return simGames


def train_margin_total_models(games):
    """
    Fit linear regressions predicting home margin and combined total points.
    Reports out-of-sample residual spread (used as simulation noise), then
    refits on all historical data for the final 2026 predictions.
    """
    gameReg = games.dropna(subset=FEATURES + ["home_score", "away_score"]).copy()
    gameReg["margin"] = gameReg["home_score"] - gameReg["away_score"]
    gameReg["total"] = gameReg["home_score"] + gameReg["away_score"]

    X = gameReg[FEATURES]
    yMargin = gameReg["margin"]
    yTotal = gameReg["total"]

    X_train, X_test, ym_train, ym_test, yt_train, yt_test = train_test_split(
        X, yMargin, yTotal, random_state=42, test_size=0.3
    )

    marginModel = LinearRegression().fit(X_train, ym_train)
    totalModel = LinearRegression().fit(X_train, yt_train)
    marginResidSTD = (ym_test - marginModel.predict(X_test)).std()
    # NOTE: carried over from the original notebook — this uses marginModel's
    # predictions rather than totalModel's, so it isn't really measuring the
    # total-points model's residual spread. Left as-is during the split.
    totalResidSTD = (yt_test - marginModel.predict(X_test)).std()

    print(f"Margin Residual STD: {marginResidSTD:.2f} points")
    print(f"Total Residual STD: {totalResidSTD:.2f} points")

    # Refit on ALL historical data for the actual 2026 point predictions —
    # the train/test split above was only to get an honest (out-of-sample) residual spread.
    marginModelFinal = LinearRegression().fit(X, yMargin)
    totalModelFinal = LinearRegression().fit(X, yTotal)

    return marginModelFinal, totalModelFinal, marginResidSTD, totalResidSTD


def predict_2026(simGames, marginModelFinal, totalModelFinal, target_weeks=None):
    """Apply the fitted margin/total models to the 2026 matchups."""
    simFeatures = simGames.dropna(subset=FEATURES).copy()
    if target_weeks is not None:
        simFeatures = simFeatures[simFeatures["week"].isin(target_weeks)]

    simFeatures["pred_margin"] = marginModelFinal.predict(simFeatures[FEATURES])
    simFeatures["pred_total"] = totalModelFinal.predict(simFeatures[FEATURES])
    return simFeatures


def simulate_scores(simFeatures, marginResidSTD, totalResidSTD, nsims=NSIMS, seed=42):
    """
    Monte Carlo simulate final scores from the predicted margin/total,
    using the residual spread as noise, to get win probabilities and
    expected scores for each game.
    """
    rng = np.random.default_rng(seed)
    result = []

    for _, row in simFeatures.iterrows():
        margins = rng.normal(row["pred_margin"], marginResidSTD, nsims)
        totals = rng.normal(row["pred_total"], totalResidSTD, nsims)

        homeScore = np.clip(np.round((totals + margins) / 2), 0, None)
        awayScore = np.clip(np.round((totals - margins) / 2), 0, None)

        result.append({
            "game_id": row["game_id"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "week": row["week"],
            "home_win_pct": (homeScore > awayScore).mean(),
            "away_win_pct": (awayScore > homeScore).mean(),
            "tie_pct": (homeScore == awayScore).mean(),
            "avg_home_score": homeScore.mean(),
            "avg_away_score": awayScore.mean()
        })

    return pd.DataFrame(result)


if __name__ == "__main__":
    from data_cleaning import load_and_clean_games

    games, teamWeek = load_and_clean_games()
    frozen_ratings = build_frozen_ratings(teamWeek)
    print(frozen_ratings)

    simGames = build_2026_matchups(frozen_ratings)
    marginModelFinal, totalModelFinal, marginResidSTD, totalResidSTD = train_margin_total_models(games)

    simFeatures = predict_2026(simGames, marginModelFinal, totalModelFinal)
    print(simFeatures[["game_id", "week", "home_team", "away_team", "pred_margin", "pred_total"]])

    pd.set_option("display.max_rows", None)
    simResults = simulate_scores(simFeatures, marginResidSTD, totalResidSTD)
    print(simResults)
