"""
Projects the 2026 season using "frozen" end-of-2025 team ratings, then runs
a Monte Carlo simulation over predicted margin/total points to get win
probabilities and expected scores for every game.
"""
import numpy as np
import pandas as pd
import nflreadpy as nfl
from pandasql import sqldf
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from model import FEATURES, fit_win_model
from data_cleaning import (ROLL_WINDOW, ADJUST_OPPONENTS, ADJ_LOOKBACK_WEEKS, ADJ_ALPHA, current_adjusted_ratings)

NSIMS = 1000


def build_current_ratings(teamWeek, window=ROLL_WINDOW, adjust=ADJUST_OPPONENTS, adjLookback=ADJ_LOOKBACK_WEEKS, adjAlpha=ADJ_ALPHA):
    """
    Each team's rating = average of its last 3 PLAYED weeks.
    Since teamWeek now includes 2026, this is a live rating: it rolls
    forward every time you re-run after new games are played.
    """
    ratings = sqldf(f"""
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
        WHERE rn <= {int(window)}
        GROUP BY team
    """, locals())

    if adjust:
        adj = current_adjusted_ratings(teamWeek, adjLookback, adjAlpha)
        ratings = (ratings.drop(columns=["off_epa_frozen", "def_epa_frozen"]).merge(adj, on="team", how="inner"))

    return ratings



def build_2026_matchups(frozen_ratings):
    """Attach frozen home/away ratings onto the 2026 schedule."""
    
    schedule2026 = nfl.load_schedules([2026]).to_pandas()
    nTotal = len(schedule2026)
    schedule2026 = schedule2026[schedule2026["home_score"].isna()].copy()
    print(f"{nTotal} games in schedule -> {len(schedule2026)} unplayed")


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
               s.home_rest, s.away_rest,
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

    # Time-based split: fit on the past, measure the noise on the future.
    isTrain = gameReg["season"] <= 2023
    X_train, X_test = X[isTrain], X[~isTrain]
    ym_train, ym_test = yMargin[isTrain], yMargin[~isTrain]
    yt_train, yt_test = yTotal[isTrain], yTotal[~isTrain]

    marginModel = LinearRegression().fit(X_train, ym_train)
    totalModel = LinearRegression().fit(X_train, yt_train)
    marginResidSTD = (ym_test - marginModel.predict(X_test)).std()
    totalResidSTD = (yt_test - totalModel.predict(X_test)).std()

    print(f"Margin Residual STD: {marginResidSTD:.2f} points")
    print(f"Total Residual STD: {totalResidSTD:.2f} points")

    # Refit on ALL historical data for the actual 2026 point predictions —
    # the train/test split above was only to get an honest (out-of-sample) residual spread.
    marginModelFinal = LinearRegression().fit(X, yMargin)
    totalModelFinal = LinearRegression().fit(X, yTotal)

    return marginModelFinal, totalModelFinal, marginResidSTD, totalResidSTD


def predict_2026(simGames, marginModelFinal, totalModelFinal, winModel, winScaler, target_weeks=None):
    """Apply the fitted margin/total models to the 2026 matchups.
       margin/total regression -> expected points, logistic model -> P(home win)
    """
    simFeatures = simGames.dropna(subset=FEATURES).copy()
    if target_weeks is not None:
        simFeatures = simFeatures[simFeatures["week"].isin(target_weeks)]

    simFeatures["pred_margin"] = marginModelFinal.predict(simFeatures[FEATURES])
    simFeatures["pred_total"] = totalModelFinal.predict(simFeatures[FEATURES])
    simFeatures["p_home_win"] = winModel.predict_proba(winScaler.transform(simFeatures[FEATURES]))[:, 1]
    return simFeatures

def build_allowed_scores(games, min_share=0.004):
    """
    Learn which final scores are 'realistic' from history: any score that makes up
    at least min_share (0.4%) of all team-scores since 2015 is allowed.
    This drops 1, 2, 4, 5, ... automatically without a hand-typed list.
    """
    allScores = pd.concat([games["home_score"], games["away_score"]]).dropna().astype(int)
    share = allScores.value_counts(normalize=True)
    return sorted(int(s) for s in share[share >= min_share].index)

def snap_score(x, allowed):
    """Nearest allowed score to x (ties go to the lower score)."""
    return min(allowed, key=lambda s: (abs(s - x), s))

def next_allowed_above(x, allowed):
    """Smallest allowed score strictly greater than x (falls back to x + 1)."""
    higher = [s for s in allowed if s > x]
    return min(higher) if higher else int(x) + 1

def simulate_scores(simFeatures, marginResidSTD, totalResidSTD, nsims=NSIMS, seed=42, allowed=None):
    """
    Monte Carlo simulate final scores from the predicted margin/total,
    using the residual spread as noise, to get win probabilities and
    expected scores for each game.
    """
    if allowed is None:
        allowed = list(range(0, 100))
    rng = np.random.default_rng(seed)
    result = []

    for _, row in simFeatures.iterrows():
        pMargin = float(norm.cdf(row["pred_margin"] / marginResidSTD))
        p = 0.5 * float(row["p_home_win"]) + 0.5 * pMargin
        # margin centre implied by the logistic probability (clip avoids +/-inf)
        impliedMargin = marginResidSTD * norm.ppf(np.clip(p, 0.01, 0.99))

        margins = rng.normal(impliedMargin, marginResidSTD, nsims)
        totals = rng.normal(row["pred_total"], totalResidSTD, nsims)

        homeScore = np.clip(np.round((totals + margins) / 2), 0, None)
        awayScore = np.clip(np.round((totals - margins) / 2), 0, None)

        homeFav = p >= 0.5
        winner = row["home_team"] if homeFav else row["away_team"]
        predHome = snap_score(homeScore.mean(), allowed)
        predAway = snap_score(awayScore.mean(), allowed)
        if homeFav and predHome <= predAway:
            predHome = next_allowed_above(predAway, allowed)
        elif not homeFav and predHome <= predAway:
            predAway = next_allowed_above(predHome, allowed)

        result.append({
            "game_id": row["game_id"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "week": row["week"],
            "home_win_pct": p,
            "away_win_pct": 1.0 - p,
            "tie_pct": 0.0,
            "pred_home_score": predHome,
            "pred_away_score": predAway
        })

    return pd.DataFrame(result)


if __name__ == "__main__":
    from data_cleaning import load_and_clean_games

    games, teamWeek = load_and_clean_games()
    frozen_ratings = build_current_ratings(teamWeek)
    print(frozen_ratings)

    simGames = build_2026_matchups(frozen_ratings)
    marginModelFinal, totalModelFinal, marginResidSTD, totalResidSTD = train_margin_total_models(games)

    winModel, winScaler = fit_win_model(games)
    simFeatures = predict_2026(simGames, marginModelFinal, totalModelFinal, winModel, winScaler)
    print(simFeatures[["game_id", "week", "home_team", "away_team", "pred_margin", "pred_total"]])

    pd.set_option("display.max_rows", None)
    simResults = simulate_scores(simFeatures, marginResidSTD, totalResidSTD)
    print(simResults)
