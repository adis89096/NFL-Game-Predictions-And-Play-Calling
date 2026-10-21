"""
Data cleaning and feature engineering for the Weekly Game Predictor.

Loads play-by-play and schedule data from nflreadpy, builds per-team
rolling offensive/defensive EPA features (a trailing 3-week average using
only prior weeks, so there's no leakage), and joins those onto each game
to produce one row per game with home/away features and the actual result.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import nflreadpy as nfl
from pandasql import sqldf

SEASONS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
SCHEDULE_SEASONS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
ROLL_WINDOW = 6
REG_SEASON = True
WIN_PROBABILITY_BOUND = (0.05, 0.95)
ADJUST_OPPONENTS = True
ADJ_LOOKBACK_WEEKS = 17   # league-weeks of games used to fit each rating
ADJ_ALPHA = 10.0          # ridge strength ~ "pseudo-games" of league-average prior
ADJ_MIN_ROWS = 100        # don't fit ratings on fewer team-games than this


def _week_index(df):
    """Chronological league-week number (0,1,2,...) for each (season, week)."""
    return df.groupby(["season", "week"]).ngroup()


def _fit_ridge(train, teams, alpha):
    """
    One ridge regression:  off_epa = intercept + off[team] + def[opp]
    Each row is one team's offense in one game, against a specific opponent.
    off[t] > 0: offense beats what the average opponent allows.
    def[t] > 0: defense ALLOWS more EPA than average (same sign as raw def_epa).
    """
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    rows = np.arange(len(train))
    X = np.zeros((len(train), 2 * n))
    X[rows, train["team"].map(idx).to_numpy()] = 1.0       # offense indicator
    X[rows, n + train["opp"].map(idx).to_numpy()] = 1.0    # opposing defense indicator
    fit = Ridge(alpha=alpha).fit(X, train["off_epa"])
    return fit.coef_[:n], fit.coef_[n:]


def add_opponent_adjusted(teamWeek, lookback=ADJ_LOOKBACK_WEEKS, alpha=ADJ_ALPHA):
    """
    For every (team, season, week) row, attach off_epa_adj / def_epa_adj =
    ratings fit ONLY on the `lookback` league-weeks strictly BEFORE that week.
    """
    tw = teamWeek.copy()
    tw["_t"] = _week_index(tw)
    teams = sorted(set(tw["team"]) | set(tw["opp"]))

    pieces = []
    for t in sorted(tw["_t"].unique()):
        train = tw[(tw["_t"] < t) & (tw["_t"] >= t - lookback)]
        if len(train) < ADJ_MIN_ROWS:
            continue                                  # not enough history -> NaN
        off, dfn = _fit_ridge(train, teams, alpha)
        now = tw.loc[tw["_t"] == t, ["team", "season", "week"]].copy()
        now["off_epa_adj"] = now["team"].map(dict(zip(teams, off)))
        now["def_epa_adj"] = now["team"].map(dict(zip(teams, dfn)))
        pieces.append(now)

    adj = pd.concat(pieces, ignore_index=True)
    return tw.merge(adj, on=["team", "season", "week"], how="left").drop(columns="_t")


def current_adjusted_ratings(teamWeek, lookback=ADJ_LOOKBACK_WEEKS, alpha=ADJ_ALPHA):
    """Ratings for the NEXT game: fit on the most recent `lookback` league-weeks."""
    tw = teamWeek.copy()
    tw["_t"] = _week_index(tw)
    train = tw[tw["_t"] > tw["_t"].max() - lookback]
    teams = sorted(set(train["team"]) | set(train["opp"]))
    off, dfn = _fit_ridge(train, teams, alpha)
    return pd.DataFrame({"team": teams, "off_epa_frozen": off, "def_epa_frozen": dfn})


def load_and_clean_games(rollWindow=ROLL_WINDOW, regOnly=REG_SEASON, wp=WIN_PROBABILITY_BOUND, adjust=ADJUST_OPPONENTS, adjLookback=ADJ_LOOKBACK_WEEKS, adjAlpha=ADJ_ALPHA):
    """
    Build the model-ready `games` table and the intermediate `teamWeek`
    table (the per-team-per-week rolling EPA stats, reused later by
    simulation.py to build "frozen" end-of-season ratings).

    Returns
    -------
    games : pd.DataFrame
        One row per historical game with home/away rolling EPA/pace
        features and the encoded result (1 = home win, 0 = away win).
    teamWeek : pd.DataFrame
        Per-team-per-week rolling offensive/defensive EPA and pace.
    """
    pbp = nfl.load_pbp(SEASONS).to_pandas()
    schedule = nfl.load_schedules(SCHEDULE_SEASONS).to_pandas()

    extraFilters = ""
    if regOnly:
        extraFilters += "\n         AND season_type = 'REG'"
    if wp is not None:
        lo, hi = wp
        extraFilters += f"\n        AND wp BETWEEN {float(lo)} AND {float(hi)}"

    nRaw = len(pbp)

    # choose columns, drop nulls, keep only run/pass plays
    pbp = sqldf(f"""
        SELECT game_id, season, week, posteam, defteam, epa, play_type, wp, season_type
        FROM pbp
        WHERE epa IS NOT NULL
          AND play_type IS NOT NULL
          AND posteam IS NOT NULL
          AND defteam IS NOT NULL
          AND play_type IN ('pass', 'run'){extraFilters}
    """, locals())
    print(f"plays kept: {len(pbp):,} of {nRaw:,} raw rows")

    # choose columns, drop rows with no result, and ties
    schedule = sqldf("""
        SELECT game_id, season, week, gameday, home_team, away_team, home_score, away_score, result, home_rest, away_rest, spread_line
        FROM schedule
        WHERE result IS NOT NULL AND result != 0
    """, locals())

    # per-team-per-week offensive and defensive aggregates
    offStats = sqldf("""
        SELECT week, season, posteam AS team, defteam AS opp,
               AVG(epa) AS off_epa,
               COUNT(epa) AS plays
        FROM pbp
        GROUP BY week, season, posteam, defteam
    """, locals())

    defStats = sqldf("""
        SELECT week, season, defteam AS team,
               AVG(epa) AS def_epa,
               COUNT(epa) AS def_plays
        FROM pbp
        GROUP BY week, season, defteam
    """, locals())

    win = f"""OVER (
        PARTITION BY team ORDER BY season, week
        ROWS BETWEEN {int(rollWindow)} PRECEDING AND 1 PRECEDING
    )"""

    # join off/def stats, then trailing N-week rolling averages
    # ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING mirrors shift(1).rolling(3, min_periods=1).mean():
    # it only looks at prior weeks and is happy to average fewer than 3 rows early in a team's history.
    teamWeek = sqldf(f"""
        WITH joined AS (
            SELECT o.team, o.opp, o.season, o.week, o.off_epa, o.plays, d.def_epa, d.def_plays
            FROM offStats o
            JOIN defStats d
              ON o.team = d.team AND o.season = d.season AND o.week = d.week
        )
        SELECT team, opp, season, week, off_epa, plays, def_epa, def_plays,
               AVG(off_epa) {win} AS off_epa_roll,
               AVG(def_epa) {win} AS def_epa_roll,
               AVG(plays) {win} AS plays_roll
        FROM joined
        ORDER BY team, season, week
    """, locals())

    if adjust:
        teamWeek = add_opponent_adjusted(teamWeek, adjLookback, adjAlpha)
        teamWeek["off_epa_roll"] = teamWeek["off_epa_adj"]
        teamWeek["def_epa_roll"] = teamWeek["def_epa_adj"]

    homeStats = teamWeek.rename(columns={"team": "home_team"})
    awayStats = teamWeek.rename(columns={"team": "away_team"})

    # attach home/away rolling stats to each game, encode the result
    games = sqldf("""
        SELECT s.*,
               h.off_epa_roll AS home_off_epa,
               h.def_epa_roll AS home_def_epa,
               h.plays_roll   AS home_plays,
               a.off_epa_roll AS away_off_epa,
               a.def_epa_roll AS away_def_epa,
               a.plays_roll   AS away_plays,
               CASE WHEN s.result > 0 THEN 1 ELSE 0 END AS result_encoded
        FROM schedule s
        LEFT JOIN homeStats h
            ON s.home_team = h.home_team AND s.season = h.season AND s.week = h.week
        LEFT JOIN awayStats a
            ON s.away_team = a.away_team AND s.season = a.season AND s.week = a.week
    """, locals())

    games["result"] = games["result_encoded"]
    games = games.drop(columns=["result_encoded"])

    assert games["result"].isin([0, 1]).all(), \
        "result has unexpected values — ties should have been dropped upstream"

    return games, teamWeek


if __name__ == "__main__":
    games, teamWeek = load_and_clean_games()
    print(games[["game_id", "home_team", "away_team", "home_off_epa",
                 "away_off_epa", "home_def_epa", "away_def_epa", "result"]])
