"""
Data cleaning and feature engineering for the Weekly Game Predictor.

Loads play-by-play and schedule data from nflreadpy, builds per-team
rolling offensive/defensive EPA features (a trailing 3-week average using
only prior weeks, so there's no leakage), and joins those onto each game
to produce one row per game with home/away features and the actual result.
"""
import nflreadpy as nfl
from pandasql import sqldf

SEASONS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
SCHEDULE_SEASONS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]


def load_and_clean_games():
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

    # choose columns, drop nulls, keep only run/pass plays
    pbp = sqldf("""
        SELECT game_id, season, week, posteam, defteam, epa, play_type
        FROM pbp
        WHERE epa IS NOT NULL
          AND play_type IS NOT NULL
          AND posteam IS NOT NULL
          AND defteam IS NOT NULL
          AND play_type IN ('pass', 'run')
    """, locals())

    # choose columns, drop rows with no result, and ties
    schedule = sqldf("""
        SELECT game_id, season, week, gameday, home_team, away_team, home_score, away_score, result
        FROM schedule
        WHERE result IS NOT NULL AND result != 0
    """, locals())

    # per-team-per-week offensive and defensive aggregates
    offStats = sqldf("""
        SELECT week, season, posteam AS team,
               AVG(epa) AS off_epa,
               COUNT(epa) AS plays
        FROM pbp
        GROUP BY week, season, posteam
    """, locals())

    defStats = sqldf("""
        SELECT week, season, defteam AS team,
               AVG(epa) AS def_epa,
               COUNT(epa) AS def_plays
        FROM pbp
        GROUP BY week, season, defteam
    """, locals())

    # join off/def stats, then trailing 3-week rolling averages
    # ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING mirrors shift(1).rolling(3, min_periods=1).mean():
    # it only looks at prior weeks and is happy to average fewer than 3 rows early in a team's history.
    teamWeek = sqldf("""
        WITH joined AS (
            SELECT o.team, o.season, o.week, o.off_epa, o.plays, d.def_epa, d.def_plays
            FROM offStats o
            JOIN defStats d
              ON o.team = d.team AND o.season = d.season AND o.week = d.week
        )
        SELECT team, season, week, off_epa, plays, def_epa, def_plays,
               AVG(off_epa) OVER (
                   PARTITION BY team ORDER BY season, week
                   ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
               ) AS off_epa_roll,
               AVG(def_epa) OVER (
                   PARTITION BY team ORDER BY season, week
                   ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
               ) AS def_epa_roll,
               AVG(plays) OVER (
                   PARTITION BY team ORDER BY season, week
                   ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
               ) AS plays_roll
        FROM joined
        ORDER BY team, season, week
    """, locals())

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
