"""
Data cleaning and feature engineering for the Play Call Predictor.

Loads play-by-play data, keeps run/pass plays, and adds each team's
trailing 3-week rolling pass rate as a feature (using only prior weeks,
so there's no leakage from the current week).
"""
import nflreadpy as nfl
from pandasql import sqldf

SEASONS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]


def load_and_clean_plays():
    """
    Returns
    -------
    pbp : pd.DataFrame
        One row per play with down/distance/field-position/game-state
        features, the encoded play type (0 = run, 1 = pass), and each
        team's trailing pass rate.
    """
    pbp = nfl.load_pbp(SEASONS).to_pandas()

    # choose columns, drop nulls, keep only run/pass plays, encode booleans as int,
    # and encode play_type directly here so it can never go stale on a re-run
    # 0 = run, 1 = pass
    pbp = sqldf("""
        SELECT game_id, season, week, posteam, defteam, down, ydstogo, yardline_100, score_differential, game_seconds_remaining, play_type, qtr,
               CAST(shotgun AS INTEGER)   AS shotgun,
               CAST(no_huddle AS INTEGER) AS no_huddle,
               CASE WHEN play_type = 'pass' THEN 1 ELSE 0 END AS play_type_encoded
        FROM pbp
        WHERE down IS NOT NULL
          AND yardline_100 IS NOT NULL
          AND score_differential IS NOT NULL
          AND game_seconds_remaining IS NOT NULL
          AND play_type IS NOT NULL
          AND play_type IN ('pass', 'run')
    """, locals())

    # per-team-per-week pass rate (how often this team passes, aggregated to one row per team-week)
    teamWeekPass = sqldf("""
        SELECT posteam AS team, season, week,
                  AVG(CASE WHEN play_type = 'pass' THEN 1.0 ELSE 0.0 END) AS pass_rate,
                  COUNT(*) AS plays
        FROM pbp
        GROUP BY posteam, season, week
    """, locals())

    # trailing 3-week rolling pass rate, using ONLY prior weeks (no current week / no leakage)
    teamWeekPass = sqldf("""
        SELECT team, season, week, pass_rate, plays,
                  AVG(pass_rate) OVER (
                        PARTITION BY team ORDER BY season, week
                        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                  ) AS pass_rate_roll
        FROM teamWeekPass
        GROUP BY team, season, week
    """, locals())

    pbp = sqldf("""
        SELECT p.*, t.pass_rate_roll
        FROM pbp p
        LEFT JOIN teamWeekPass t
                ON p.posteam = t.team AND p.season = t.season AND p.week = t.week
    """, locals())

    return pbp


if __name__ == "__main__":
    pbp = load_and_clean_plays()
    print(pbp[["down", "ydstogo", "yardline_100", "score_differential",
               "game_seconds_remaining", "play_type", "play_type_encoded",
               "qtr", "shotgun", "no_huddle", "pass_rate_roll"]])
