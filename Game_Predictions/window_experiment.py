"""Step 5: raw rolling EPA vs opponent-adjusted (ridge) ratings. Train <=2023, test 2024-25."""
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, brier_score_loss

from data_cleaning import load_and_clean_games
from model import FEATURES

configs = [                                  # (label, kwargs)
    ("raw rolling",      dict(adjust=False)),
    ("ridge L=17 a=10",  dict(adjust=True, adjLookback=17, adjAlpha=10.0)),
    ("ridge L=17 a=30",  dict(adjust=True, adjLookback=17, adjAlpha=30.0)),
    ("ridge L=10 a=10",  dict(adjust=True, adjLookback=10, adjAlpha=10.0)),
]

for label, kw in configs:
    games, _ = load_and_clean_games(**kw)
    g = games.dropna(subset=FEATURES + ["home_score", "away_score"]).copy()
    g["margin"] = g["home_score"] - g["away_score"]
    train = g[g["season"] <= 2023]
    test = g[(g["season"] >= 2024) & (g["season"] <= 2025) & (g["week"] <= 18)]

    m = LinearRegression().fit(train[FEATURES], train["margin"])
    sd = (train["margin"] - m.predict(train[FEATURES])).std()
    pred = m.predict(test[FEATURES])
    p_home = norm.cdf(pred / sd)
    y = test["result"]
    print(f"{label:<16} | acc={accuracy_score(y, p_home > 0.5):.3f} "
          f"| brier={brier_score_loss(y, p_home):.4f} "
          f"| margin RMSE={np.sqrt(((test['margin'] - pred) ** 2).mean()):.2f} | n={len(test)}")