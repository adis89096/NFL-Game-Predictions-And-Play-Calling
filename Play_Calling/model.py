"""
XGBoost classifier that predicts whether a play will be a run or a pass
from down/distance/field-position/game-state and each team's tendencies.
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier

FEATURES = ["down", "ydstogo", "yardline_100", "score_differential",
            "game_seconds_remaining", "qtr", "shotgun", "no_huddle", "pass_rate_roll"]


def train_play_predictor(pbp):
    """
    Fit an XGBoost classifier on historical plays and report holdout
    accuracy against a naive "always pick pass" baseline.

    Returns
    -------
    model : fitted XGBClassifier
    """
    assert pbp["play_type_encoded"].isin([0, 1]).all(), \
        "play_type_encoded has unexpected values — rerun data_cleaning first"

    pbp_clean = pbp.dropna(subset=FEATURES + ["play_type_encoded"])
    print(len(pbp), "->", len(pbp_clean))

    X = pbp_clean[FEATURES]
    y = pbp_clean["play_type_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    baseline = (y_test == 1).mean()
    print(f"\nNaive 'always pick pass' baseline accuracy: {baseline:.3f}")

    return model


if __name__ == "__main__":
    from data_cleaning import load_and_clean_plays

    pbp = load_and_clean_plays()
    train_play_predictor(pbp)
