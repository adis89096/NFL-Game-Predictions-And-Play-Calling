"""
Logistic regression model that predicts the winner of an NFL game from
each team's rolling offensive/defensive EPA and pace (plays per game).
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

FEATURES = ["home_off_epa", "away_off_epa", "home_def_epa", "away_def_epa", "home_plays", "away_plays"]


def train_game_predictor(games):
    """
    Fit a logistic regression classifier on historical games and report
    holdout accuracy against a naive "always pick home team" baseline.

    Returns
    -------
    model : fitted LogisticRegression
    scaler : fitted StandardScaler used on the features
    """
    games_clean = games.dropna(subset=FEATURES + ["result"])
    print(len(games), "->", len(games_clean))

    X = games_clean[FEATURES]
    y = games_clean["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.3, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    baseline = (y_test == 1).mean()
    print(f"\nNaive 'always pick home team' baseline accuracy: {baseline:.3f}")

    return model, scaler


if __name__ == "__main__":
    from data_cleaning import load_and_clean_games

    games, _ = load_and_clean_games()
    train_game_predictor(games)
