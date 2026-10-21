"""
Logistic regression model that predicts the winner of an NFL game from
each team's rolling offensive/defensive EPA and pace (plays per game).
"""
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, brier_score_loss, log_loss


BASE_FEATURE = ["home_off_epa", "away_off_epa", "home_def_epa", "away_def_epa", "home_plays", "away_plays"]
CONTEXT_FEATURES = ["home_rest", "away_rest"]
FEATURES = BASE_FEATURE + CONTEXT_FEATURES


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

    tscv = TimeSeriesSplit(n_splits=5)
    *_, (train_idx, test_idx) = tscv.split(X)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nBrier Score: ", brier_score_loss(y_test, y_prob))
    print("\nLog Loss: ", log_loss(y_test, y_prob))

    baseline = (y_test == 1).mean()
    print(f"\nNaive 'always pick home team' baseline accuracy: {baseline:.3f}")

    return model, scaler


if __name__ == "__main__":
    from data_cleaning import load_and_clean_games

    games, _ = load_and_clean_games()
    train_game_predictor(games)



"""
Original Report:

Confusion Matrix: 
 [[104 115]
 [ 65 198]]

Classification Report:
               precision    recall  f1-score   support

           0       0.62      0.47      0.54       219
           1       0.63      0.75      0.69       263

    accuracy                           0.63       482
   macro avg       0.62      0.61      0.61       482
weighted avg       0.62      0.63      0.62       482


Brier Score:  0.22206882542972298

Log Loss:  0.6339517864609686

Naive 'always pick home team' baseline accuracy: 0.546
"""
