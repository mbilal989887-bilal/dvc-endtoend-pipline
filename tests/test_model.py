import pandas as pd
from sklearn.linear_model import LinearRegression


def test_model_can_train():
    """Check that the model can train successfully."""

    data = pd.read_csv("data/processed/train.csv")

    X = data[
        [
            "hours_studied",
            "attendance_percent",
            "previous_scores",
        ]
    ]

    y = data["exam_score"]

    model = LinearRegression()
    model.fit(X, y)

    assert model is not None


def test_model_can_predict():
    """Check that the trained model can make a prediction."""

    data = pd.read_csv("data/processed/train.csv")

    X = data[
        [
            "hours_studied",
            "attendance_percent",
            "previous_scores",
        ]
    ]

    y = data["exam_score"]

    model = LinearRegression()
    model.fit(X, y)

    sample = pd.DataFrame(
        {
            "hours_studied": [5],
            "attendance_percent": [80],
            "previous_scores": [70],
        }
    )

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] >= 0