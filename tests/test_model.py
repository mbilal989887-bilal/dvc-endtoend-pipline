import os
import joblib
import numpy as np


MODEL_FILE = "models/student_marks_prediction.joblib"


def test_model_file_exists():
    """Check that the trained model exists."""
    assert os.path.exists(MODEL_FILE)


def test_model_can_predict():
    """Check that the model can make a prediction."""

    model = joblib.load(MODEL_FILE)

    sample_data = np.array([[5, 80, 70]])

    prediction = model.predict(sample_data)

    assert len(prediction) == 1
    assert isinstance(prediction[0], (float, np.floating))