from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "student_marks_prediction.joblib"

app = FastAPI(
    title="Student Marks Prediction API",
    description="Predict student exam marks using study and performance data.",
    version="1.0.0",
)

model = None


def load_model():
    global model

    if model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model file not found at {MODEL_PATH}. Train the model first."
            )
        model = joblib.load(MODEL_PATH)

    return model


class StudentData(BaseModel):
    hours_studied: float = Field(ge=0, description="Hours studied")
    attendance_percent: float = Field(ge=0, le=100)
    previous_scores: float = Field(ge=0, le=100)


@app.get("/")
def home():
    return {"message": "Student Marks Prediction API is running"}


@app.post("/predict")
def predict(data: StudentData):
    try:
        model_instance = load_model()
        input_data = pd.DataFrame(
            [
                {
                    "hours_studied": data.hours_studied,
                    "attendance_percent": data.attendance_percent,
                    "previous_scores": data.previous_scores,
                }
            ]
        )

        prediction = model_instance.predict(input_data)[0]

        return {"predicted_exam_score": round(float(prediction), 2)}
    except RuntimeError:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please check the server logs.",
        ) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

