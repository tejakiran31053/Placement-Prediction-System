import os
import sys

# Make sure project root is importable when running app.py directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request

from src.data.load_data import load_data
from src.models.pipeline import (
    train_all_models,
    preprocess_single_record,
    ONE_HOT_FEATURES,
    ORDINAL_FEATURES,
)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dataset")
def dataset():
    df = load_data()

    return render_template(
        "dataset.html",
        rows=df.shape[0],
        cols=df.shape[1],
        columns=df.columns.tolist(),
        data=df.head().values.tolist()
    )


@app.route("/eda")
def eda():
    return render_template("eda.html")


@app.route("/preprocessing")
def preprocessing():
    cache = train_all_models()
    data = cache["data"]

    return render_template(
        "preprocessing.html",
        numerical_features=data["numerical_features"],
        categorical_features=data["categorical_features"],
        one_hot_features=ONE_HOT_FEATURES,
        ordinal_features=ORDINAL_FEATURES,
        train_shape=data["X_train"].shape,
        test_shape=data["X_test"].shape,
        final_feature_count=len(cache["feature_columns"]),
    )


@app.route("/models")
def models():
    cache = train_all_models()

    model_info = {
        "Logistic Regression":
            "A linear model that estimates the probability of placement "
            "using a weighted sum of the input features passed through "
            "a sigmoid function.",
        "Decision Tree":
            "A tree-based model (entropy criterion, max depth 5) that "
            "splits the data on the most informative feature at each node.",
        "Random Forest":
            "An ensemble of 100 decision trees trained on bootstrapped "
            "samples; predictions are combined by majority vote.",
        "Gradient Boosting":
            "An ensemble of 100 shallow decision trees (max depth 3) built "
            "sequentially, where each new tree corrects the errors of the "
            "previous ones.",
    }

    return render_template(
        "models.html",
        results=cache["results"],
        model_info=model_info,
        best_model_name=cache["best_model_name"],
    )


@app.route("/evaluation")
def evaluation():
    cache = train_all_models()

    return render_template(
        "evaluation.html",
        results=cache["results"],
        best_model_name=cache["best_model_name"],
    )


@app.route("/comparison")
def comparison():
    cache = train_all_models()

    sorted_results = dict(
        sorted(
            cache["results"].items(),
            key=lambda item: item[1]["accuracy"],
            reverse=True
        )
    )

    return render_template(
        "comparison.html",
        results=sorted_results,
        best_model_name=cache["best_model_name"],
    )


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    df = load_data()
    cache = train_all_models()

    dropdown_options = {
        col: sorted(df[col].dropna().unique().tolist())
        for col in ONE_HOT_FEATURES + ORDINAL_FEATURES
    }

    prediction_result = None
    probability = None
    selected_model_name = "Gradient Boosting"
    form_values = {}
    error = None

    if request.method == "POST":
        selected_model_name = request.form.get("model_choice", "Gradient Boosting")
        form_values = request.form.to_dict()

        try:
            record = {
                "Gender": request.form.get("Gender"),
                "City": request.form.get("City"),
                "CollegeTier": request.form.get("CollegeTier"),
                "Stream": request.form.get("Stream"),
                "Specialisation": request.form.get("Specialisation"),
                "Hostel": request.form.get("Hostel"),
                "HistoryOfBacklogs": request.form.get("HistoryOfBacklogs"),
                "CGPA_Tier": request.form.get("CGPA_Tier"),
                "SGPA_Sem1": float(request.form.get("SGPA_Sem1")),
                "SGPA_Sem2": float(request.form.get("SGPA_Sem2")),
                "SGPA_Sem3": float(request.form.get("SGPA_Sem3")),
                "SGPA_Sem4": float(request.form.get("SGPA_Sem4")),
                "SGPA_Sem5": float(request.form.get("SGPA_Sem5")),
                "SGPA_Sem6": float(request.form.get("SGPA_Sem6")),
                "SGPA_Sem7": float(request.form.get("SGPA_Sem7")),
                "SGPA_Sem8": float(request.form.get("SGPA_Sem8")),
                "CGPA": float(request.form.get("CGPA")),
                "AttendancePercent": float(request.form.get("AttendancePercent")),
                "Internships": int(request.form.get("Internships")),
                "Projects": int(request.form.get("Projects")),
                "Workshops": float(request.form.get("Workshops")),
                "Certifications": int(request.form.get("Certifications")),
                "Publications": int(request.form.get("Publications")),
                "AptitudeTestScore": float(request.form.get("AptitudeTestScore")),
                "SoftSkillsRating": float(request.form.get("SoftSkillsRating")),
                "CodingTestScore": float(request.form.get("CodingTestScore")),
                "MockInterviewScore": float(request.form.get("MockInterviewScore")),
                "ExtraCurricular": int(request.form.get("ExtraCurricular")),
            }

            processed_row = preprocess_single_record(record)
            model = cache["models"][selected_model_name]

            pred = model.predict(processed_row)[0]
            prediction_result = "Placed" if int(pred) == 1 else "Not Placed"

            if hasattr(model, "predict_proba"):
                probability = round(
                    float(model.predict_proba(processed_row)[0][1]) * 100, 2
                )

        except (TypeError, ValueError) as exc:
            error = f"Please fill in every field with a valid value. ({exc})"

    return render_template(
        "prediction.html",
        dropdown_options=dropdown_options,
        model_names=list(cache["models"].keys()),
        selected_model_name=selected_model_name,
        prediction_result=prediction_result,
        probability=probability,
        form_values=form_values,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)