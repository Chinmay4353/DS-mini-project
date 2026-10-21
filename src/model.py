"""
Model training, serialization, evaluation, and inference pipelines for Student Placement Prediction.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    make_scorer,
)

from src.data import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    PROJECT_ROOT,
    get_clean_data,
    validate_input_dataframe,
)

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "placement_model.joblib"
RANDOM_STATE = 42


def build_preprocessor():
    """Build leakage-safe ColumnTransformer for scaling and one-hot encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def build_pipeline(classifier=None):
    """Construct full scikit-learn pipeline with preprocessor and classifier."""
    if classifier is None:
        classifier = LogisticRegression(
            C=1.0,
            max_iter=500,
            random_state=RANDOM_STATE,
            solver="lbfgs"
        )
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier)
    ])


def train_and_save_pipeline(model_path=DEFAULT_MODEL_PATH, random_state=RANDOM_STATE):
    """
    Train Logistic Regression pipeline using Phase 1 protocol,
    compute verified CV and holdout test metrics, and serialize bundle with Joblib.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    X, y = get_clean_data()

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )

    scorers = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }

    # 5-fold CV strictly on 80% training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    pipeline = build_pipeline()

    cv_scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scorers,
        return_train_score=False,
    )

    cv_results = {
        "cv_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
        "cv_accuracy_std": float(cv_scores["test_accuracy"].std()),
        "cv_f1_mean": float(cv_scores["test_f1"].mean()),
        "cv_f1_std": float(cv_scores["test_f1"].std()),
        "cv_precision_mean": float(cv_scores["test_precision"].mean()),
        "cv_recall_mean": float(cv_scores["test_recall"].mean()),
    }

    # Fit pipeline on training data
    pipeline.fit(X_train, y_train)

    # Evaluate on untouched holdout test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    test_results = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp) if (tn + fp) > 0 else 0.0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
    }

    # Extract coefficients and odds ratios
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    coefs = pipeline.named_steps["classifier"].coef_[0]
    intercept = float(pipeline.named_steps["classifier"].intercept_[0])

    coef_data = []
    for f, c in zip(feature_names, coefs):
        coef_data.append({
            "feature": f,
            "coefficient": float(c),
            "odds_ratio": float(np.exp(c)),
        })

    bundle = {
        "pipeline": pipeline,
        "model_name": "Logistic Regression (L2, C=1.0)",
        "cv_results": cv_results,
        "test_results": test_results,
        "confusion_matrix": cm.tolist(),
        "coefficients": coef_data,
        "intercept": intercept,
        "random_state": random_state,
        "features": NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
    }

    joblib.dump(bundle, model_path)
    return bundle


def load_pipeline(model_path=DEFAULT_MODEL_PATH):
    """
    Load serialized model bundle. If missing, automatically trains and saves bundle.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        return train_and_save_pipeline(model_path)
    bundle = joblib.load(model_path)
    return bundle


def predict_single_student(pipeline_bundle, input_data):
    """
    Make placement prediction for a single student input.
    input_data can be a dict or a 1-row DataFrame.
    Returns structured prediction response with estimated placement probability.
    """
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    else:
        df_input = input_data.copy()

    # Reorder columns to guarantee exact match with training pipeline
    expected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    df_input = df_input[expected_cols]

    is_valid, errors = validate_input_dataframe(df_input)
    if not is_valid:
        raise ValueError(f"Input validation failed: {'; '.join(errors)}")

    pipeline = pipeline_bundle["pipeline"]
    pred = int(pipeline.predict(df_input)[0])
    proba_placed = float(pipeline.predict_proba(df_input)[0][1])

    label = "Placed" if pred == 1 else "Not Placed"

    return {
        "prediction": pred,
        "label": label,
        "probability_placed": proba_placed,
        "probability_not_placed": 1.0 - proba_placed,
    }


def explain_single_prediction(pipeline_bundle, input_data):
    """
    Deconstruct the linear inference process for educational and viva explanation.
    Calculates:
      1. Standardized numerical values and one-hot encoded flags
      2. Model weights and intercept
      3. Individual feature contributions (w_i * x_i) to log-odds score z
      4. Sigmoid transformation P = 1 / (1 + exp(-z))
      5. Classification threshold decision
    """
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    else:
        df_input = input_data.copy()

    expected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    df_input = df_input[expected_cols]

    is_valid, errors = validate_input_dataframe(df_input)
    if not is_valid:
        raise ValueError(f"Input validation failed: {'; '.join(errors)}")

    pipeline = pipeline_bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    x_transformed = preprocessor.transform(df_input)[0]
    feature_names = preprocessor.get_feature_names_out()
    weights = classifier.coef_[0]
    intercept = float(classifier.intercept_[0])

    contributions = weights * x_transformed
    z_score = float(intercept + np.sum(contributions))
    prob_placed = float(1.0 / (1.0 + np.exp(-z_score)))
    threshold = 0.50
    pred = 1 if prob_placed >= threshold else 0
    label = "Placed" if pred == 1 else "Not Placed"

    feature_rows = []
    for name, w, x_val, c in zip(feature_names, weights, x_transformed, contributions):
        direction = "Increases Odds" if c > 0 else ("Decreases Odds" if c < 0 else "Neutral")
        feature_rows.append({
            "feature": name,
            "weight": float(w),
            "transformed_value": float(x_val),
            "contribution": float(c),
            "direction": direction,
        })

    df_contrib = pd.DataFrame(feature_rows)

    return {
        "intercept": intercept,
        "z_score": z_score,
        "probability_placed": prob_placed,
        "probability_not_placed": 1.0 - prob_placed,
        "threshold": threshold,
        "prediction": pred,
        "label": label,
        "contributions_df": df_contrib,
        "top_positive": df_contrib[df_contrib["contribution"] > 0].sort_values("contribution", ascending=False).to_dict("records"),
        "top_negative": df_contrib[df_contrib["contribution"] < 0].sort_values("contribution", ascending=True).to_dict("records"),
    }


def compute_benchmark_comparison(random_state=RANDOM_STATE):
    """
    Run candidate algorithm comparison to dynamically populate the dashboard evaluation tab.
    Ensures that values displayed on the dashboard are computed empirically.
    """
    X, y = get_clean_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )

    models = {
        "Logistic Regression (L2, C=1.0)": LogisticRegression(
            C=1.0, max_iter=500, random_state=random_state, solver="lbfgs"
        ),
        "Decision Tree (Pruned, depth=3)": DecisionTreeClassifier(
            max_depth=3, random_state=random_state
        ),
        "Decision Tree (Unconstrained)": DecisionTreeClassifier(
            random_state=random_state
        ),
        "Random Forest (n=100, depth=4)": RandomForestClassifier(
            n_estimators=100, max_depth=4, random_state=random_state
        ),
        "Random Forest (n=100, default)": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
    }

    scorers = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    rows = []
    for name, clf in models.items():
        pipe = build_pipeline(clf)
        cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scorers)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        rows.append({
            "Algorithm": name,
            "CV Accuracy (Train 80%)": f"{cv_scores['test_accuracy'].mean()*100:.2f}% +/- {cv_scores['test_accuracy'].std()*100:.2f}%",
            "CV F1-Score (Train 80%)": f"{cv_scores['test_f1'].mean()*100:.2f}% +/- {cv_scores['test_f1'].std()*100:.2f}%",
            "Test Accuracy": f"{accuracy_score(y_test, y_pred)*100:.2f}%",
            "Test F1-Score": f"{f1_score(y_test, y_pred, zero_division=0)*100:.2f}%",
            "Test Precision": f"{precision_score(y_test, y_pred, zero_division=0)*100:.2f}%",
            "Test Recall": f"{recall_score(y_test, y_pred, zero_division=0)*100:.2f}%",
            "Confusion Matrix [TN, FP; FN, TP]": f"[{tn}, {fp}; {fn}, {tp}]",
        })

    return pd.DataFrame(rows)


def compute_threshold_comparison(random_state=RANDOM_STATE):
    """
    Compare classification thresholds using out-of-fold training probabilities only.

    The holdout set is deliberately excluded so it remains an unbiased final
    evaluation set rather than being used to choose a decision threshold.
    """
    X, y = get_clean_data()
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    probabilities = cross_val_predict(
        build_pipeline(),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    rows = []
    for threshold in (0.30, 0.40, 0.50, 0.60, 0.70):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, predictions, labels=[0, 1]).ravel()
        rows.append({
            "Threshold": threshold,
            "Accuracy (%)": accuracy_score(y_train, predictions) * 100,
            "Precision (%)": precision_score(y_train, predictions, zero_division=0) * 100,
            "Recall (%)": recall_score(y_train, predictions, zero_division=0) * 100,
            "Specificity (%)": tn / (tn + fp) * 100 if tn + fp else 0.0,
            "F1-Score (%)": f1_score(y_train, predictions, zero_division=0) * 100,
            "Predicted Placed": int(predictions.sum()),
            "Training Samples": int(len(y_train)),
        })

    return pd.DataFrame(rows)
