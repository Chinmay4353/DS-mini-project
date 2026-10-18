"""
Unit and integration tests for Student Placement Prediction pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data import (
    load_raw_data,
    get_clean_data,
    validate_input_dataframe,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.model import (
    load_pipeline,
    predict_single_student,
    compute_benchmark_comparison,
    DEFAULT_MODEL_PATH,
)


def test_raw_data_loading():
    df = load_raw_data()
    assert df.shape == (215, 15)
    assert "status" in df.columns
    assert "salary" in df.columns
    assert "sl_no" in df.columns


def test_clean_data_leakage_free():
    X, y = get_clean_data()
    assert X.shape == (215, 12)
    assert len(y) == 215
    assert "sl_no" not in X.columns
    assert "salary" not in X.columns
    assert "status" not in X.columns
    assert y.sum() == 148
    assert (len(y) - y.sum()) == 67


def test_saved_model_bundle_integrity():
    bundle = load_pipeline(DEFAULT_MODEL_PATH)
    assert "pipeline" in bundle
    assert "test_results" in bundle
    assert "cv_results" in bundle
    assert bundle["test_results"]["accuracy"] > 0.85
    assert bundle["test_results"]["precision"] > 0.90


def test_prediction_inference_valid():
    bundle = load_pipeline(DEFAULT_MODEL_PATH)
    valid_high_marks = {
        "ssc_p": 80.0,
        "hsc_p": 78.0,
        "degree_p": 76.0,
        "etest_p": 85.0,
        "mba_p": 70.0,
        "gender": "M",
        "ssc_b": "Central",
        "hsc_b": "Central",
        "hsc_s": "Commerce",
        "degree_t": "Comm&Mgmt",
        "workex": "Yes",
        "specialisation": "Mkt&Fin",
    }
    result = predict_single_student(bundle, valid_high_marks)
    assert result["prediction"] == 1
    assert result["label"] == "Placed"
    assert 0.0 <= result["probability_placed"] <= 1.0
    assert result["probability_placed"] > 0.80

    valid_low_marks = {
        "ssc_p": 45.0,
        "hsc_p": 42.0,
        "degree_p": 50.0,
        "etest_p": 52.0,
        "mba_p": 52.0,
        "gender": "F",
        "ssc_b": "Others",
        "hsc_b": "Others",
        "hsc_s": "Arts",
        "degree_t": "Others",
        "workex": "No",
        "specialisation": "Mkt&HR",
    }
    result_low = predict_single_student(bundle, valid_low_marks)
    assert result_low["prediction"] == 0
    assert result_low["label"] == "Not Placed"
    assert result_low["probability_placed"] < 0.20


def test_input_validation_catches_invalid_ranges():
    invalid_data = pd.DataFrame([{
        "ssc_p": 150.0,  # Invalid (> 100)
        "hsc_p": 78.0,
        "degree_p": 76.0,
        "etest_p": 85.0,
        "mba_p": 70.0,
        "gender": "M",
        "ssc_b": "Central",
        "hsc_b": "Central",
        "hsc_s": "Commerce",
        "degree_t": "Comm&Mgmt",
        "workex": "Yes",
        "specialisation": "Mkt&Fin",
    }])
    is_valid, errors = validate_input_dataframe(invalid_data)
    assert not is_valid
    assert any("outside allowable range" in e for e in errors)


def test_input_validation_catches_invalid_category():
    invalid_cat_data = pd.DataFrame([{
        "ssc_p": 70.0,
        "hsc_p": 78.0,
        "degree_p": 76.0,
        "etest_p": 85.0,
        "mba_p": 70.0,
        "gender": "Unknown",  # Invalid
        "ssc_b": "Central",
        "hsc_b": "Central",
        "hsc_s": "Commerce",
        "degree_t": "Comm&Mgmt",
        "workex": "Yes",
        "specialisation": "Mkt&Fin",
    }])
    is_valid, errors = validate_input_dataframe(invalid_cat_data)
    assert not is_valid
    assert any("Invalid option" in e for e in errors)


def test_benchmark_comparison_generates_dataframe():
    df_comp = compute_benchmark_comparison()
    assert len(df_comp) == 5
    assert "Logistic Regression (L2, C=1.0)" in df_comp["Algorithm"].values
    assert "Random Forest (n=100, default)" in df_comp["Algorithm"].values


def test_profile_validation_12th_hsc_compatible():
    from src.data import validate_student_profile
    profile = {
        "education_route": "12th/HSC route",
        "mba_status": "Completed",
        "ssc_p": 72.0,
        "hsc_p": 70.0,
        "hsc_b": "Central",
        "hsc_s": "Commerce",
        "degree_p": 68.0,
        "degree_t": "Comm&Mgmt",
        "etest_p": 75.0,
        "mba_p": 65.0,
        "gender": "M",
        "ssc_b": "Central",
        "workex": "Yes",
        "specialisation": "Mkt&Fin",
    }
    res = validate_student_profile(profile)
    assert res["is_valid"] is True
    assert res["is_model_compatible"] is True
    assert res["clean_model_input"] is not None
    assert len(res["clean_model_input"]) == 12

    # Verify model prediction runs successfully with compatible input
    bundle = load_pipeline(DEFAULT_MODEL_PATH)
    pred_res = predict_single_student(bundle, res["clean_model_input"])
    assert pred_res["prediction"] in [0, 1]
    assert 0.0 <= pred_res["probability_placed"] <= 1.0


def test_profile_validation_diploma_incompatible_safely():
    from src.data import validate_student_profile
    profile = {
        "education_route": "Diploma route",
        "mba_status": "Completed",
        "ssc_p": 72.0,
        "diploma_p": 78.5,  # Diploma entered; HSC fields absent
        "degree_p": 68.0,
        "degree_t": "Sci&Tech",
        "etest_p": 75.0,
        "mba_p": 65.0,
        "gender": "M",
        "ssc_b": "Central",
        "workex": "Yes",
        "specialisation": "Mkt&Fin",
    }
    res = validate_student_profile(profile)
    assert res["is_valid"] is True
    # Crucial: Model must NOT be marked compatible, clean_model_input must be None
    assert res["is_model_compatible"] is False
    assert res["clean_model_input"] is None
    # Must explain why model is disabled and confirm zero-imputation is prevented
    assert any("diploma" in a.lower() for a in res["advisories"])
    assert any("silently imputed" in a.lower() for a in res["advisories"])


def test_profile_validation_diploma_catches_invalid_percentage():
    from src.data import validate_student_profile
    profile = {
        "education_route": "Diploma route",
        "mba_status": "Completed",
        "ssc_p": 72.0,
        "diploma_p": 125.0,  # Invalid (> 100)
        "degree_p": 68.0,
        "degree_t": "Sci&Tech",
        "etest_p": 75.0,
        "mba_p": 65.0,
        "gender": "M",
        "ssc_b": "Central",
        "workex": "Yes",
        "specialisation": "Mkt&Fin",
    }
    res = validate_student_profile(profile)
    assert res["is_valid"] is False
    assert any("allowable range" in e for e in res["errors"])


def test_profile_validation_mba_in_progress_incompatible_safely():
    from src.data import validate_student_profile
    profile = {
        "education_route": "12th/HSC route",
        "mba_status": "Pursuing",  # MBA in progress; mba_p absent
        "ssc_p": 72.0,
        "hsc_p": 70.0,
        "hsc_b": "Central",
        "hsc_s": "Commerce",
        "degree_p": 68.0,
        "degree_t": "Comm&Mgmt",
        "etest_p": 75.0,
        "gender": "M",
        "ssc_b": "Central",
        "workex": "No",
    }
    res = validate_student_profile(profile)
    assert res["is_valid"] is True
    # Crucial: Model must NOT be marked compatible, clean_model_input must be None
    assert res["is_model_compatible"] is False
    assert res["clean_model_input"] is None
    assert any("mba" in a.lower() for a in res["advisories"])


def test_explain_single_prediction_fidelity():
    from src.model import explain_single_prediction
    bundle = load_pipeline(DEFAULT_MODEL_PATH)
    sample = {
        "ssc_p": 75.0,
        "hsc_p": 70.0,
        "degree_p": 72.0,
        "etest_p": 80.0,
        "mba_p": 65.0,
        "gender": "M",
        "ssc_b": "Central",
        "hsc_b": "Central",
        "hsc_s": "Commerce",
        "degree_t": "Comm&Mgmt",
        "workex": "Yes",
        "specialisation": "Mkt&Fin",
    }
    explanation = explain_single_prediction(bundle, sample)
    assert "z_score" in explanation
    assert "probability_placed" in explanation
    assert "contributions_df" in explanation
    assert len(explanation["contributions_df"]) == 14
    assert explanation["prediction"] == 1
    assert explanation["label"] == "Placed"
    assert explanation["probability_placed"] > 0.90
    assert len(explanation["top_positive"]) > 0
    import numpy as np
    expected_p = 1.0 / (1.0 + np.exp(-explanation["z_score"]))
    assert abs(explanation["probability_placed"] - expected_p) < 1e-6


if __name__ == "__main__":
    test_raw_data_loading()
    test_clean_data_leakage_free()
    test_saved_model_bundle_integrity()
    test_prediction_inference_valid()
    test_input_validation_catches_invalid_ranges()
    test_input_validation_catches_invalid_category()
    test_benchmark_comparison_generates_dataframe()
    test_profile_validation_12th_hsc_compatible()
    test_profile_validation_diploma_incompatible_safely()
    test_profile_validation_diploma_catches_invalid_percentage()
    test_profile_validation_mba_in_progress_incompatible_safely()
    test_explain_single_prediction_fidelity()
    print("ALL 12 PIPELINE, DATA, DIPLOMA & EXPLANATION TESTS PASSED SUCCESSFULLY!")
