"""
Streamlit AppTest suite verifying all five pages, form submissions, presets,
and diploma safeguards across the dashboard.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest


APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_overview_page():
    """Verify Page 1: Overview & KPIs renders without exceptions and contains metrics."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    assert not at.exception, f"Overview page raised exceptions: {at.exception}"
    # Verify metric cards exist
    assert len(at.metric) >= 5
    metric_labels = [m.label for m in at.metric]
    assert "Total Candidates" in metric_labels
    assert "Placed Students" in metric_labels


def test_app_eda_page():
    """Verify Page 2: Exploratory Data Insights renders tabs and plots without exceptions."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    # Navigate to Page 2
    at.radio[0].set_value("🔍 Exploratory Data Insights")
    at.run()
    assert not at.exception, f"EDA page raised exceptions: {at.exception}"
    assert len(at.tabs) >= 3


def test_app_prediction_placed_preset():
    """Verify Page 3: Placed Student preset loads, submits, and renders placed result card."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    # Navigate to Page 3
    at.radio[0].set_value("🎯 Student Prediction")
    at.run()
    assert not at.exception

    # Click 'Load Placed Profile' preset button (first button)
    at.button[0].click()
    at.run()
    assert not at.exception

    # Submit form
    at.button[4].click()  # Form submit button
    at.run()
    assert not at.exception

    # Check that success result markdown is rendered
    markdown_texts = " ".join([m.value for m in at.markdown])
    assert "PREDICTION: PLACED" in markdown_texts or "Likely to be Placed" in markdown_texts
    assert "Explain This Prediction" in markdown_texts or len(at.expander) > 0


def test_app_prediction_at_risk_preset():
    """Verify Page 3: At-Risk Student preset loads, submits, and renders at-risk result card."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    # Navigate to Page 3
    at.radio[0].set_value("🎯 Student Prediction")
    at.run()

    # Click 'Load At-Risk Profile' preset button (second button)
    at.button[1].click()
    at.run()
    assert not at.exception

    # Submit form
    at.button[4].click()  # Form submit button
    at.run()
    assert not at.exception

    markdown_texts = " ".join([m.value for m in at.markdown])
    assert "AT-RISK / NOT PLACED" in markdown_texts or "Not Placed" in markdown_texts


def test_app_prediction_diploma_safeguard():
    """Verify Page 3: Diploma route properly withholds unvalidated automated prediction."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    # Navigate to Page 3
    at.radio[0].set_value("🎯 Student Prediction")
    at.run()

    # Click 'Load Diploma Profile' button (third button)
    at.button[2].click()
    at.run()
    assert not at.exception

    # Submit form
    at.button[4].click()
    at.run()
    assert not at.exception

    markdown_texts = " ".join([m.value for m in at.markdown])
    assert "Automated Prediction Inapplicable" in markdown_texts or "withheld" in markdown_texts.lower()


def test_app_evaluation_page():
    """Verify Page 4: Model Evaluation renders metrics, benchmark table, and confusion matrix."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    # Navigate to Page 4
    at.radio[0].set_value("📈 Model Evaluation")
    at.run()
    assert not at.exception, f"Evaluation page raised exceptions: {at.exception}"
    assert len(at.metric) >= 5
    metric_labels = [m.label for m in at.metric]
    assert "Test Accuracy" in metric_labels
    assert "Test Precision" in metric_labels
    markdown_texts = " ".join([m.value for m in at.markdown])
    assert "Decision Threshold Trade-offs" in markdown_texts


def test_app_about_page():
    """Verify Page 5: About & Methodology renders tabs and mathematical formulation."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    # Navigate to Page 5
    at.radio[0].set_value("📖 About & Methodology")
    at.run()
    assert not at.exception, f"About page raised exceptions: {at.exception}"
    assert len(at.tabs) >= 3


if __name__ == "__main__":
    print("Running Streamlit AppTest suite across all 5 pages...")
    test_app_overview_page()
    print("  [1/6] Overview page test passed.")
    test_app_eda_page()
    print("  [2/6] EDA page test passed.")
    test_app_prediction_placed_preset()
    print("  [3/6] Prediction Placed preset test passed.")
    test_app_prediction_at_risk_preset()
    print("  [4/6] Prediction At-Risk preset test passed.")
    test_app_prediction_diploma_safeguard()
    print("  [5/6] Prediction Diploma safeguard test passed.")
    test_app_evaluation_page()
    print("  [6/7] Evaluation page test passed.")
    test_app_about_page()
    print("  [7/7] About page test passed.")
    print("ALL 7 STREAMLIT APPTEST CHECKS PASSED WITH 0 EXCEPTIONS!")
