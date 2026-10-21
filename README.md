# Student Placement Prediction

A Streamlit dashboard and command-line demo for exploring a historical student placement dataset. The project demonstrates a scikit-learn preprocessing/model pipeline, model evaluation, and input validation.

## Requirements

- Python 3.11 or 3.13
- pip

## Run the dashboard

From the repository root, create and activate a virtual environment, install the dependencies, and launch Streamlit:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If Streamlit does not open a browser automatically, use the local URL printed in the terminal.

## Other commands

```powershell
# Run the shared-pipeline interactive CLI
python student.py

# Run the model-comparison experiment
python compare_models.py

# Install test dependencies and run all tests
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The model artifact is stored at `models/placement_model.joblib`. If it is absent, the dashboard trains and saves a replacement using `Placement.csv`.

## Project layout

- `app.py` — Streamlit dashboard
- `src/data.py` — dataset loading, feature schema, and input validation
- `src/model.py` — training, inference, evaluation, and threshold comparison
- `student.py` — interactive command-line inference through the shared model
- `compare_models.py` — separate candidate-algorithm experiment
- `tests/` — pipeline and Streamlit app tests
- `Placement.csv` — historical placement data used by the project

## Dataset and responsible use

The repository does not currently identify the dataset's source or license. Confirm that its original license permits redistribution and use before publishing or reusing the data. Do not add private or personally identifying student data.

This dataset contains 215 historical records. Predictions and probabilities are educational estimates, may reflect historical hiring patterns, and are not guarantees. Do not use this project to deny students access to placement opportunities.

## Contributing

Bug fixes, documentation improvements, and tests are welcome. Please keep changes focused, add regression tests for behavior changes, and avoid reporting holdout test metrics as evidence for threshold/model selection. Run `python -m pytest -q` before opening a pull request.
