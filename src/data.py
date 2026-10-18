"""
Data handling, validation, and schema definitions for Student Placement Prediction.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Feature Schema Definitions
NUMERICAL_FEATURES = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
CATEGORICAL_FEATURES = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"]
TARGET_COLUMN = "status"
LEAKAGE_COLUMNS = ["sl_no", "salary"]

# UI and Validation Metadata
FEATURE_METADATA = {
    "ssc_p": {
        "label": "10th Grade Percentage (SSC %)",
        "min": 0.0,
        "max": 100.0,
        "default": 67.0,
        "step": 0.1,
        "help": "Secondary school percentage (Standard 10). Dataset range: 40.89% - 89.40%.",
    },
    "hsc_p": {
        "label": "12th Grade Percentage (HSC %)",
        "min": 0.0,
        "max": 100.0,
        "default": 66.0,
        "step": 0.1,
        "help": "Higher secondary school percentage (Standard 12). Dataset range: 37.00% - 97.70%.",
    },
    "degree_p": {
        "label": "Undergraduate Degree Percentage",
        "min": 0.0,
        "max": 100.0,
        "default": 66.0,
        "step": 0.1,
        "help": "Degree marks percentage. Dataset range: 50.00% - 91.00%.",
    },
    "etest_p": {
        "label": "Employability / Aptitude Test Score (%)",
        "min": 0.0,
        "max": 100.0,
        "default": 71.0,
        "step": 0.1,
        "help": "Entrance employability test percentage. Dataset range: 50.00% - 98.00%.",
    },
    "mba_p": {
        "label": "MBA Percentage",
        "min": 0.0,
        "max": 100.0,
        "default": 62.0,
        "step": 0.1,
        "help": "Post-graduate MBA percentage. Dataset range: 51.21% - 77.89%.",
    },
    "gender": {
        "label": "Gender",
        "options": ["M", "F"],
        "display_names": {"M": "Male (M)", "F": "Female (F)"},
        "default": "M",
        "help": "Candidate gender.",
    },
    "ssc_b": {
        "label": "SSC Board of Education",
        "options": ["Central", "Others"],
        "display_names": {"Central": "Central Board", "Others": "State / Other Boards"},
        "default": "Central",
        "help": "Secondary school examination board.",
    },
    "hsc_b": {
        "label": "HSC Board of Education",
        "options": ["Central", "Others"],
        "display_names": {"Central": "Central Board", "Others": "State / Other Boards"},
        "default": "Others",
        "help": "Higher secondary examination board.",
    },
    "hsc_s": {
        "label": "HSC Specialization Stream",
        "options": ["Commerce", "Science", "Arts"],
        "display_names": {"Commerce": "Commerce", "Science": "Science", "Arts": "Arts"},
        "default": "Commerce",
        "help": "Academic specialization in 12th standard.",
    },
    "degree_t": {
        "label": "Undergraduate Degree Type",
        "options": ["Comm&Mgmt", "Sci&Tech", "Others"],
        "display_names": {
            "Comm&Mgmt": "Commerce & Management",
            "Sci&Tech": "Science & Technology",
            "Others": "Others (Arts / Law / etc.)",
        },
        "default": "Comm&Mgmt",
        "help": "Field of undergraduate education.",
    },
    "workex": {
        "label": "Prior Work Experience",
        "options": ["No", "Yes"],
        "display_names": {"No": "No Prior Experience", "Yes": "Has Prior Work Experience"},
        "default": "No",
        "help": "Whether candidate has relevant pre-MBA industry work experience.",
    },
    "specialisation": {
        "label": "MBA Specialization Track",
        "options": ["Mkt&Fin", "Mkt&HR"],
        "display_names": {
            "Mkt&Fin": "Marketing & Finance (Mkt&Fin)",
            "Mkt&HR": "Marketing & Human Resources (Mkt&HR)",
        },
        "default": "Mkt&Fin",
        "help": "Elective track chosen during MBA degree.",
    },
}

FEATURE_NAME_MAP = {
    "num__ssc_p": "10th Grade Percentage (SSC %)",
    "num__hsc_p": "12th Grade Percentage (HSC %)",
    "num__degree_p": "Undergraduate Degree Percentage",
    "num__etest_p": "Employability / Aptitude Test Score (%)",
    "num__mba_p": "MBA Percentage",
    "cat__gender_M": "Gender: Male (vs Female)",
    "cat__ssc_b_Others": "10th Board: Others (vs Central)",
    "cat__hsc_b_Others": "12th Board: Others (vs Central)",
    "cat__hsc_s_Commerce": "12th Stream: Commerce (vs Arts)",
    "cat__hsc_s_Science": "12th Stream: Science (vs Arts)",
    "cat__degree_t_Others": "Degree Stream: Others (vs Comm&Mgmt)",
    "cat__degree_t_Sci&Tech": "Degree Stream: Sci&Tech (vs Comm&Mgmt)",
    "cat__workex_Yes": "Prior Work Experience: Yes (vs No)",
    "cat__specialisation_Mkt&HR": "MBA Track: Mkt&HR (vs Mkt&Fin)",
}

EDUCATION_ROUTES = ["12th/HSC route", "Diploma route"]
MBA_STATUS_OPTIONS = ["Completed", "Pursuing", "Not Pursuing"]

DIPLOMA_METADATA = {
    "diploma_p": {
        "label": "Diploma Aggregate Percentage",
        "min": 0.0,
        "max": 100.0,
        "default": 70.0,
        "step": 0.1,
        "help": "Polytechnic / vocational 3-year diploma percentage. Valid range: 0.0% - 100.0%.",
    }
}

SAMPLE_PROFILES = {
    "Placed Student (Strong Academic & Experience Profile)": {
        "education_route": "12th/HSC route",
        "mba_status": "Completed",
        "ssc_p": 79.0,
        "hsc_p": 78.0,
        "hsc_b": "Central",
        "hsc_s": "Commerce",
        "degree_p": 75.0,
        "degree_t": "Comm&Mgmt",
        "etest_p": 85.0,
        "mba_p": 67.0,
        "gender": "M",
        "ssc_b": "Central",
        "workex": "Yes",
        "specialisation": "Mkt&Fin",
    },
    "At-Risk Student (Lower Marks, No Experience Profile)": {
        "education_route": "12th/HSC route",
        "mba_status": "Completed",
        "ssc_p": 54.0,
        "hsc_p": 51.0,
        "hsc_b": "Others",
        "hsc_s": "Science",
        "degree_p": 55.0,
        "degree_t": "Sci&Tech",
        "etest_p": 56.0,
        "mba_p": 54.0,
        "gender": "F",
        "ssc_b": "Others",
        "workex": "No",
        "specialisation": "Mkt&HR",
    },
    "Diploma Student (Polytechnic Route Demonstration)": {
        "education_route": "Diploma route",
        "mba_status": "Completed",
        "ssc_p": 72.0,
        "diploma_p": 76.0,
        "degree_p": 70.0,
        "degree_t": "Sci&Tech",
        "etest_p": 75.0,
        "mba_p": 64.0,
        "gender": "M",
        "ssc_b": "Central",
        "workex": "Yes",
        "specialisation": "Mkt&Fin",
    },
}


def find_data_file(filename="Placement.csv"):
    """Locate dataset file safely across case-sensitive environments."""
    for path in [Path(filename), Path("Placement.csv"), Path("placement.csv"), Path("../Placement.csv")]:
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"Dataset '{filename}' could not be located in workspace.")


def load_raw_data(filepath=None):
    """Load raw dataset as pandas DataFrame."""
    resolved_path = find_data_file(filepath if filepath else "Placement.csv")
    df = pd.read_csv(resolved_path)
    return df


def get_clean_data(filepath=None):
    """
    Load dataset and separate into pre-placement features X and binary target y.
    Excludes identifiers (sl_no) and target leakage (salary).
    """
    df = load_raw_data(filepath)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataset.")

    drop_cols = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    X = df.drop(columns=drop_cols + [TARGET_COLUMN])
    y = (df[TARGET_COLUMN] == "Placed").astype(int)

    # Validate feature consistency
    expected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    missing = [c for c in expected_cols if c not in X.columns]
    if missing:
        raise ValueError(f"Clean dataset missing expected features: {missing}")

    return X, y


def validate_input_dataframe(input_df):
    """
    Ensure user input adheres to schema, valid ranges, and expected categories.
    Returns (is_valid: bool, errors: list[str]).
    """
    errors = []
    expected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    for col in expected_cols:
        if col not in input_df.columns:
            errors.append(f"Missing required feature '{col}'.")

    if errors:
        return False, errors

    # Check numerical ranges
    for col in NUMERICAL_FEATURES:
        val = input_df[col].iloc[0]
        try:
            val_float = float(val)
            meta = FEATURE_METADATA.get(col, {})
            if val_float < meta.get("min", 0.0) or val_float > meta.get("max", 100.0):
                errors.append(f"{meta.get('label', col)} ({val_float}) is outside allowable range [0, 100].")
        except (ValueError, TypeError):
            errors.append(f"Value for {col} must be a valid float.")

    # Check categorical options
    for col in CATEGORICAL_FEATURES:
        val = str(input_df[col].iloc[0])
        valid_opts = FEATURE_METADATA.get(col, {}).get("options", [])
        if val not in valid_opts:
            errors.append(f"Invalid option '{val}' for {col}. Must be one of {valid_opts}.")

    return len(errors) == 0, errors


def validate_student_profile(profile_dict):
    """
    Validate a comprehensive student profile handling both 12th/HSC and Diploma routes,
    as well as MBA completion status.
    
    Guarantees:
    - Never silently replaces missing HSC or MBA marks with zeros or arbitrary values.
    - Accurately checks whether the profile matches the training distribution of the
      Logistic Regression model.
    - If incompatible (Diploma route or in-progress/non-MBA), sets is_model_compatible=False
      and returns detailed academic advisories.
      
    Returns:
        dict containing:
            - is_valid (bool): whether user inputs are syntactically and numerically valid.
            - is_model_compatible (bool): whether the profile matches the trained model schema.
            - errors (list[str]): validation error messages.
            - advisories (list[str]): academic and domain advisory messages.
            - clean_model_input (dict or None): 12-feature dict for inference if compatible.
    """
    errors = []
    advisories = []

    # 1. Validate route and status selectors
    route = profile_dict.get("education_route", "12th/HSC route")
    if route not in EDUCATION_ROUTES:
        errors.append(f"Invalid education route '{route}'. Expected one of {EDUCATION_ROUTES}.")

    mba_status = profile_dict.get("mba_status", "Completed")
    if mba_status not in MBA_STATUS_OPTIONS:
        errors.append(f"Invalid MBA status '{mba_status}'. Expected one of {MBA_STATUS_OPTIONS}.")

    # 2. Validate common numerical fields
    common_num = ["ssc_p", "degree_p", "etest_p"]
    for col in common_num:
        val = profile_dict.get(col)
        if val is None:
            errors.append(f"Missing required field '{col}'.")
            continue
        try:
            val_float = float(val)
            meta = FEATURE_METADATA.get(col, {})
            if val_float < meta.get("min", 0.0) or val_float > meta.get("max", 100.0):
                errors.append(f"{meta.get('label', col)} ({val_float}) is outside allowable range [0, 100].")
        except (ValueError, TypeError):
            errors.append(f"Value for {col} must be a valid float percentage.")

    # 3. Validate common categorical fields
    common_cat = ["gender", "ssc_b", "degree_t", "workex"]
    for col in common_cat:
        val = profile_dict.get(col)
        if val is None:
            errors.append(f"Missing required field '{col}'.")
            continue
        valid_opts = FEATURE_METADATA.get(col, {}).get("options", [])
        if str(val) not in valid_opts:
            errors.append(f"Invalid option '{val}' for {col}. Must be one of {valid_opts}.")

    # 4. Route-specific validation
    if route == "Diploma route":
        diploma_val = profile_dict.get("diploma_p")
        if diploma_val is None:
            errors.append("Diploma percentage is required for the Diploma pathway.")
        else:
            try:
                dip_float = float(diploma_val)
                dip_meta = DIPLOMA_METADATA["diploma_p"]
                if dip_float < dip_meta["min"] or dip_float > dip_meta["max"]:
                    errors.append(f"Diploma Percentage ({dip_float}) is outside allowable range [0, 100].")
            except (ValueError, TypeError):
                errors.append("Diploma percentage must be a valid float.")

        advisories.append(
            "The current placement model was trained on historical data containing only 12th/HSC candidates "
            "(215 records, 0 diploma students). Missing HSC features will NOT be silently imputed with zero "
            "or arbitrary values. A retrained model on diploma-student data is required for valid inference."
        )

    elif route == "12th/HSC route":
        hsc_val = profile_dict.get("hsc_p")
        if hsc_val is None:
            errors.append("12th/HSC percentage is required for the 12th/HSC pathway.")
        else:
            try:
                hsc_float = float(hsc_val)
                hsc_meta = FEATURE_METADATA["hsc_p"]
                if hsc_float < hsc_meta["min"] or hsc_float > hsc_meta["max"]:
                    errors.append(f"HSC Percentage ({hsc_float}) is outside allowable range [0, 100].")
            except (ValueError, TypeError):
                errors.append("HSC percentage must be a valid float.")

        for col in ["hsc_b", "hsc_s"]:
            val = profile_dict.get(col)
            valid_opts = FEATURE_METADATA.get(col, {}).get("options", [])
            if str(val) not in valid_opts:
                errors.append(f"Invalid option '{val}' for {col}. Must be one of {valid_opts}.")

    # 5. MBA status validation
    if mba_status == "Completed":
        mba_val = profile_dict.get("mba_p")
        if mba_val is None:
            errors.append("MBA percentage is required when MBA status is 'Completed'.")
        else:
            try:
                mba_float = float(mba_val)
                mba_meta = FEATURE_METADATA["mba_p"]
                if mba_float < mba_meta["min"] or mba_float > mba_meta["max"]:
                    errors.append(f"MBA Percentage ({mba_float}) is outside allowable range [0, 100].")
            except (ValueError, TypeError):
                errors.append("MBA percentage must be a valid float.")

        spec_val = profile_dict.get("specialisation")
        valid_specs = FEATURE_METADATA["specialisation"]["options"]
        if str(spec_val) not in valid_specs:
            errors.append(f"Invalid MBA specialization '{spec_val}'. Must be one of {valid_specs}.")

    elif mba_status in ["Pursuing", "Not Pursuing"]:
        advisories.append(
            f"The candidate has MBA status '{mba_status}'. The existing model was trained on post-MBA placement "
            "records and requires completed MBA marks and specialization. Missing MBA credentials will NOT be "
            "silently replaced with zeros or arbitrary values."
        )

    is_valid = len(errors) == 0
    is_model_compatible = is_valid and (route == "12th/HSC route") and (mba_status == "Completed")

    clean_model_input = None
    if is_model_compatible:
        clean_model_input = {
            "ssc_p": float(profile_dict["ssc_p"]),
            "hsc_p": float(profile_dict["hsc_p"]),
            "degree_p": float(profile_dict["degree_p"]),
            "etest_p": float(profile_dict["etest_p"]),
            "mba_p": float(profile_dict["mba_p"]),
            "gender": str(profile_dict["gender"]),
            "ssc_b": str(profile_dict["ssc_b"]),
            "hsc_b": str(profile_dict["hsc_b"]),
            "hsc_s": str(profile_dict["hsc_s"]),
            "degree_t": str(profile_dict["degree_t"]),
            "workex": str(profile_dict["workex"]),
            "specialisation": str(profile_dict["specialisation"]),
        }

    return {
        "is_valid": is_valid,
        "is_model_compatible": is_model_compatible,
        "errors": errors,
        "advisories": advisories,
        "clean_model_input": clean_model_input,
    }

