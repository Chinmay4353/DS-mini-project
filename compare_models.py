"""
DS Lab Student Placement Prediction - Phase 1: Algorithm Comparison
====================================================================
Academic Mini-Project Validation Script
Evaluates candidate classification algorithms under leakage-safe conditions.

Algorithms Compared:
1. Logistic Regression (L2 regularization, max_iter=500, random_state=42)
2. Decision Tree Classifier (pruned & unconstrained baselines, random_state=42)
3. Random Forest Classifier (ensemble of 100 trees, random_state=42)

Validation Protocol:
- Pre-placement features only (12 features)
- sl_no (identifier) and salary (target leakage) excluded
- 80/20 Stratified train/test split (random_state=42)
- 5-Fold Stratified Cross-Validation on the 80% training set only
- Preprocessing inside scikit-learn Pipeline (StandardScaler + OneHotEncoder)
- Evaluation on untouched 20% holdout test set
"""

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    make_scorer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Fixed random state for strict reproducibility
RANDOM_SEED = 42

def load_and_validate_data(filepath="Placement.csv"):
    """Load dataset, validate schema, and isolate eligible pre-placement features."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        # Fallback in case of lowercase on case-sensitive filesystems
        df = pd.read_csv("placement.csv")

    print("=" * 80)
    print("DATASET INGESTION & FEATURE ISOLATION")
    print("=" * 80)
    print(f"Raw dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Target variable
    target_col = "status"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset!")

    # Verify class balance
    class_counts = df[target_col].value_counts()
    print(f"Target variable: '{target_col}'")
    print(f"  Placed:     {class_counts.get('Placed', 0)} ({class_counts.get('Placed', 0)/len(df)*100:.2f}%)")
    print(f"  Not Placed: {class_counts.get('Not Placed', 0)} ({class_counts.get('Not Placed', 0)/len(df)*100:.2f}%)")

    # Features to exclude:
    # 'sl_no' is an arbitrary row serial number (zero predictive value)
    # 'salary' is post-placement information and 100% target leakage (NaN for Not Placed)
    drop_cols = ["sl_no", "salary", target_col]
    X = df.drop(columns=drop_cols)
    
    # Binary target encoding: Placed = 1 (positive class), Not Placed = 0 (negative class)
    y = (df[target_col] == "Placed").astype(int)

    # Segregate continuous and categorical features
    num_features = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
    cat_features = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"]

    # Verify that all designated features exist in X
    missing_num = [col for col in num_features if col not in X.columns]
    missing_cat = [col for col in cat_features if col not in X.columns]
    if missing_num or missing_cat:
        raise ValueError(f"Missing expected columns! Num: {missing_num}, Cat: {missing_cat}")

    print(f"\nEligible pre-placement features ({len(X.columns)}):")
    print(f"  Continuous ({len(num_features)}):  {num_features}")
    print(f"  Categorical ({len(cat_features)}): {cat_features}")
    print(f"Excluded features: {['sl_no', 'salary']} (identifier & post-placement target leakage)")

    return X, y, num_features, cat_features


def build_preprocessor(num_features, cat_features):
    """
    Build a leakage-safe ColumnTransformer.
    StandardScaler for continuous features, OneHotEncoder(drop='first') for categorical features.
    Transformations are fitted strictly on training data within each fold or split.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features),
        ],
        remainder="drop",
    )
    return preprocessor


def get_candidate_models():
    """Define candidates for algorithm comparison."""
    return {
        "Logistic Regression (L2, C=1.0)": LogisticRegression(
            C=1.0,
            max_iter=500,
            random_state=RANDOM_SEED,
            solver="lbfgs"
        ),
        "Decision Tree (Pruned, max_depth=3)": DecisionTreeClassifier(
            max_depth=3,
            random_state=RANDOM_SEED
        ),
        "Decision Tree (Unconstrained)": DecisionTreeClassifier(
            random_state=RANDOM_SEED
        ),
        "Random Forest (n=100, max_depth=4)": RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=RANDOM_SEED
        ),
        "Random Forest (n=100, Unconstrained)": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED
        ),
    }


def run_experiment():
    """Execute stratified holdout split, 5-fold CV on train set, and test evaluation."""
    X, y, num_features, cat_features = load_and_validate_data()

    # Reserve 20% stratified holdout test set (N=43), untouched during model selection
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )

    print("\n" + "=" * 80)
    print("DATASET PARTITIONING (80/20 STRATIFIED SPLIT)")
    print("=" * 80)
    print(f"Training set: {X_train.shape[0]} samples (Placed: {y_train.sum()}, Not Placed: {len(y_train) - y_train.sum()})")
    print(f"Holdout test: {X_test.shape[0]} samples (Placed: {y_test.sum()}, Not Placed: {len(y_test) - y_test.sum()})")

    # Define scorers with zero_division=0 to explicitly handle edge cases where a model predicts 1 class
    scorers = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }

    # Stratified 5-Fold Cross Validation ONLY on the 80% training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    models = get_candidate_models()
    results = []

    print("\n" + "=" * 80)
    print("PERFORMING 5-FOLD STRATIFIED CV ON 80% TRAINING SET (N=172)")
    print("=" * 80)

    for name, clf in models.items():
        preprocessor = build_preprocessor(num_features, cat_features)
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

        # 5-Fold Cross-Validation on training set
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scorers,
            return_train_score=False,
        )

        cv_acc_mean = cv_scores["test_accuracy"].mean()
        cv_acc_std = cv_scores["test_accuracy"].std()
        cv_f1_mean = cv_scores["test_f1"].mean()
        cv_f1_std = cv_scores["test_f1"].std()
        cv_prec_mean = cv_scores["test_precision"].mean()
        cv_rec_mean = cv_scores["test_recall"].mean()

        # Fit pipeline on full 80% training set and evaluate on 20% holdout test set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        test_prec = precision_score(y_test, y_pred, zero_division=0)
        test_rec = recall_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)  # [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()

        # Specificity / Recall of the negative class ('Not Placed')
        not_placed_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results.append({
            "name": name,
            "cv_acc_mean": cv_acc_mean,
            "cv_acc_std": cv_acc_std,
            "cv_f1_mean": cv_f1_mean,
            "cv_f1_std": cv_f1_std,
            "cv_prec_mean": cv_prec_mean,
            "cv_rec_mean": cv_rec_mean,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "test_prec": test_prec,
            "test_rec": test_rec,
            "not_placed_rec": not_placed_recall,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "pipeline": pipeline,
            "y_pred": y_pred,
        })

    # Print summary comparison table
    print("\n" + "=" * 115)
    print("PHASE 1 ALGORITHM COMPARISON RESULTS")
    print("=" * 115)
    header = (
        f"{'Algorithm':38s} | "
        f"{'Train 5-Fold CV Acc':19s} | "
        f"{'Train 5-Fold CV F1':18s} | "
        f"{'Test Acc':8s} | "
        f"{'Test F1':8s} | "
        f"{'Test Prec':9s} | "
        f"{'Test Rec':8s} | "
        f"{'CM [TN,FP;FN,TP]':16s}"
    )
    print(header)
    print("-" * 115)

    for r in results:
        cm_str = f"[{r['tn']},{r['fp']};{r['fn']},{r['tp']}]"
        cv_acc_str = f"{r['cv_acc_mean']*100:5.2f}% +/- {r['cv_acc_std']*100:4.2f}%"
        cv_f1_str = f"{r['cv_f1_mean']*100:5.2f}% +/- {r['cv_f1_std']*100:4.2f}%"
        row = (
            f"{r['name']:38s} | "
            f"{cv_acc_str:19s} | "
            f"{cv_f1_str:18s} | "
            f"{r['test_acc']*100:6.2f}%  | "
            f"{r['test_f1']*100:6.2f}%  | "
            f"{r['test_prec']*100:6.2f}%   | "
            f"{r['test_rec']*100:6.2f}%  | "
            f"{cm_str:16s}"
        )
        print(row)
    print("=" * 115)

    # Detailed per-candidate inspection
    print("\n" + "=" * 80)
    print("DETAILED HOLDOUT TEST EVALUATION PER CANDIDATE (N=43)")
    print("=" * 80)
    for r in results:
        print(f"\n>>> Model: {r['name']}")
        print(f"Confusion Matrix (True Labels down, Predicted across):")
        print(f"                 Pred Not Placed (0)   Pred Placed (1)")
        print(f"  Actual Not Placed (0)       {r['tn']:2d}                 {r['fp']:2d}")
        print(f"  Actual Placed (1)           {r['fn']:2d}                 {r['tp']:2d}")
        print(f"\nKey Metrics:")
        print(f"  - Overall Accuracy:        {r['test_acc']*100:.2f}% ({r['tn']+r['tp']}/43 correct)")
        print(f"  - Placed Precision:        {r['test_prec']*100:.2f}% (when predicted Placed, {r['tp']}/{r['tp']+r['fp']} correct)")
        print(f"  - Placed Recall:           {r['test_rec']*100:.2f}% ({r['tp']}/{r['tp']+r['fn']} placed students caught)")
        print(f"  - Not Placed Specificity:  {r['not_placed_rec']*100:.2f}% ({r['tn']}/{r['tn']+r['fp']} unplaced students correctly identified)")
        print(f"  - Macro F1-Score:          {f1_score(y_test, r['y_pred'], average='macro', zero_division=0)*100:.2f}%")
        print(f"  - Weighted F1-Score:       {f1_score(y_test, r['y_pred'], average='weighted', zero_division=0)*100:.2f}%")
        print(f"  - Generalization Gap (CV Acc - Test Acc): {(r['cv_acc_mean'] - r['test_acc'])*100:+.2f}%")

    # Logistic Regression coefficient inspection for academic defense
    lr_res = [r for r in results if "Logistic Regression" in r["name"]][0]
    lr_pipe = lr_res["pipeline"]
    feature_names = lr_pipe.named_steps["preprocessor"].get_feature_names_out()
    coefs = lr_pipe.named_steps["classifier"].coef_[0]
    intercept = lr_pipe.named_steps["classifier"].intercept_[0]

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs,
        "Odds Ratio [exp(B)]": np.exp(coefs),
    }).sort_values(by="Coefficient", ascending=False)

    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION FEATURE IMPORTANCE (LOG-ODDS & ODDS RATIOS)")
    print("=" * 80)
    print(f"Model Intercept: {intercept:.4f} (Base log-odds)")
    for _, row in coef_df.iterrows():
        direction = "Positive (Increases Placement Odds)" if row["Coefficient"] > 0 else "Negative (Decreases Placement Odds)"
        print(f"  {row['Feature']:35s} | B = {row['Coefficient']:+7.4f} | exp(B) = {row['Odds Ratio [exp(B)]']:6.3f} | {direction}")

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE: No model files saved. Final selection awaiting user review.")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
