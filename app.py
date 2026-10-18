"""
Student Placement Prediction & Decision Support System
======================================================
DS Lab Academic Mini-Project Dashboard
Demonstrates end-to-end data science methodology: dataset exploration,
leakage-free preprocessing, candidate model benchmarking, pre-placement inference,
and educational prediction explanations for viva examination.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Local modular imports
from src.data import (
    load_raw_data,
    get_clean_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_METADATA,
    DIPLOMA_METADATA,
    EDUCATION_ROUTES,
    MBA_STATUS_OPTIONS,
    SAMPLE_PROFILES,
    FEATURE_NAME_MAP,
    validate_input_dataframe,
    validate_student_profile,
)
from src.model import (
    load_pipeline,
    predict_single_student,
    explain_single_prediction,
    compute_benchmark_comparison,
    DEFAULT_MODEL_PATH,
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & REFINED LIGHT-THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Placement Prediction | DS Lab",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Centralized light-theme CSS
st.markdown(
    """
    <style>
    /* Global Typography & Palette */
    :root {
        --primary-navy: #0F172A;
        --secondary-navy: #1E3A8A;
        --accent-blue: #2563EB;
        --bg-light: #F8FAFC;
        --border-color: #E2E8F0;
        --success-emerald: #059669;
        --danger-rose: #DC2626;
        --warning-amber: #D97706;
    }

    /* Streamlit Sidebar Polish & Active Navigation Accent */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > label {
        font-size: 0.85rem;
        font-weight: 700;
        color: #334155;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        transition: all 0.15s ease-in-out;
        cursor: pointer;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: #F1F5F9;
        border-color: #CBD5E1;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background-color: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
        border-left: 5px solid #1E3A8A !important;
        font-weight: 600 !important;
        color: #1E3A8A !important;
    }

    /* Dashboard Metric Card Polish */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        font-weight: 600;
        color: #475569;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0F172A;
    }

    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.02rem;
        color: #475569;
        margin-bottom: 1.4rem;
        line-height: 1.5;
    }
    .section-title {
        font-size: 1.22rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    /* Informational & Status Cards */
    .card-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    .notice-box {
        background-color: #F0F9FF;
        border: 1px solid #BAE6FD;
        border-left: 4px solid #0284C7;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 16px;
        font-size: 0.93rem;
        color: #0C4A6E;
        line-height: 1.55;
    }
    .warning-box {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid #D97706;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 16px;
        font-size: 0.93rem;
        color: #78350F;
        line-height: 1.55;
    }
    .boundary-box {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 16px 18px;
        height: 100%;
    }

    /* Prominent Prediction Result Cards */
    .result-placed {
        background-color: #ECFDF5;
        border: 2px solid #059669;
        border-radius: 10px;
        padding: 22px 24px;
        color: #064E3B;
        box-shadow: 0 4px 6px -1px rgba(5, 150, 105, 0.1);
    }
    .result-not-placed {
        background-color: #FEF2F2;
        border: 2px solid #DC2626;
        border-radius: 10px;
        padding: 22px 24px;
        color: #7F1D1D;
        box-shadow: 0 4px 6px -1px rgba(220, 38, 38, 0.1);
    }
    .result-badge-placed {
        display: inline-block;
        background-color: #D1FAE5;
        color: #065F46;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .result-badge-not-placed {
        display: inline-block;
        background-color: #FEE2E2;
        color: #991B1B;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .result-score-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        opacity: 0.85;
    }
    .result-score-value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 4px 0 8px 0;
    }
    .result-sub-metric {
        font-size: 0.93rem;
        margin-bottom: 8px;
    }
    .result-desc {
        font-size: 0.88rem;
        line-height: 1.45;
        border-top: 1px solid rgba(0, 0, 0, 0.08);
        padding-top: 8px;
        margin-top: 6px;
    }

    /* Pipeline Flow Step Box */
    .flow-step {
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 10px 8px;
        text-align: center;
        font-size: 0.85rem;
        font-weight: 500;
        color: #1E293B;
    }
    .flow-arrow {
        text-align: center;
        font-size: 1.2rem;
        color: #64748B;
        font-weight: bold;
        line-height: 2.2;
    }

    /* Viva Talking Point Card */
    .viva-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
        font-size: 0.92rem;
        line-height: 1.5;
        color: #1E293B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_raw_data():
    """Load raw dataset with caching."""
    return load_raw_data()


@st.cache_resource
def get_cached_model_bundle():
    """Load pre-trained model bundle once to avoid retraining on interactions."""
    return load_pipeline(DEFAULT_MODEL_PATH)


@st.cache_data
def get_cached_benchmark():
    """Run candidate model comparison dynamically for the evaluation tab."""
    return compute_benchmark_comparison()


# Initialize state safely
try:
    raw_df = get_cached_raw_data()
    model_bundle = get_cached_model_bundle()
except Exception as e:
    st.error(f"Error initializing application: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & PROJECT METADATA
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 DS Lab Project")
    st.markdown("**Student Placement Prediction System**")
    st.caption("Machine Learning classification & decision support dashboard.")

    st.markdown("---")
    page = st.radio(
        "Navigate Application:",
        [
            "📊 Overview & KPIs",
            "🔍 Exploratory Data Insights",
            "🎯 Student Prediction",
            "📈 Model Evaluation",
            "📖 About & Methodology",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### System Configuration")
    st.markdown("**Selected Model:** Logistic Regression ($L_2$)")
    st.markdown("**Regularization:** $C = 1.0$ (Ridge)")
    st.markdown("**Training Set:** 172 students (80%)")
    st.markdown("**Holdout Test:** 43 students (20%)")
    st.markdown("**Test Accuracy:** **86.05%** (37/43)")
    st.markdown("**Test Precision:** **92.86%**")
    st.markdown("**Submission Deadline:** 23 October 2026")

    st.markdown("---")
    st.caption("Department of Data Science & Engineering")

# -----------------------------------------------------------------------------
# PAGE 1: OVERVIEW & DATASET STATISTICS
# -----------------------------------------------------------------------------
if page == "📊 Overview & KPIs":
    st.markdown('<div class="main-header">Student Placement Prediction & Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">An interpretable Data Science system designed to predict campus placement likelihood and identify students needing early academic coaching.</div>', unsafe_allow_html=True)

    # Dataset Leakage Hygiene Alert
    st.markdown(
        """
        <div class="notice-box">
            <b>Data Hygiene & Leakage Safeguard:</b> Candidate predictions utilize exclusively <b>pre-placement attributes</b>. 
            The database identifier (<code>sl_no</code>) and post-recruitment salary (<code>salary</code>) were removed prior to model training 
            to prevent 100% target leakage, as salary is strictly populated for already-placed students.
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_records = len(raw_df)
    placed_count = int((raw_df["status"] == "Placed").sum())
    not_placed_count = int((raw_df["status"] == "Not Placed").sum())
    placement_rate = (placed_count / total_records) * 100.0
    non_placement_rate = (not_placed_count / total_records) * 100.0
    median_degree = float(raw_df["degree_p"].median())
    workex_pct = float((raw_df["workex"] == "Yes").sum()) / total_records * 100.0

    # KPI Summary Cards
    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    with col_k1:
        st.metric("Total Candidates", f"{total_records}", help="Total observations in Placement.csv")
    with col_k2:
        st.metric("Placed Students", f"{placed_count}", f"{placement_rate:.1f}% placement rate")
    with col_k3:
        st.metric("Not Placed Students", f"{not_placed_count}", f"{non_placement_rate:.1f}% unplaced")
    with col_k4:
        st.metric("Median Degree Marks", f"{median_degree:.1f}%", help="Undergraduate percentage median")
    with col_k5:
        st.metric("Prior Experience", f"{workex_pct:.1f}%", help="Candidates with pre-MBA work experience")

    st.markdown("---")

    # Target Distribution Chart & Summary Table
    col_chart, col_stats = st.columns([1, 1])

    with col_chart:
        st.markdown('<div class="section-title">Target Variable Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=130)
        palette = {"Placed": "#059669", "Not Placed": "#DC2626"}
        sns.countplot(x="status", data=raw_df, hue="status", palette=palette, legend=False, ax=ax)
        ax.set_title("Ground Truth Placement Outcome (N = 215)", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Placement Status", fontsize=10)
        ax.set_ylabel("Candidate Count", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        
        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                pct = (h / total_records) * 100
                ax.annotate(f"{int(h)} ({pct:.1f}%)",
                            (p.get_x() + p.get_width() / 2.0, h / 2.0),
                            ha="center", va="center", fontsize=10.5, color="white", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_stats:
        st.markdown('<div class="section-title">Academic Marks Summary Statistics</div>', unsafe_allow_html=True)
        num_summary = raw_df[NUMERICAL_FEATURES].describe().T[["mean", "std", "min", "50%", "max"]]
        num_summary.index = [FEATURE_METADATA[c]["label"] for c in num_summary.index]
        num_summary.columns = ["Mean (%)", "Std Dev", "Min (%)", "Median (%)", "Max (%)"]
        st.dataframe(num_summary.style.format("{:.2f}"), width="stretch")
        st.caption("Distribution of continuous academic percentages across 10th, 12th, Degree, Employability Test, and MBA.")

    st.markdown("---")

    # Scope & Boundaries (What system CAN & CANNOT do)
    st.markdown('<div class="section-title">System Scope & Operational Boundaries</div>', unsafe_allow_html=True)
    col_can, col_cannot = st.columns(2)

    with col_can:
        st.markdown(
            """
            <div class="boundary-box">
                <h4 style="color: #065F46; margin-top: 0;">✅ What This System CAN Do:</h4>
                <ul style="font-size: 0.92rem; line-height: 1.6; padding-left: 20px;">
                    <li><b>Early Screening:</b> Estimate placement likelihood before campus recruitment drives commence.</li>
                    <li><b>Actionable Risk Alerting:</b> Identify students with academic combinations statistically associated with low placement odds.</li>
                    <li><b>Transparent Interpretability:</b> Explain how academic percentages and work experience contributed to the model score.</li>
                    <li><b>Decision Support:</b> Aid faculty advisors in allocating interview coaching and resume mentorship resources.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_cannot:
        st.markdown(
            """
            <div class="boundary-box">
                <h4 style="color: #991B1B; margin-top: 0;">❌ What This System CANNOT Do:</h4>
                <ul style="font-size: 0.92rem; line-height: 1.6; padding-left: 20px;">
                    <li><b>No Placement Guarantees:</b> The output is a mathematical probability, not a deterministic hiring contract.</li>
                    <li><b>No Salary Estimation:</b> Post-placement compensation is excluded due to complete target leakage.</li>
                    <li><b>No Company-Specific Predictions:</b> The model evaluates generic campus placement, not specific corporate hiring filters.</li>
                    <li><b>No Unvalidated Diploma Inferences:</b> Historical data contains zero polytechnic entrants; evaluating diploma routes without retraining is strictly prevented.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Dataset Explorer Expander
    with st.expander("🔍 Interactive Dataset Explorer & Filtering", expanded=False):
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            filter_status = st.multiselect("Filter by Placement Outcome:", options=["Placed", "Not Placed"], default=["Placed", "Not Placed"])
        with f_col2:
            filter_workex = st.multiselect("Filter by Work Experience:", options=["Yes", "No"], default=["Yes", "No"])

        filtered_df = raw_df[
            raw_df["status"].isin(filter_status) & raw_df["workex"].isin(filter_workex)
        ]
        st.dataframe(filtered_df, width="stretch")
        st.caption(f"Displaying {len(filtered_df)} of {len(raw_df)} student records.")

# -----------------------------------------------------------------------------
# PAGE 2: EXPLORATORY DATA INSIGHTS
# -----------------------------------------------------------------------------
elif page == "🔍 Exploratory Data Insights":
    st.markdown('<div class="main-header">Exploratory Data Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Bivariate distributions and empirical associations between student characteristics and placement success.</div>', unsafe_allow_html=True)

    tab_academic, tab_categorical, tab_corr = st.tabs([
        "📚 Academic Performance",
        "💼 Experience & Specialization",
        "🧩 Correlation Analysis"
    ])

    with tab_academic:
        st.markdown('<div class="section-title">Academic Percentages vs. Placement Outcome</div>', unsafe_allow_html=True)
        st.markdown("Examine how academic percentages across secondary, higher secondary, undergraduate, and post-graduate levels separate placed vs. unplaced cohorts.")

        feature_pick = st.selectbox(
            "Select Academic Metric to Compare:",
            options=["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"],
            format_func=lambda x: FEATURE_METADATA[x]["label"],
        )

        col_box, col_kde = st.columns(2)
        with col_box:
            fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=130)
            palette = {"Placed": "#059669", "Not Placed": "#DC2626"}
            sns.boxplot(x="status", y=feature_pick, data=raw_df, hue="status", palette=palette, legend=False, ax=ax)
            ax.set_title(f"Boxplot: {FEATURE_METADATA[feature_pick]['label']}", fontsize=11, fontweight="bold")
            ax.set_ylabel("Marks Percentage (%)", fontsize=10)
            ax.set_xlabel("Placement Status", fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_kde:
            fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=130)
            sns.kdeplot(data=raw_df, x=feature_pick, hue="status", palette=palette, fill=True, common_norm=False, ax=ax)
            ax.set_title(f"Density Curve: {FEATURE_METADATA[feature_pick]['label']}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Marks Percentage (%)", fontsize=10)
            ax.set_ylabel("Estimated Density", fontsize=10)
            ax.grid(axis="x", linestyle="--", alpha=0.4)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown(
            f"""
            <div class="notice-box">
                <b>Chart Interpretation:</b> Placed candidates consistently show higher median percentages in 10th (<code>ssc_p</code>, $r = +0.61$) 
                and Undergrad Degree (<code>degree_p</code>, $r = +0.48$). 
                <br><b>Academic Integrity Note:</b> These charts demonstrate <b>statistical association</b> in historical university recruitment. 
                They do <i>not</i> demonstrate direct causality (e.g., marks alone do not automatically grant jobs without interview performance).
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_categorical:
        st.markdown('<div class="section-title">Impact of Prior Experience & MBA Specialization Track</div>', unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("#### Prior Work Experience")
            workex_ct = pd.crosstab(raw_df["workex"], raw_df["status"], normalize="index") * 100
            fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=130)
            workex_ct.plot(kind="bar", stacked=True, color=["#DC2626", "#059669"], ax=ax)
            ax.set_title("Placement Rate by Work Experience (%)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Cohort Percentage (%)", fontsize=10)
            ax.set_xlabel("Candidate Has Prior Work Experience", fontsize=10)
            plt.xticks(rotation=0)
            ax.legend(["Not Placed", "Placed"], loc="upper left")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Candidates with prior work experience achieved an **86.5% placement rate**, versus **59.6%** without work experience (χ² = 15.15, p < 0.001).")

        with col_c2:
            st.markdown("#### MBA Specialization Track")
            spec_ct = pd.crosstab(raw_df["specialisation"], raw_df["status"], normalize="index") * 100
            fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=130)
            spec_ct.plot(kind="bar", stacked=True, color=["#DC2626", "#059669"], ax=ax)
            ax.set_title("Placement Rate by Specialization (%)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Cohort Percentage (%)", fontsize=10)
            ax.set_xlabel("MBA Specialization Elective", fontsize=10)
            plt.xticks(rotation=0)
            ax.legend(["Not Placed", "Placed"], loc="upper left")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Marketing & Finance students observed a **79.2% placement rate**, versus **55.8%** for Marketing & HR (χ² = 12.44, p < 0.001).")

    with tab_corr:
        st.markdown('<div class="section-title">Linear Correlation Matrix (Continuous Features)</div>', unsafe_allow_html=True)
        corr_matrix = raw_df[NUMERICAL_FEATURES].corr()
        fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=130)
        readable_ticks = ["10th (SSC)", "12th (HSC)", "Degree", "E-Test", "MBA"]
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="Blues",
            fmt=".2f",
            linewidths=0.8,
            xticklabels=readable_ticks,
            yticklabels=readable_ticks,
            ax=ax
        )
        ax.set_title("Pearson Correlation Coefficients (r)", fontsize=11, fontweight="bold", pad=10)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(
            """
            <div class="notice-box">
                <b>Collinearity Assessment:</b> The highest inter-feature correlation is between 10th and 12th marks ($r = +0.51$). 
                All pairwise correlations fall well below the critical collinearity threshold ($r < 0.80$). 
                Furthermore, the model's $L_2$ regularization penalty provides stable weight shrinkage.
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# PAGE 3: STUDENT PREDICTION & EDUCATIONAL EXPLANATION
# -----------------------------------------------------------------------------
elif page == "🎯 Student Prediction":
    st.markdown('<div class="main-header">Predict Student Placement & Explain the Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluate candidate placement likelihood using 12 pre-placement academic and background attributes.</div>', unsafe_allow_html=True)

    # Viva Demonstration Presets
    st.markdown('<div class="section-title">Quick Profile Presets for Viva Demonstration</div>', unsafe_allow_html=True)
    preset_cols = st.columns(4)

    # Initialize preset session state defaults if absent
    if "preset_data" not in st.session_state:
        st.session_state["preset_data"] = None

    with preset_cols[0]:
        if st.button("📋 Load Placed Profile", width="stretch", help="Loads a high-probability profile (strong marks + work experience)"):
            st.session_state["preset_data"] = SAMPLE_PROFILES["Placed Student (Strong Academic & Experience Profile)"]
    with preset_cols[1]:
        if st.button("⚠️ Load At-Risk Profile", width="stretch", help="Loads a borderline profile with lower scores and no experience"):
            st.session_state["preset_data"] = SAMPLE_PROFILES["At-Risk Student (Lower Marks, No Experience Profile)"]
    with preset_cols[2]:
        if st.button("🎓 Load Diploma Profile", width="stretch", help="Loads a polytechnic diploma route to demonstrate safety safeguards"):
            st.session_state["preset_data"] = SAMPLE_PROFILES["Diploma Student (Polytechnic Route Demonstration)"]
    with preset_cols[3]:
        if st.button("🔄 Reset to Defaults", width="stretch"):
            st.session_state["preset_data"] = None

    p_data = st.session_state["preset_data"]

    # 1. Pathway Selectors
    st.markdown('<div class="section-title">1. Education Pathway & MBA Program Status</div>', unsafe_allow_html=True)
    col_path1, col_path2 = st.columns(2)

    default_route_idx = 0
    if p_data and p_data.get("education_route") == "Diploma route":
        default_route_idx = 1

    with col_path1:
        education_route = st.radio(
            "Secondary Education Pathway:",
            options=EDUCATION_ROUTES,
            index=default_route_idx,
            horizontal=True,
            help="Choose standard 12th/HSC or a polytechnic/vocational 3-year diploma."
        )

    default_mba_status_idx = 0
    if p_data and p_data.get("mba_status") in MBA_STATUS_OPTIONS:
        default_mba_status_idx = MBA_STATUS_OPTIONS.index(p_data["mba_status"])

    with col_path2:
        mba_status = st.selectbox(
            "MBA Degree Completion Status:",
            options=MBA_STATUS_OPTIONS,
            index=default_mba_status_idx,
            help="Placement.csv exclusively evaluates post-MBA campus recruitment."
        )

    # Dynamic Pathway Notices
    if education_route == "Diploma route":
        st.markdown(
            """
            <div class="warning-box">
                <b>Diploma Pathway Active:</b> The 12th/HSC marks, board, and stream fields are hidden. 
                Please enter the student's <b>Diploma aggregate percentage</b>.
                <br><b>Academic Safety Guard:</b> The trained Logistic Regression model was trained exclusively on 12th/HSC students ($N=215$). 
                Missing HSC marks will <b>never</b> be silently replaced with zeros or arbitrary values.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if mba_status != "Completed":
        st.markdown(
            f"""
            <div class="notice-box">
                <b>MBA Status '{mba_status}':</b> The model requires finalized MBA credentials. 
                Incomplete marks will <b>not</b> be silently filled with zeros, and automated prediction will be safely withheld.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Main Input Form
    with st.form("placement_prediction_form"):
        st.markdown('<div class="section-title">2. Academic Performance (Percentage Scores)</div>', unsafe_allow_html=True)
        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            ssc_p = st.number_input(
                FEATURE_METADATA["ssc_p"]["label"],
                min_value=0.0, max_value=100.0,
                value=float(p_data["ssc_p"]) if p_data else 67.0,
                step=0.1, help=FEATURE_METADATA["ssc_p"]["help"]
            )
            degree_p = st.number_input(
                FEATURE_METADATA["degree_p"]["label"],
                min_value=0.0, max_value=100.0,
                value=float(p_data["degree_p"]) if p_data else 66.0,
                step=0.1, help=FEATURE_METADATA["degree_p"]["help"]
            )

        with col_m2:
            if education_route == "Diploma route":
                diploma_p = st.number_input(
                    DIPLOMA_METADATA["diploma_p"]["label"],
                    min_value=0.0, max_value=100.0,
                    value=float(p_data["diploma_p"]) if p_data and "diploma_p" in p_data else 72.0,
                    step=0.1, help=DIPLOMA_METADATA["diploma_p"]["help"]
                )
                hsc_p = None
            else:
                hsc_p = st.number_input(
                    FEATURE_METADATA["hsc_p"]["label"],
                    min_value=0.0, max_value=100.0,
                    value=float(p_data["hsc_p"]) if p_data and "hsc_p" in p_data else 66.0,
                    step=0.1, help=FEATURE_METADATA["hsc_p"]["help"]
                )
                diploma_p = None

            etest_p = st.number_input(
                FEATURE_METADATA["etest_p"]["label"],
                min_value=0.0, max_value=100.0,
                value=float(p_data["etest_p"]) if p_data else 71.0,
                step=0.1, help=FEATURE_METADATA["etest_p"]["help"]
            )

        with col_m3:
            if mba_status == "Completed":
                mba_p = st.number_input(
                    FEATURE_METADATA["mba_p"]["label"],
                    min_value=0.0, max_value=100.0,
                    value=float(p_data["mba_p"]) if p_data and "mba_p" in p_data else 62.0,
                    step=0.1, help=FEATURE_METADATA["mba_p"]["help"]
                )
            else:
                st.info(f"MBA Marks: {mba_status} (Not Applicable)")
                mba_p = None

        st.markdown('<div class="section-title">3. Educational Streams & Boards</div>', unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            if education_route == "12th/HSC route":
                hsc_s_opts = FEATURE_METADATA["hsc_s"]["options"]
                hsc_s_idx = hsc_s_opts.index(p_data["hsc_s"]) if p_data and "hsc_s" in p_data and p_data["hsc_s"] in hsc_s_opts else 0
                hsc_s = st.selectbox(
                    FEATURE_METADATA["hsc_s"]["label"],
                    options=hsc_s_opts, index=hsc_s_idx,
                    format_func=lambda x: FEATURE_METADATA["hsc_s"]["display_names"][x],
                    help=FEATURE_METADATA["hsc_s"]["help"]
                )
            else:
                st.info("HSC Stream: Omitted (Diploma Route)")
                hsc_s = None

            ssc_b_opts = FEATURE_METADATA["ssc_b"]["options"]
            ssc_b_idx = ssc_b_opts.index(p_data["ssc_b"]) if p_data and "ssc_b" in p_data and p_data["ssc_b"] in ssc_b_opts else 0
            ssc_b = st.selectbox(
                FEATURE_METADATA["ssc_b"]["label"],
                options=ssc_b_opts, index=ssc_b_idx,
                format_func=lambda x: FEATURE_METADATA["ssc_b"]["display_names"][x],
                help=FEATURE_METADATA["ssc_b"]["help"]
            )

        with col_s2:
            degree_t_opts = FEATURE_METADATA["degree_t"]["options"]
            degree_t_idx = degree_t_opts.index(p_data["degree_t"]) if p_data and "degree_t" in p_data and p_data["degree_t"] in degree_t_opts else 0
            degree_t = st.selectbox(
                FEATURE_METADATA["degree_t"]["label"],
                options=degree_t_opts, index=degree_t_idx,
                format_func=lambda x: FEATURE_METADATA["degree_t"]["display_names"][x],
                help=FEATURE_METADATA["degree_t"]["help"]
            )

            if education_route == "12th/HSC route":
                hsc_b_opts = FEATURE_METADATA["hsc_b"]["options"]
                hsc_b_idx = hsc_b_opts.index(p_data["hsc_b"]) if p_data and "hsc_b" in p_data and p_data["hsc_b"] in hsc_b_opts else 0
                hsc_b = st.selectbox(
                    FEATURE_METADATA["hsc_b"]["label"],
                    options=hsc_b_opts, index=hsc_b_idx,
                    format_func=lambda x: FEATURE_METADATA["hsc_b"]["display_names"][x],
                    help=FEATURE_METADATA["hsc_b"]["help"]
                )
            else:
                st.info("HSC Board: Omitted (Diploma Route)")
                hsc_b = None

        with col_s3:
            if mba_status == "Completed":
                spec_opts = FEATURE_METADATA["specialisation"]["options"]
                spec_idx = spec_opts.index(p_data["specialisation"]) if p_data and "specialisation" in p_data and p_data["specialisation"] in spec_opts else 0
                specialisation = st.selectbox(
                    FEATURE_METADATA["specialisation"]["label"],
                    options=spec_opts, index=spec_idx,
                    format_func=lambda x: FEATURE_METADATA["specialisation"]["display_names"][x],
                    help=FEATURE_METADATA["specialisation"]["help"]
                )
            else:
                st.info(f"Specialization: N/A ({mba_status})")
                specialisation = None

        st.markdown('<div class="section-title">4. Background & Industry Experience</div>', unsafe_allow_html=True)
        col_b1, col_b2 = st.columns(2)

        with col_b1:
            gender_opts = FEATURE_METADATA["gender"]["options"]
            gender_idx = gender_opts.index(p_data["gender"]) if p_data and "gender" in p_data and p_data["gender"] in gender_opts else 0
            gender = st.selectbox(
                FEATURE_METADATA["gender"]["label"],
                options=gender_opts, index=gender_idx,
                format_func=lambda x: FEATURE_METADATA["gender"]["display_names"][x],
                help=FEATURE_METADATA["gender"]["help"]
            )

        with col_b2:
            workex_opts = FEATURE_METADATA["workex"]["options"]
            workex_idx = workex_opts.index(p_data["workex"]) if p_data and "workex" in p_data and p_data["workex"] in workex_opts else 0
            workex = st.selectbox(
                FEATURE_METADATA["workex"]["label"],
                options=workex_opts, index=workex_idx,
                format_func=lambda x: FEATURE_METADATA["workex"]["display_names"][x],
                help=FEATURE_METADATA["workex"]["help"]
            )

        submit_btn = st.form_submit_button("🔮 Evaluate Candidate Placement Profile", width="stretch")

    # 3. Handle Prediction Submission & Explanation
    if submit_btn:
        profile_dict = {
            "education_route": education_route,
            "mba_status": mba_status,
            "ssc_p": ssc_p,
            "degree_p": degree_p,
            "etest_p": etest_p,
            "gender": gender,
            "ssc_b": ssc_b,
            "degree_t": degree_t,
            "workex": workex,
        }
        if education_route == "Diploma route":
            profile_dict["diploma_p"] = diploma_p
        else:
            profile_dict["hsc_p"] = hsc_p
            profile_dict["hsc_b"] = hsc_b
            profile_dict["hsc_s"] = hsc_s

        if mba_status == "Completed":
            profile_dict["mba_p"] = mba_p
            profile_dict["specialisation"] = specialisation

        validation = validate_student_profile(profile_dict)

        if not validation["is_valid"]:
            st.error(f"Input Validation Error: {'; '.join(validation['errors'])}")
        elif validation["is_model_compatible"]:
            # Valid 12th/HSC + Completed MBA: compatible with trained model
            try:
                # 1. Run inference
                pred_output = predict_single_student(model_bundle, validation["clean_model_input"])
                pred_label = pred_output["label"]
                prob_placed = pred_output["probability_placed"]
                prob_not_placed = pred_output["probability_not_placed"]

                # 2. Extract linear breakdown
                explanation = explain_single_prediction(model_bundle, validation["clean_model_input"])

                st.markdown("---")
                st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

                res_left, res_right = st.columns([1.1, 0.9])

                with res_left:
                    if pred_label == "Placed":
                        st.markdown(
                            f"""
                            <div class="result-placed">
                                <div class="result-badge-placed">✅ PREDICTION: PLACED (CLASS 1)</div>
                                <div class="result-score-title">Placement Likelihood</div>
                                <div class="result-score-value">{prob_placed*100:.1f}%</div>
                                <div class="result-sub-metric">
                                    Non-Placement Probability: <b>{prob_not_placed*100:.1f}%</b> | Decision Cutoff: <b>50.0%</b>
                                </div>
                                <div class="result-desc">
                                    <b>Verdict:</b> The student's academic profile and background characteristics are statistically 
                                    consistent with successful campus placement based on historical training patterns.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="result-not-placed">
                                <div class="result-badge-not-placed">⚠️ PREDICTION: AT-RISK / NOT PLACED (CLASS 0)</div>
                                <div class="result-score-title">Placement Likelihood</div>
                                <div class="result-score-value" style="color: #991B1B;">{prob_placed*100:.1f}%</div>
                                <div class="result-sub-metric">
                                    Non-Placement Probability: <b>{prob_not_placed*100:.1f}%</b> | Decision Cutoff: <b>50.0%</b>
                                </div>
                                <div class="result-desc">
                                    <b>Verdict:</b> The student's estimated score falls below the 50.0% cutoff. 
                                    Early academic advising, aptitude coaching, and resume mentorship are strongly recommended.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with res_right:
                    st.markdown("#### Estimated Probability Score")
                    st.progress(prob_placed)
                    c_pr1, c_pr2 = st.columns(2)
                    with c_pr1:
                        st.metric("Placement Likelihood", f"{prob_placed*100:.1f}%")
                    with c_pr2:
                        st.metric("Non-Placement Risk", f"{prob_not_placed*100:.1f}%")

                    st.markdown(
                        """
                        <div class="notice-box" style="margin-top: 8px; font-size: 0.88rem;">
                            <b>Model Architecture:</b> Logistic Regression with $L_2$ Regularization ($C=1.0$, `lbfgs`).<br>
                            <b>Responsible Use Disclaimer:</b> This estimated probability is a mathematical decision-support indicator, 
                            <b>not a contractual guarantee of employment</b>.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # -------------------------------------------------------------
                # PHASE 3 & 4: ESSENTIAL EDUCATIONAL FEATURE - EXPLAIN PREDICTION
                # -------------------------------------------------------------
                st.markdown("---")
                with st.expander("📘 Explain This Prediction (Educational & Viva Analysis)", expanded=True):
                    st.markdown("### How the Model Arrived at This Prediction")

                    # Phase 4: Visual Model Pipeline Flow Diagram
                    st.markdown("#### Pipeline Execution Flow")
                    f_c1, f_a1, f_c2, f_a2, f_c3, f_a3, f_c4, f_a4, f_c5 = st.columns([2, 0.4, 2, 0.4, 2, 0.4, 2, 0.4, 2])
                    with f_c1:
                        st.markdown('<div class="flow-step">1. Entered Profile<br><small>12 Features</small></div>', unsafe_allow_html=True)
                    with f_a1:
                        st.markdown('<div class="flow-arrow">➔</div>', unsafe_allow_html=True)
                    with f_c2:
                        st.markdown('<div class="flow-step">2. Preprocessing<br><small>Scale & One-Hot (14 dim)</small></div>', unsafe_allow_html=True)
                    with f_a2:
                        st.markdown('<div class="flow-arrow">➔</div>', unsafe_allow_html=True)
                    with f_c3:
                        st.markdown(f'<div class="flow-step">3. Log-Odds Score<br><small>z = {explanation["z_score"]:+.2f}</small></div>', unsafe_allow_html=True)
                    with f_a3:
                        st.markdown('<div class="flow-arrow">➔</div>', unsafe_allow_html=True)
                    with f_c4:
                        st.markdown(f'<div class="flow-step">4. Sigmoid Function<br><small>P = {prob_placed*100:.1f}%</small></div>', unsafe_allow_html=True)
                    with f_a4:
                        st.markdown('<div class="flow-arrow">➔</div>', unsafe_allow_html=True)
                    with f_c5:
                        st.markdown(f'<div class="flow-step">5. Decision Rule<br><small>P ≥ 0.50 ➔ <b>{pred_label}</b></small></div>', unsafe_allow_html=True)

                    st.markdown("---")

                    # Subsection A: Which algorithm was used?
                    st.markdown("#### A. Which Algorithm Was Used?")
                    st.markdown(
                        """
                        **Logistic Regression** is a supervised machine-learning algorithm used for classification. 
                        This project uses it to classify a student as **Placed** or **Not Placed**. 
                        Although its name contains *regression*, it is universally used for binary classification by mapping a continuous linear score into a probability between 0 and 1.
                        
                        The model was trained with **$L_2$ regularization** (Ridge penalty with regularization parameter $C = 1.0$), 
                        which penalizes overly large weight coefficients to prevent overfitting on our 215-record dataset.
                        """
                    )

                    # Subsection B: How is the output calculated?
                    st.markdown("#### B. How Is the Output Calculated?")
                    st.markdown(
                        f"""
                        The inference pipeline follows six explicit mathematical stages:
                        1. **Input Normalization:** Continuous marks (10th, 12th, Degree, E-Test, MBA) are standardized using the training set means and standard deviations:
                           $$z_{{\\text{{scaled}}}} = \\frac{{x - \\mu}}{{\\sigma}}$$
                        2. **Categorical Encoding:** Categorical inputs are one-hot encoded with reference category removal (`drop='first'`) matching the training design.
                        3. **Weighted Linear Score ($z$):** The model calculates a weighted linear combination of the 14 encoded features and adds the base intercept ($b = +{explanation['intercept']:.4f}$):
                           $$z = b + \\sum_{{i=1}}^{{14}} w_i x_i = {explanation['z_score']:+.4f}$$
                        4. **Sigmoid Probability Transformation:** The log-odds score $z$ is passed through the standard logistic sigmoid function $\\sigma(z)$:
                           $$P(\\text{{Placed}} = 1) = \\frac{{1}}{{1 + e^{{-z}}}} = \\frac{{1}}{{1 + e^{{-({explanation['z_score']:+.4f})}}}} = {prob_placed*100:.2f}\\%$$
                        5. **Threshold Decision:** With the default decision threshold of **0.50**:
                           $$\\hat{{y}} = \\begin{{cases}} \\text{{Placed}}, & \\text{{if }} P \\ge 0.50 \\\\ \\text{{Not Placed}}, & \\text{{if }} P < 0.50 \\end{{cases}}$$
                           Because calculated probability **P = {prob_placed*100:.1f}%** is **{'≥' if prob_placed >= 0.50 else '<'} 0.50**, the final prediction is **{pred_label}**.
                        """
                    )

                    # Subsection C: Why was Logistic Regression selected?
                    st.markdown("#### C. Why Was Logistic Regression Selected Over Other Candidates?")
                    st.markdown(
                        r"""
                        In our pre-implementation benchmarking experiment ([`compare_models.py`](file:///c:/Users/Asus/DS-mini-project/compare_models.py)), 
                        candidate algorithms were evaluated under identical 5-fold stratified cross-validation on the training set and verified on the untouched holdout test set:
                        
                        - **Holdout Accuracy:** Logistic Regression achieved **$86.05\%$** ($37/43$ correct), outperforming Random Forest ($83.72\%$) and Decision Trees ($67.44\% - 69.77\%$).
                        - **Holdout Precision:** Logistic Regression attained **$92.86\%$** precision, producing only **2 False Positives** (versus 5 for Random Forest).
                        - **Overfitting Resistance:** Single Decision Trees overfitted severely on this dataset size, experiencing an accuracy drop of over $10\%$ from CV to test. Logistic Regression maintained a negligible generalization gap of **$-0.60\%$**.
                        - **Model Interpretability:** Logistic Regression yields direct odds ratios ($\exp(\beta)$) and transparent linear weights, providing an ideal academic framework for thesis defense.
                        """
                    )

                    # Subsection D: What do the probability and prediction mean?
                    st.markdown("#### D. What Do the Probability and Prediction Mean?")
                    st.markdown(
                        f"""
                        - **The Class Label ({pred_label}):** Reflects the discrete binary decision derived from applying the 0.50 probability cutoff.
                        - **The Probability Score ({prob_placed*100:.1f}%):** Reflects the mathematical likelihood that a student with this feature profile shares the placement outcome of historical students in `Placement.csv`.
                        - **Error Margins:** The model is not infallible. On the held-out test set, it had 2 False Positives (predicted Placed, actually unplaced) and 4 False Negatives (predicted Not Placed, actually placed).
                        - **Cohort Specificity:** These findings are derived from a historical dataset of 215 students and may not generalize unconditionally to institutions with differing curricula or industry partnerships.
                        """
                    )

                    # Subsection E: What factors influenced this prediction?
                    st.markdown("#### E. Exact Feature Contributions for This Student")
                    st.markdown("In a linear model, each feature's contribution to the log-odds score $z$ is exactly $w_i \\cdot x_{i,\\text{scaled}}$.")

                    contrib_display = explanation["contributions_df"].copy()
                    contrib_display["Candidate Attribute"] = contrib_display["feature"].map(lambda f: FEATURE_NAME_MAP.get(f, f))
                    contrib_display = contrib_display.sort_values(by="contribution", ascending=False)
                    contrib_display = contrib_display[[
                        "Candidate Attribute",
                        "feature",
                        "weight",
                        "transformed_value",
                        "contribution",
                        "direction"
                    ]]
                    contrib_display.columns = [
                        "Candidate Attribute",
                        "Pipeline Feature (Encoded)",
                        "Learned Weight (w)",
                        "Scaled Value (x)",
                        "Linear Contribution (w · x)",
                        "Impact Direction"
                    ]
                    st.dataframe(
                        contrib_display.style.format({
                            "Learned Weight (w)": "{:+.4f}",
                            "Scaled Value (x)": "{:+.4f}",
                            "Linear Contribution (w · x)": "{:+.4f}",
                        }),
                        width="stretch"
                    )

                    st.markdown(
                        """
                        <div class="notice-box">
                            <b>Interpretation Note:</b> Positive contributions push the log-odds score $z$ higher (increasing placement probability), 
                            while negative contributions pull $z$ lower. 
                            These weights describe <b>statistical relationships</b> in the historical training data and do not constitute personal causation.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Subsection F: Viva Talking Points
                    st.markdown("#### F. Viva Talking Points: How to Explain This Result to Your Teacher")
                    st.markdown(
                        f"""
                        <div class="viva-box">
                            <b>1. Final Prediction:</b> "The model predicted <b>{pred_label}</b> because the calculated sigmoid probability is <b>{prob_placed*100:.1f}%</b>, which is {'above' if prob_placed >= 0.50 else 'below'} the standard 0.50 classification cutoff."
                        </div>
                        <div class="viva-box">
                            <b>2. Algorithm Justification:</b> "We selected <b>Logistic Regression with L2 Regularization (C=1.0)</b>. Despite containing 'regression' in its title, it is a linear classifier that computes log-odds and passes them through a sigmoid function."
                        </div>
                        <div class="viva-box">
                            <b>3. Leakage-Free Preprocessing:</b> "Continuous variables are normalized via <code>StandardScaler</code> and categorical variables are encoded with reference-category <code>OneHotEncoder</code> strictly inside a scikit-learn <code>Pipeline</code>, guaranteeing zero data leakage between training and testing folds."
                        </div>
                        <div class="viva-box">
                            <b>4. Performance vs. Ensembles:</b> "In our empirical benchmarking, Logistic Regression attained <b>86.05% test accuracy</b> and <b>92.86% precision</b> with only 2 False Positives. Decision Trees overfitted heavily (dropping to 67.4% test accuracy), while Random Forest had higher false positives (5)."
                        </div>
                        <div class="viva-box">
                            <b>5. Minimizing Asymmetric Cost:</b> "In campus recruitment, False Positives (predicting placed when a student fails) carry the highest institutional harm because struggling students are deprived of remedial workshops. Our model explicitly optimized for high placement precision."
                        </div>
                        <div class="viva-box">
                            <b>6. Academic Safety & Scope:</b> "Identifiers (<code>sl_no</code>) and post-placement salary (<code>salary</code>) were removed to avoid 100% target leakage. Furthermore, diploma students are safely separated because `Placement.csv` contains zero polytechnic entrants, preventing invalid zero-imputation."
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as err:
                st.error(f"Inference Pipeline Error: {err}")

        else:
            # Diploma route or incomplete MBA: Model is incompatible
            st.markdown("---")
            st.markdown(
                """
                <div class="warning-box">
                    <h3 style="color: #92400E; margin-top: 0; font-size: 1.4rem;">⚠️ Automated Prediction Inapplicable for This Pathway</h3>
                    <p style="font-size: 1rem; margin-bottom: 6px;">
                        <b>Automated placement prediction is intentionally withheld to prevent unvalidated, biased, or misleading inferences.</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Why Prediction Is Withheld (Academic Integrity Safeguards):")
            st.markdown(
                """
                1. **Historical Dataset Distribution:** The existing Logistic Regression model was trained strictly on `Placement.csv` ($N=215$), 
                   where **100% of candidates completed the standard 10th → 12th (HSC) → Degree → MBA pathway**.
                2. **Zero Polytechnic Entrants:** There are **zero diploma-route candidates** in the historical training dataset. 
                   The model has no learned mathematical weights connecting diploma specialization to campus recruitment success.
                3. **Zero Arbitrary Imputation Policy:** In accordance with academic data science standards, the system **refuses to silently replace 
                   missing HSC marks or MBA credentials with zeros, cohort means, or diploma scores**. Such synthetic imputation would distort 
                   the linear log-odds equation and yield invalid predictions.
                4. **No False Accuracy Claims:** The model makes **no claim of accuracy** for diploma students or pre-MBA undergraduates 
                   because it has never been trained or validated on appropriate out-of-domain data.
                """
            )

            st.markdown("#### Candidate Profile Summary (Recorded Safely):")
            summary_items = [
                ("Secondary Education Pathway", education_route),
                ("10th Grade Percentage (SSC %)", f"{ssc_p:.1f}%"),
            ]
            if education_route == "Diploma route":
                summary_items.append(("Diploma Aggregate Percentage", f"{diploma_p:.1f}%"))
                summary_items.append(("12th Grade (HSC %)", "Not applicable (Diploma Route)"))
                summary_items.append(("HSC Stream & Board", "Not applicable (Diploma Route)"))
            else:
                summary_items.extend([
                    ("12th Grade Percentage (HSC %)", f"{hsc_p:.1f}%"),
                    ("HSC Stream", hsc_s),
                    ("HSC Board", hsc_b),
                ])

            summary_items.extend([
                ("Undergraduate Degree", f"{degree_p:.1f}% ({degree_t})"),
                ("Employability Aptitude Test", f"{etest_p:.1f}%"),
                ("Candidate Gender", gender),
                ("Prior Industry Experience", workex),
                ("MBA Program Status", mba_status),
            ])
            if mba_status == "Completed":
                summary_items.extend([
                    ("MBA Percentage", f"{mba_p:.1f}%"),
                    ("MBA Specialization Track", specialisation),
                ])
            else:
                summary_items.extend([
                    ("MBA Percentage", f"Not applicable ({mba_status})"),
                    ("MBA Specialization Track", f"Not applicable ({mba_status})"),
                ])

            df_summary = pd.DataFrame(summary_items, columns=["Candidate Attribute", "Entered Record"])
            st.table(df_summary)

            st.markdown(
                """
                <div class="notice-box">
                    <b>Institutional Recommendation:</b> To evaluate diploma-route and bachelor's-only candidates fairly, 
                    the institution should gather dedicated recruitment records for polytechnic entrants and train an independent, 
                    appropriately validated multi-pathway classification pipeline.
                </div>
                """,
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------
# PAGE 4: MODEL EVALUATION & BENCHMARKS
# -----------------------------------------------------------------------------
elif page == "📈 Model Evaluation":
    st.markdown('<div class="main-header">Model Evaluation & Benchmark Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous performance metrics evaluated via 5-Fold Stratified Cross-Validation on the training set and an untouched Holdout Test Set.</div>', unsafe_allow_html=True)

    t_res = model_bundle["test_results"]
    cv_res = model_bundle["cv_results"]

    # Top KPI Cards
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric("Test Accuracy", f"{t_res['accuracy']*100:.2f}%", "37 / 43 correct")
    with col_m2:
        st.metric("Test Precision", f"{t_res['precision']*100:.2f}%", "26 / 28 placed")
    with col_m3:
        st.metric("Test Recall", f"{t_res['recall']*100:.2f}%", "26 / 30 placed")
    with col_m4:
        st.metric("Test F1-Score", f"{t_res['f1']*100:.2f}%", "Macro F1: 84.1%")
    with col_m5:
        st.metric("Train CV Accuracy", f"{cv_res['cv_accuracy_mean']*100:.2f}%", f"+/- {cv_res['cv_accuracy_std']*100:.2f}%")

    st.markdown("---")

    col_cm, col_gap = st.columns([1, 1])

    with col_cm:
        st.markdown('<div class="section-title">Confusion Matrix (Held-out Test Set, N=43)</div>', unsafe_allow_html=True)
        cm_matrix = np.array([[t_res['tn'], t_res['fp']], [t_res['fn'], t_res['tp']]])
        fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=130)
        sns.heatmap(
            cm_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Not Placed (0)", "Predicted Placed (1)"],
            yticklabels=["Actual Not Placed (0)", "Actual Placed (1)"],
            cbar=False,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"}
        )
        ax.set_title("Test Confusion Matrix (Logistic Regression, N=43)", fontsize=11, fontweight="bold", pad=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            f"**True Negatives (TN):** {t_res['tn']} | **False Positives (FP):** {t_res['fp']} | "
            f"**False Negatives (FN):** {t_res['fn']} | **True Positives (TP):** {t_res['tp']}"
        )

    with col_gap:
        st.markdown('<div class="section-title">Generalization & Split Analysis</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            - **Training Partition (80%):** 172 students ($118$ Placed, $54$ Not Placed)
            - **Held-out Test Partition (20%):** 43 students ($30$ Placed, $13$ Not Placed)
            - **Stratified 5-Fold CV Accuracy (Train):** `{cv_res['cv_accuracy_mean']*100:.2f}% +/- {cv_res['cv_accuracy_std']*100:.2f}%`
            - **Stratified 5-Fold CV F1-Score (Train):** `{cv_res['cv_f1_mean']*100:.2f}% +/- {cv_res['cv_f1_std']*100:.2f}%`
            - **Holdout Test Accuracy:** `{t_res['accuracy']*100:.2f}%`
            - **Generalization Gap (CV - Test):** `{(cv_res['cv_accuracy_mean'] - t_res['accuracy'])*100:+.2f}%`
            
            <div class="notice-box">
                <b>Generalization Verdict:</b> The gap between training cross-validation (85.45%) and holdout test accuracy (86.05%) 
                is only <b>0.60%</b>. This demonstrates exceptional generalization stability without data leakage or variance collapse.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Algorithm Comparison Table & Chart
    st.markdown('<div class="section-title">Candidate Algorithm Benchmark Comparison</div>', unsafe_allow_html=True)
    st.markdown("Fair comparison of Logistic Regression, Random Forest, and Decision Trees trained on identical partitions using leakage-safe pipelines.")

    benchmark_df = get_cached_benchmark()
    st.dataframe(benchmark_df, width="stretch")

    # Visual Bar Comparison of Test Metrics
    fig_bench, ax_b = plt.subplots(figsize=(8.5, 4.0), dpi=130)
    plot_data = benchmark_df.copy()
    plot_data["Test_Acc_Num"] = plot_data["Test Accuracy"].str.rstrip("%").astype(float)
    plot_data["Test_Prec_Num"] = plot_data["Test Precision"].str.rstrip("%").astype(float)
    plot_data["Test_F1_Num"] = plot_data["Test F1-Score"].str.rstrip("%").astype(float)

    x_inds = np.arange(len(plot_data))
    bar_w = 0.25
    b1 = ax_b.bar(x_inds - bar_w, plot_data["Test_Acc_Num"], width=bar_w, label="Test Accuracy (%)", color="#2563EB")
    b2 = ax_b.bar(x_inds, plot_data["Test_Prec_Num"], width=bar_w, label="Test Precision (%)", color="#059669")
    b3 = ax_b.bar(x_inds + bar_w, plot_data["Test_F1_Num"], width=bar_w, label="Test F1-Score (%)", color="#D97706")

    # Add data labels on top of each bar for crystal-clear readability
    for bar_group in [b1, b2, b3]:
        for bar in bar_group:
            h = bar.get_height()
            if h > 0:
                ax_b.annotate(f"{h:.1f}%",
                             (bar.get_x() + bar.get_width() / 2.0, h + 1.2),
                             ha="center", va="bottom", fontsize=7.5, color="#1E293B", fontweight="bold")

    ax_b.set_ylabel("Metric Score (%)", fontsize=10)
    ax_b.set_title("Holdout Test Metric Comparison Across Candidate Models", fontsize=11, fontweight="bold")
    ax_b.set_xticks(x_inds)
    ax_b.set_xticklabels(plot_data["Algorithm"], rotation=12, ha="right", fontsize=8.5)
    ax_b.set_ylim(45, 110)
    ax_b.grid(axis="y", linestyle="--", alpha=0.4)
    ax_b.legend(loc="upper right", framealpha=0.9)
    fig_bench.tight_layout()
    st.pyplot(fig_bench)
    plt.close(fig_bench)

    st.markdown("---")

    # Feature Importance (Logistic Regression Coefficients)
    st.markdown('<div class="section-title">Fitted Model Coefficients & Odds Ratios</div>', unsafe_allow_html=True)
    st.markdown("In Logistic Regression, the exponential of the coefficient $\\exp(\\beta)$ represents the **Odds Ratio**: the multiplicative change in placement odds per one standard deviation increase in feature value.")

    coef_df = pd.DataFrame(model_bundle["coefficients"]).sort_values(by="coefficient", ascending=False)
    coef_df["Candidate Attribute"] = coef_df["feature"].map(lambda f: FEATURE_NAME_MAP.get(f, f))
    coef_df["Impact"] = coef_df["coefficient"].apply(lambda x: "Increases Placement Odds" if x > 0 else "Decreases Placement Odds")
    
    coef_display = coef_df[[
        "Candidate Attribute",
        "feature",
        "coefficient",
        "odds_ratio",
        "Impact"
    ]].rename(columns={
        "feature": "Pipeline Feature (Encoded)",
        "coefficient": "Log-Odds Coef (B)",
        "odds_ratio": "Odds Ratio [exp(B)]",
        "Impact": "Statistical Impact Direction",
    })
    
    st.dataframe(
        coef_display.style.format({"Log-Odds Coef (B)": "{:+.4f}", "Odds Ratio [exp(B)]": "{:.3f}"}),
        width="stretch"
    )

# -----------------------------------------------------------------------------
# PAGE 5: ABOUT, METHODOLOGY & ACADEMIC LOGBOOK
# -----------------------------------------------------------------------------
elif page == "📖 About & Methodology":
    st.markdown('<div class="main-header">About the Project & Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Formal academic documentation, mathematical framework, validation protocol, and ethical guidelines for DS Lab evaluation.</div>', unsafe_allow_html=True)

    tab_doc1, tab_doc2, tab_doc3 = st.tabs([
        "📐 Problem Statement & Algorithm",
        "⚠️ Limitations & Ethical Guidelines",
        "📝 Academic Submission Details"
    ])

    with tab_doc1:
        st.markdown('<div class="section-title">1. Problem Statement & Motivation</div>', unsafe_allow_html=True)
        st.markdown(
            """
            In higher education institutions, campus placement is a vital benchmark of institutional quality and student career progression. 
            However, student readiness is traditionally assessed late in the final semester, leaving little time for remedial intervention.
            
            This mini-project develops an **interpretable binary classification pipeline** to predict whether an MBA student is likely to be placed 
            using their prior academic history (10th, 12th, Degree, Employability Test, MBA percentages) and professional background (work experience, specialization track).
            """
        )

        st.markdown('<div class="section-title">2. Mathematical Formulation of Logistic Regression</div>', unsafe_allow_html=True)
        st.markdown(
            r"""
            Logistic Regression models the posterior class probability via the logistic sigmoid link function:
            
            $$P(Y = 1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
            
            where the linear log-odds score $z$ is:
            
            $$z = b + \mathbf{w}^T \mathbf{x} = b + w_1 x_1 + w_2 x_2 + \dots + w_{14} x_{14}$$
            
            **Regularized Objective Function ($L_2$ Ridge Penalty):**
            
            $$\min_{\mathbf{w}, b} \left( -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \sigma(z_i) + (1 - y_i) \ln(1 - \sigma(z_i)) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2 \right)$$
            
            Where:
            - $C = 1.0$ controls regularization strength (preventing extreme weights).
            - $\mathbf{x}$ consists of 5 standardized numerical features and 9 reference-coded dummy categories.
            - The loss is minimized using the deterministic `lbfgs` quasi-Newton optimization routine.
            """
        )

    with tab_doc2:
        st.markdown('<div class="section-title">Academic Limitations & Ethical Use</div>', unsafe_allow_html=True)
        st.markdown(
            r"""
            <div class="warning-box">
                <b>Known Model Limitations:</b>
                <ol>
                    <li><b>Sample Size ($N=215$):</b> While sufficient for linear models with L2 shrinkage, 215 students is a relatively modest sample. In the 20% holdout split ($N=43$), each individual student represents $2.33\%$ of test accuracy.</li>
                    <li><b>Omitted Behavioral Dimensions:</b> The dataset records academic marks but lacks qualitative factors such as oral communication skills, coding proficiency, problem-solving, and resume strength.</li>
                    <li><b>Historical Hiring Biases:</b> If historical hiring favored specific academic streams or specializations, the model reflects those patterns rather than innate student capability.</li>
                    <li><b>Cohort Specificity:</b> The data represents a single postgraduate management cohort and must not be applied to other degree programs without retraining.</li>
                </ol>
            </div>
            
            <div class="notice-box">
                <b>Responsible Use Policy:</b> This system is designed solely as a <b>proactive decision-support tool</b> for student development centers 
                to offer early resume clinics and interview coaching. 
                It must <b>NEVER</b> be used as an automated filter to deny students registration for campus placement drives.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_doc3:
        st.markdown('<div class="section-title">DS Lab Mini-Project Submission Log</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - **Project Title:** Student Placement Prediction System Using Machine Learning
            - **Repository:** `https://github.com/Chinmay4353/DS-mini-project`
            - **Author:** Chinmay Nagap
            - **Primary Technology:** Python, Streamlit, Scikit-learn, Pandas, Joblib
            - **Architecture:** Decoupled `src/data.py` (schemas & validation) and `src/model.py` (pipelines & inference)
            - **Experiment Script:** `compare_models.py` (Reproducible 5-fold CV & test benchmarking)
            - **Serialized Artifact:** `models/placement_model.joblib`
            - **Submission Deadline:** 23 October 2026
            """
        )
