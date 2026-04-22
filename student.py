# ================= IMPORT LIBRARIES =================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix

from mlxtend.frequent_patterns import apriori, association_rules

# ================= LOAD DATA =================
df = pd.read_csv("placement.csv")

# ================= DATA PREPROCESSING =================
df.drop(["sl_no", "salary"], axis=1, inplace=True)
df.dropna(inplace=True)

# Convert categorical → numeric
df = pd.get_dummies(df, drop_first=True)

# ================= CLASSIFICATION =================
print("\n===== CLASSIFICATION (PLACEMENT PREDICTION) =====")

X = df.drop("status_Placed", axis=1)
y = df["status_Placed"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ================= CLUSTERING =================
print("\n===== CLUSTERING (STUDENT GROUPING) =====")

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaler.transform(X))

print("Cluster Counts:\n", df["Cluster"].value_counts())

# Visualization
plt.scatter(df["degree_p"], df["etest_p"], c=df["Cluster"])
plt.xlabel("Degree %")
plt.ylabel("Aptitude Score")
plt.title("Student Clusters")
plt.show()

# ================= ASSOCIATION RULE =================
print("\n===== ASSOCIATION RULE MINING =====")

df_assoc = df.copy()

# Convert to boolean (High / Low)
df_assoc = df_assoc.apply(lambda x: x > x.mean())

frequent = apriori(df_assoc, min_support=0.3, use_colnames=True)
rules = association_rules(frequent, metric="confidence", min_threshold=0.6)

# 🔥 FILTER ONLY PLACEMENT RULES
placement_rules = rules[rules['consequents'].apply(lambda x: 'status_Placed' in str(x))]

# Sort by confidence
placement_rules = placement_rules.sort_values(by='confidence', ascending=False)

print("\n===== PLACEMENT RULES (IMPORTANT) =====")
print(placement_rules[['antecedents', 'consequents', 'support', 'confidence']])

# ================= VISUALIZATION =================

sns.countplot(x=y)
plt.title("Placement Distribution")
plt.show()

sns.boxplot(x=y, y=df["degree_p"])
plt.title("Degree % vs Placement")
plt.show()

# ================= USER INPUT PREDICTION =================

print("\n===== STUDENT INPUT PREDICTION =====")

def predict_student():
    print("\nEnter student details:")

    # Numeric inputs
    ssc_p = float(input("Enter SSC %: "))
    hsc_p = float(input("Enter HSC %: "))
    degree_p = float(input("Enter Degree %: "))
    etest_p = float(input("Enter Aptitude Score: "))
    mba_p = float(input("Enter MBA %: "))

    # Categorical inputs
    gender = input("Enter Gender (M/F): ").strip().upper()
    workex = input("Work Experience? (Yes/No): ").strip().lower()
    specialization = input("Specialization (Mkt&HR / Mkt&Fin / None): ").strip()

    # Create base dataframe
    input_data = pd.DataFrame([[ssc_p, hsc_p, degree_p, etest_p, mba_p]],
                             columns=["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"])

    # Encode categorical values
    input_data["gender_M"] = 1 if gender == "M" else 0
    input_data["workex_Yes"] = 1 if workex == "yes" else 0

    # Handle specialization safely
    if specialization.lower() == "mkt&hr":
        input_data["specialisation_Mkt&HR"] = 1
    else:
        input_data["specialisation_Mkt&HR"] = 0  # default

    # Add missing columns
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Correct column order
    input_data = input_data[X.columns]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    result = model.predict(input_scaled)

    if result[0] == 1:
        print("\n✅ Prediction: Student WILL be Placed")
    else:
        print("\n❌ Prediction: Student will NOT be Placed")

# Call function
predict_student()