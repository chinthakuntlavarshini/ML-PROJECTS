# AI RESUME SCREENING - COMPLETE ML PROJECT
# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
# 2. LOAD DATASET
df = pd.read_csv("AI_Resume_Screening.csv")
print(df["Recruiter Decision"].value_counts(dropna=False))

print("\n==============================")
print("FIRST 5 ROWS")
print("==============================")

print(df.head())

print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print(df.info())

print("\n==============================")
print("MISSING VALUES")
print("==============================")

print(df.isnull().sum())

# 3. REMOVE DUPLICATES
df = df.drop_duplicates()
print("\nDuplicates removed.")
print("Dataset shape:", df.shape)

# 4. DEFINE TEXT AND NUMERIC COLUMNS
text_columns = [
    "Skills",
    "Education",
    "Certifications",
    "Job Role"
]
numeric_columns = [
    "Experience (Years)",
    "Salary Expectation ($)",
    "Projects Count",
    "AI Score (0-100)"
]

# 5. CLEAN TEXT COLUMNS
for col in text_columns:

    df[col] = df[col].fillna("Unknown")

    df[col] = df[col].astype(str)

    df[col] = df[col].str.lower().str.strip()

# 6. CLEAN NUMERIC COLUMNS
for col in numeric_columns:

    # Convert text/object values into numbers
    # Invalid values become NaN
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# 7. FILL NUMERIC MISSING VALUES
# Experience
df["Experience (Years)"] = df[
    "Experience (Years)"
].fillna(0)


# Salary
salary_median = df[
    "Salary Expectation ($)"
].median()

df["Salary Expectation ($)"] = df[
    "Salary Expectation ($)"
].fillna(salary_median)


# Projects
df["Projects Count"] = df[
    "Projects Count"
].fillna(0)


# AI Score
ai_score_median = df[
    "AI Score (0-100)"
].median()

df["AI Score (0-100)"] = df[
    "AI Score (0-100)"
].fillna(ai_score_median)

# 8. CHECK DATA TYPES

print("\n==============================")
print("DATA TYPES AFTER CLEANING")
print("==============================")

print(df[numeric_columns].dtypes)

# 9. CREATE RESUME  TEXT

df["resume_text"] = (
    df["Skills"] + " " +
    df["Education"] + " " +
    df["Certifications"] + " " +
    df["Job Role"]
)


print("\n==============================")
print("RESUME TEXT")
print("==============================")

print(df["resume_text"].head())

# 10. CONVERT TEXT INTO TF-IDF


tfidf = TfidfVectorizer(
    max_features=100,
    stop_words="english"
)

X_text = tfidf.fit_transform(
    df["resume_text"]
)


print("\nTF-IDF shape:", X_text.shape)

# 11. CLEAN TARGET COLUMN


df["Recruiter Decision"] = (
    df["Recruiter Decision"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)



# 12. CONVERT TARGET VALUES

decision_mapping = {

    "select": 1,
    "selected": 1,
    "accept": 1,
    "accepted": 1,
    "hire": 1,
    "hired": 1,
    "yes": 1,
    "shortlisted": 1,

    "reject": 0,
    "rejected": 0,
    "decline": 0,
    "declined": 0,
    "no": 0,
    "not selected": 0,
    "notselected": 0
}

df["target"] = df["Recruiter Decision"].map(
    decision_mapping
)



# 13. REMOVE UNKNOWN TARGETS
df = df.dropna(
    subset=["target"]
).copy()

df["target"] = df["target"].astype(int)

# 14. CHECK TARGET

print("\n==============================")
print("TARGET DISTRIBUTION")
print("==============================")

print(df["target"].value_counts())

print(
    "Select:",
    (df["target"] == 1).sum()
)

print(
    "Reject:",
    (df["target"] == 0).sum()
)

# 15. CHECK BOTH CLASSES

if df["target"].nunique() < 2:

    print("\nERROR: Only one class found.")

    print(
        "\nYour Recruiter Decision column "
        "must contain BOTH:"
    )

    print("Select")
    print("Reject")

    raise ValueError(
        "Dataset contains only one class. "
        "Check the Recruiter Decision column."
    )


# 13. REMOVE UNKNOWN TARGET ROWS


df = df.dropna(
    subset=["target"]
)

df["target"] = df[
    "target"
].astype(int)


print("\n==============================")
print("TARGET DISTRIBUTION")
print("==============================")

print(
    df["target"].value_counts()
)


# IMPORTANT:
# Because rows may have been removed above,
# create TF-IDF again using the cleaned dataset.

df["resume_text"] = (
    df["Skills"] + " " +
    df["Education"] + " " +
    df["Certifications"] + " " +
    df["Job Role"]
)


tfidf = TfidfVectorizer(
    max_features=100,
    stop_words="english"
)

X_text = tfidf.fit_transform(
    df["resume_text"]
)



# 14. CREATE NUMERIC FEATURES


numeric_features = df[
    numeric_columns
].astype(
    np.float64
).values


print("\nNumeric feature dtype:")
print(numeric_features.dtype)



# 15. CONVERT NUMERIC FEATURES
# TO SPARSE MATRIX

numeric_sparse = csr_matrix(
    numeric_features,
    dtype=np.float64
)

# 16. COMBINE TF-IDF + NUMERIC

X = hstack([
    X_text,
    numeric_sparse
]).tocsr()


y = df["target"].values


print("\n==============================")
print("FINAL FEATURE MATRIX")
print("==============================")

print("X shape:", X.shape)
print("X dtype:", X.dtype)
print("y shape:", y.shape)
#check target classes
print(df["target"].value_counts())

if df["target"].nunique() < 2:
    raise ValueError(
        "Dataset contains only one class. "
        "You need both 'select' and 'reject' values "
        "in the Recruiter Decision column."
    )


# 17. TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])



# 18. CREATE RANDOM FOREST MODEL


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)



# 19. TRAIN MODEL


print("\n==============================")
print("TRAINING MODEL")
print("==============================")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")



# 20. PREDICT TEST DATA


y_pred = model.predict(
    X_test
)



# 21. MODEL PERFORMANCE

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(
    "Accuracy  :",
    round(accuracy, 4)
)

print(
    "Precision :",
    round(precision, 4)
)

print(
    "Recall    :",
    round(recall, 4)
)

print(
    "F1 Score  :",
    round(f1, 4)
)


# 22. CLASSIFICATION REPORT


print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0,1],
        target_names=[
            "Reject",
            "Select"
        ],
        zero_division=0
    )
)


# 23. CONFUSION MATRIX


cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0,1]
)


plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "Reject",
        "Select"
    ],
    yticklabels=[
        "Reject",
        "Select"
    ]
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "Resume Screening Confusion Matrix"
)

plt.tight_layout()

plt.show()



# 24. PERFORMANCE BAR CHART


metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    metrics.keys(),
    metrics.values()
)

plt.ylim(
    0,
    1
)

plt.ylabel(
    "Score"
)

plt.title(
    "AI Resume Screening Model Performance"
)


for i, value in enumerate(
    metrics.values()
):

    plt.text(
        i,
        value + 0.02,
        f"{value:.2f}",
        ha="center"
    )


plt.tight_layout()

plt.show()



# 25. PERFORMANCE PIE CHART


plt.figure(
    figsize=(7, 7)
)

plt.pie(
    metrics.values(),
    labels=metrics.keys(),
    autopct="%1.1f%%",
    startangle=90
)

plt.title(
    "Resume Screening Model Performance"
)

plt.tight_layout()

plt.show()



# 26. PREDICT A NEW RESUME


new_resume = {

    "Skills":
        "python machine learning sql",

    "Experience (Years)":
        2,

    "Education":
        "btech",

    "Certifications":
        "aws",

    "Job Role":
        "data scientist",

    "Salary Expectation ($)":
        600000,

    "Projects Count":
        4,

    "AI Score (0-100)":
        88
}



# 27. CLEAN NEW RESUME TEXT


new_text = (
    str(new_resume["Skills"]).lower() + " " +
    str(new_resume["Education"]).lower() + " " +
    str(new_resume["Certifications"]).lower() + " " +
    str(new_resume["Job Role"]).lower()
)



# 28. TF-IDF TRANSFORMATION


new_text_tfidf = tfidf.transform(
    [new_text]
)

# 29. NEW RESUME NUMERIC FEATURES


new_numeric = np.array([[
    float(new_resume["Experience (Years)"]),
    float(new_resume["Salary Expectation ($)"]),
    float(new_resume["Projects Count"]),
    float(new_resume["AI Score (0-100)"])
]], dtype=np.float64)
# 30. CONVERT TO SPARSE MATRIX
new_numeric_sparse = csr_matrix(
    new_numeric,
    dtype=np.float64
)
# 31. COMBINE NEW RESUME FEATURES
new_X = hstack([
    new_text_tfidf,
    new_numeric_sparse
]).tocsr()

# 32. PREDICTION
prediction = model.predict(
    new_X
)
# 33. PREDICTION PROBABILITY

probability = model.predict_proba(
    new_X
)
# 34. DISPLAY RESULT

print("\n========================================")
print("          RESUME PREDICTION")
print("========================================")
if prediction[0] == 1:
    print(
        "Recruiter Decision : SELECT"
    )
    print(
        "Selection Probability :",
        round(
            probability[0][1] * 100,
            2
        ),
        "%"
    )
else:
    print(
        "Recruiter Decision : REJECT"
    )
    print(
        "Rejection Probability :",
        round(
            probability[0][0] * 100,
            2
        ),
        "%"
    )
print(" ")