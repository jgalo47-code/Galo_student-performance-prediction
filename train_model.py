# train_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---------- Part 1: Retrieve and inspect the dataset ----------
if os.path.exists("student_data.csv"):
    df = pd.read_csv("student_data.csv")                 # already in the repo
elif os.path.exists("student-mat.csv"):
    df = pd.read_csv("student-mat.csv", sep=";")         # manual UCI download
    df.to_csv("student_data.csv", index=False)
else:
    from ucimlrepo import fetch_ucirepo                  # download from UCI
    student = fetch_ucirepo(id=320)
    df = pd.concat([student.data.features, student.data.targets], axis=1)
    df.to_csv("student_data.csv", index=False)

print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())

# ---------- Part 2: Preprocess, train, evaluate ----------
df = df.drop_duplicates()
y = (df["G3"] < 10).astype(int)   # 1 = at risk (G3 < 10); 0 = not at risk
X = df.drop(columns=["G3"])       # never use G3 as a predictor

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0,
))

joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl")
