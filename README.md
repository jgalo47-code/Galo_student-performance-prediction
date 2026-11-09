# Student Performance Prediction System

A classroom prototype that flags students who may be at risk of failing final grades.

## Problem
Academic risk is often noticed only after poor results. This prototype explores early
identification using available student indicators (study time, absences, failures, G1, G2, etc.).

## Dataset
UCI Machine Learning Repository - Student Performance
https://archive.ics.uci.edu/dataset/320/student+performance
(`student_data.csv` in this repo: 649 records, 33 columns.)

## Prediction Target
`G3 < 10` = **AT RISK**; `G3 >= 10` = **NOT AT RISK**.
G3 is the target only and is never used as an input feature (avoids target leakage).

## Technologies
Python, pandas, scikit-learn (Random Forest), Streamlit, joblib, GitHub

## Installation and Run
```bash
pip install -r requirements.txt
python train_model.py      # trains the model and creates student_model.pkl
streamlit run app.py       # opens http://localhost:8501
```
Test the app by uploading `sample_prediction_input.csv`.
Upload `sample_missing_column.csv` to see the missing-column error message.
If `student_model.pkl` is missing, the app trains it automatically on first run.

## Limitations
- Trained on one Portuguese secondary-school dataset; may not generalize to other institutions.
- Predictions are not diagnoses and must not be used as the sole basis for real student decisions.
