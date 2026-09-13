# Student Performance ML

A beginner-friendly machine-learning project that predicts whether a student may be at risk of failing a Mathematics course.

This project is designed to practise a complete supervised machine-learning workflow: data exploration, preprocessing, train/test splitting, model training, evaluation, comparison, prediction, and model saving.

## Objective

Build a binary classification model that predicts whether a student is at risk of failing the final course grade.

The target is created from the final grade:

- `0` — Not at risk: final grade (`G3`) is 10 or above
- `1` — At risk: final grade (`G3`) is below 10

## Dataset

This project uses the **Student Performance** dataset from the UCI Machine Learning Repository.

- Dataset page: https://archive.ics.uci.edu/dataset/320/student+performance
- Initial file: `student-mat.csv`
- Subject: Mathematics
- Records: 395 students
- Original target: `G3`, the final grade from 0 to 20

The dataset includes demographic, family, educational, social, attendance, and study-related attributes.

> Note: `G1` and `G2` are first- and second-period grades and strongly correlate with the final grade `G3`. The primary early-risk experiment will exclude `G1`, `G2`, and `G3` from input features to reduce target leakage and make the prediction task more realistic.

## Planned workflow

1. Load and inspect the dataset.
2. Check data types, duplicates, missing values, and class distribution.
3. Create the binary `at_risk` target.
4. Perform exploratory data analysis and visualizations.
5. Encode categorical features and prepare numeric features.
6. Split the data into training and testing sets.
7. Train a Logistic Regression baseline model.
8. Train and compare a Random Forest Classifier.
9. Evaluate models using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.
10. Select the final model based on performance and interpretability.
11. Save the final pipeline using Joblib.
12. Demonstrate predictions on unseen sample data.

## Tech stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Project structure

```text
student-performance-ml/
├── data/          # Dataset files; raw data is excluded from Git
├── models/        # Saved model artifacts; excluded from Git
├── notebooks/     # EDA and model-training notebooks
├── src/           # Reusable Python modules
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/student-performance-ml.git
cd student-performance-ml
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install project dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

## Status

The first end-to-end notebook workflow is implemented in `notebooks/book.ipynb`.
It loads the Mathematics dataset, performs integrity checks and exploratory
analysis, creates the leakage-safe `at_risk` target, trains two classification
pipelines, compares their metrics, and saves the selected pipeline with Joblib.

The Portuguese dataset and merged-student analysis are reserved for a later
experiment.

## Initial results

Using a fixed, stratified 80/20 split and excluding `G1`, `G2`, and `G3` from
the model features, the initial test results were:

| Model | Accuracy | Precision (at risk) | Recall (at risk) | F1 (at risk) | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.696 | 0.571 | 0.308 | 0.400 | 0.715 |
| Random Forest | 0.709 | 0.571 | 0.462 | 0.511 | 0.692 |

Random Forest was selected for the initial saved pipeline because it produced
the stronger at-risk recall and F1 score, and also performed better in the
five-fold training cross-validation summary. These results are estimates from
a small dataset and should not be treated as evidence for real educational
decisions.

The generated artifact is saved locally as
`models/student_risk_pipeline.joblib` and is excluded from Git.

## Educational-use disclaimer

This is an educational machine-learning project using a limited public dataset. It must not be used to make real decisions about student admission, grading, placement, discipline, financial aid, or access to educational opportunities. The model may be inaccurate, biased, or unsuitable in other contexts and must not replace qualified human judgment.

## License

This project is intended for educational and portfolio purposes.