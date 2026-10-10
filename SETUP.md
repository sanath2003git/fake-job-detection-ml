# Streamlit interface setup

## 1. Add the app to your existing project

Copy `app.py` into the root of your project, beside the `notebooks/` directory. Create a `models/` folder in that same root directory.

The app is wired to your **SVM notebook's feature construction**:
- combines `title`, `company_profile`, `description`, `requirements`, and `benefits`
- uses the fitted TF-IDF vectorizer
- uses the fitted categorical preprocessing pipeline
- appends `telecommuting`, `has_company_logo`, and `has_questions`
- calls the saved `LinearSVC` model

The ZIP you supplied contains notebooks but does not include the trained model artifacts, so the interface will show a missing-artifacts message until you export them.

## 2. Export the fitted SVM artifacts

Open `notebooks/05_svm.ipynb`. After the cells that fit `tfidf`, `categorical_pipeline`, and `svm_model` have run successfully, add and run this cell:

```python
from pathlib import Path
import joblib

# Run from the notebook's current working directory.
# If the notebook runs from the notebooks/ folder:
models_dir = Path("../models")
models_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(svm_model, models_dir / "svm.pkl")
joblib.dump(tfidf, models_dir / "tfidf.pkl")
joblib.dump(categorical_pipeline, models_dir / "categorical_pipeline.pkl")

print("Saved SVM and preprocessing artifacts to:", models_dir.resolve())
```

Important: save the *fitted* vectorizer and fitted categorical pipeline used to create `X_train_final` / `X_test_final`. Do not create new vectorizers or encoders for the app.

If your notebook is running in a different working directory, adjust `models_dir` so the files end up in the project's root `models/` directory.

## 3. Install packages

From the project root, run:

```bash
pip install streamlit pandas scikit-learn scipy joblib
```

Use the same scikit-learn version as the environment in which the model was trained, when possible.

## 4. Run the interface

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## 5. Two input modes

- **Enter details manually:** enter the structured fields and job description.
- **Paste the complete job posting:** paste the entire advertisement into one text area. Since the pasted text may not reliably separate structured fields, missing structured values default to `Unknown` and binary flags default to `0`.

## Important model note

Your notebook's saved `LinearSVC` produces a class prediction and a decision score, not a calibrated probability. The app deliberately does not label the score as a percentage confidence. Predictions are screening aids and do not independently prove whether an employer is genuine.
