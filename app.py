import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, hstack

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]
CATEGORICAL_COLUMNS = [
    "location", "department", "employment_type", "required_experience",
    "required_education", "industry", "function"
]
NUMERIC_COLUMNS = ["telecommuting", "has_company_logo", "has_questions"]

st.set_page_config(page_title="Fake Job Detection", page_icon="🛡️", layout="wide")

def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@st.cache_resource
def load_artifacts():
    required = {
        "svm": MODEL_DIR / "svm.pkl",
        "tfidf": MODEL_DIR / "tfidf.pkl",
        "categorical_pipeline": MODEL_DIR / "categorical_pipeline.pkl",
    }
    missing = [str(path.relative_to(BASE_DIR)) for path in required.values() if not path.exists()]
    if missing:
        return None, missing
    return {
        "svm": joblib.load(required["svm"]),
        "tfidf": joblib.load(required["tfidf"]),
        "categorical_pipeline": joblib.load(required["categorical_pipeline"]),
    }, []

def predict_job(artifacts, details):
    # Match the SVM notebook's feature construction: combine the five text fields,
    # TF-IDF transform that text, one-hot encode categorical fields, then append
    # the three numeric/binary features.
    text_parts = [str(details.get(col, "") or "") for col in TEXT_COLUMNS]
    combined_text = " ".join(text_parts)

    text_matrix = artifacts["tfidf"].transform([combined_text])

    cat_row = {col: str(details.get(col, "Unknown") or "Unknown") for col in CATEGORICAL_COLUMNS}
    cat_frame = pd.DataFrame([cat_row], columns=CATEGORICAL_COLUMNS)
    cat_matrix = artifacts["categorical_pipeline"].transform(cat_frame)

    numeric_values = [[
        int(details.get("telecommuting", 0)),
        int(details.get("has_company_logo", 0)),
        int(details.get("has_questions", 0)),
    ]]
    numeric_matrix = csr_matrix(numeric_values)
    features = hstack([text_matrix, cat_matrix, numeric_matrix]).tocsr()

    prediction = int(artifacts["svm"].predict(features)[0])
    # LinearSVC has a decision score, not a calibrated probability.
    decision_score = None
    if hasattr(artifacts["svm"], "decision_function"):
        decision_score = float(artifacts["svm"].decision_function(features)[0])
    return prediction, decision_score

st.title("🛡️ Fake Job Detection System")
st.write(
    "Check a job advertisement using the trained Support Vector Machine (SVM) model. "
    "Choose manual entry or paste the complete job posting."
)
st.info("The result is a machine-learning prediction, not a guarantee that a job is genuine or fraudulent.")

artifacts, missing_files = load_artifacts()
if artifacts is None:
    st.error("Model files are missing. Save the trained SVM, TF-IDF vectorizer, and categorical preprocessing pipeline into the models/ folder.")
    st.code(
        "models/svm.pkl\n"
        "models/tfidf.pkl\n"
        "models/categorical_pipeline.pkl"
    )
    st.markdown("Follow the **SETUP.md** file included with this app to export the artifacts from your existing SVM notebook.")
    st.stop()

mode = st.radio(
    "How would you like to enter the job posting?",
    ["Enter details manually", "Paste the complete job posting"],
    horizontal=True,
)

details = {
    "title": "",
    "company_profile": "",
    "description": "",
    "requirements": "",
    "benefits": "",
    "location": "Unknown",
    "department": "Unknown",
    "employment_type": "Unknown",
    "required_experience": "Unknown",
    "required_education": "Unknown",
    "industry": "Unknown",
    "function": "Unknown",
    "telecommuting": 0,
    "has_company_logo": 0,
    "has_questions": 0,
}

with st.form("job_form"):
    if mode == "Enter details manually":
        st.subheader("Job details")
        col1, col2 = st.columns(2)
        with col1:
            details["title"] = st.text_input("Job title *")
            details["location"] = st.text_input("Location", value="Unknown")
            details["department"] = st.text_input("Department", value="Unknown")
            details["employment_type"] = st.selectbox(
                "Employment type", ["Unknown", "Full-time", "Part-time", "Contract", "Temporary", "Other"]
            )
            details["required_experience"] = st.selectbox(
                "Required experience", ["Unknown", "Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
            )
        with col2:
            details["required_education"] = st.selectbox(
                "Required education", ["Unknown", "High School or equivalent", "Bachelor's Degree", "Master's Degree", "Doctorate", "Unspecified", "Other"]
            )
            details["industry"] = st.text_input("Industry", value="Unknown")
            details["function"] = st.text_input("Job function", value="Unknown")
            details["has_company_logo"] = int(st.checkbox("Company logo is shown"))
            details["has_questions"] = int(st.checkbox("Application includes screening questions"))
            details["telecommuting"] = int(st.checkbox("Remote/telecommuting job"))

        details["company_profile"] = st.text_area("Company profile", height=100)
        details["description"] = st.text_area("Job description *", height=180)
        details["requirements"] = st.text_area("Requirements", height=120)
        details["benefits"] = st.text_area("Benefits", height=100)
    else:
        st.subheader("Paste job advertisement")
        pasted_text = st.text_area(
            "Paste the full job post here *",
            height=320,
            placeholder="Paste the job title, company information, description, requirements, benefits, and any other details..."
        )
        st.caption(
            "The complete pasted text is sent to the text model as job-post content. "
            "Structured fields not separately provided will use 'Unknown' or 0."
        )
        details["description"] = pasted_text

    submitted = st.form_submit_button("Analyze Job Posting", type="primary", use_container_width=True)

if submitted:
    has_content = (
        bool(details["description"].strip())
        or bool(details["title"].strip())
        or bool(details["company_profile"].strip())
    )
    if not has_content:
        st.warning("Please enter job details or paste a complete job posting before analyzing.")
    else:
        with st.spinner("Analyzing job posting..."):
            prediction, decision_score = predict_job(artifacts, details)

        st.divider()
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("⚠️ Potentially Fake Job Posting")
            st.write("The model classified this posting as potentially fraudulent. Verify the company independently before sharing information or paying any fee.")
        else:
            st.success("✅ Likely Legitimate Job Posting")
            st.write("The model classified this posting as legitimate. This does not guarantee authenticity; independently verify the employer and offer.")

        st.caption("Model used: Support Vector Machine (LinearSVC)")
        if decision_score is not None:
            st.caption(
                f"Model decision score: {decision_score:.3f} "
                "(not a probability or confidence percentage)."
            )
