import streamlit as st
import pickle
import docx
import PyPDF2
import re
import torch
import pandas as pd
import ast
import os
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import plotly.graph_objects as go

# -------------------------
# Load skills dataset
# -------------------------
FILE_PATH = "merged_top_skills.xlsx"
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError("Skills file not found")

SKILLS_DF = pd.read_excel(FILE_PATH)
SKILLS_DF["Category"] = SKILLS_DF["Category"].str.lower()
SKILLS_DF["SKILLS"] = SKILLS_DF["SKILLS"].apply(ast.literal_eval)

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_models():
    if not os.path.exists("resume_bert_model"):
        raise FileNotFoundError("Model folder not found")

    tokenizer = DistilBertTokenizerFast.from_pretrained("resume_bert_model")
    model = DistilBertForSequenceClassification.from_pretrained("resume_bert_model")
    le = pickle.load(open("label_encoder.pkl", "rb"))
    model.eval()
    return model, tokenizer, le

model, tokenizer, le = load_models()

# -------------------------
# Text cleaning
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# -------------------------
# Resume readers
# -------------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return " ".join(p.extract_text() or "" for p in reader.pages)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join(p.text for p in doc.paragraphs)

def extract_text_from_txt(file):
    return file.read().decode("utf-8", errors="ignore")

def handle_file_upload(file):
    ext = file.name.split(".")[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file)
    if ext == "docx":
        return extract_text_from_docx(file)
    if ext == "txt":
        return extract_text_from_txt(file)
    raise ValueError("Unsupported file type")

# -------------------------
# Experience detection
# -------------------------
def detect_experience_level(text):
    text = text.lower()
    if re.search(r"\b(0|1|2)\+?\s*(years|yrs)\b|(fresher|intern)", text):
        return "Entry Level"
    if re.search(r"\b(3|4|5)\+?\s*(years|yrs)\b", text):
        return "Intermediate Level"
    if re.search(r"\b(6|7|8|9|10)\+?\s*(years|yrs)\b|(senior|lead)", text):
        return "Senior Level"
    return "Not specified"

# -------------------------
# Category prediction
# -------------------------
def predict_category(text):
    inputs = tokenizer(
        clean_text(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return le.inverse_transform([pred_id])[0]

# -------------------------
# Category skill analysis
# -------------------------
def category_skill_analysis(resume_text, predicted_category):
    resume_text = resume_text.lower()
    cat = predicted_category.lower()

    row = SKILLS_DF[SKILLS_DF["Category"] == cat]
    if row.empty:
        return [], [], 0.0, 0

    category_skills = [s.lower() for s in row.iloc[0]["SKILLS"]]
    extracted = sorted([s for s in category_skills if s in resume_text])
    missing = sorted(set(category_skills) - set(extracted))
    total = len(category_skills)
    score = round(len(extracted) / max(total, 1) * 100, 2)

    return extracted, missing, score, total

# -------------------------
# Resume score donut
# -------------------------
def create_resume_score_viz(score, matched, total, experience="Good", category_fit="Excellent"):
    fig = go.Figure(
        data=[go.Pie(
            values=[score, 100 - score],
            hole=0.7,
            marker=dict(colors=["#00D4FF", "#2A1F3A"]),
            textinfo="none",
            showlegend=False,
            hovertemplate=
            f"<b>Core Match:</b> {matched} / {total}<br>" +
            f"<b>Role Fit:</b> {experience}<br>" +
            f"<b>Category Fit:</b> {category_fit}<extra></extra>"
        )]
    )

    fig.update_layout(
        width=250,
        height=250,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"<b>{score}%</b>",
            x=0.5,
            y=0.5,
            font=dict(size=40, color="#00D4FF"),
            showarrow=False
        )]
    )
    return fig

# -------------------------
# Streamlit CSS
# -------------------------
def apply_css():
    st.markdown("""
    <style>
    .stApp { background-color:#0c0c0f;color:#F9CBDF }
    h1,h2,h3 { color:#B89AC9 }
    .skill-chip { background:#794C9E;color:white;padding:6px 14px;border-radius:20px;margin:4px;display:inline-block }
    .info-box {
        background-color: #2A1F3A;
        border: 1px solid #794C9E;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(121, 76, 158, 0.2);
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .info-box .box-title { color:#B89AC9;font-size:20px;font-weight:600;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px; }
    .info-box .box-content { color:#F9CBDF;font-size:18px;font-weight:500;margin:0;line-height:1.2; }
    .footer-banner {
        background: linear-gradient(135deg, #2A1F3A 0%, #794C9E 100%);
        padding: 8px;
        text-align: center;
        border-top: 2px solid #B89AC9;
        margin-top: 40px;
        border-radius: 10px;
        box-shadow: 0 -4px 12px rgba(121, 76, 158, 0.3);
    }
    .footer-text {
        color: #F9CBDF;
        font-size: 16px;
        font-weight: 500;
        margin: 0;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# Main UI
# -------------------------
def main():
    st.set_page_config(page_title="Resume Screener", layout="wide")
    apply_css()

    st.markdown("<h1 style='text-align:center'>Resume Screening App</h1>", unsafe_allow_html=True)

    file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

    if file and st.button("Screen Resume"):
        text = handle_file_upload(file)
        category = predict_category(text)
        skills, missing, score, total = category_skill_analysis(text, category)
        experience = detect_experience_level(text)

        # Category and Experience boxes
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <div class="box-title">Predicted Category</div>
                <div class="box-content">{category}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="info-box">
                <div class="box-title">Experience Level</div>
                <div class="box-content">{experience}</div>
            </div>
            """, unsafe_allow_html=True)

        # Skills + Score row
        skills_col, score_col = st.columns([1, 1])

        with skills_col:
            st.markdown("<h3 style='text-align:center'>Skills Analysis</h3>", unsafe_allow_html=True)
            st.markdown("**Extracted Skills**")
            st.markdown(" ".join([f"<span class='skill-chip'>{s.title()}</span>" for s in skills]) if skills else "None", unsafe_allow_html=True)
            st.markdown("**Missing Skills**")
            st.markdown(" ".join([f"<span class='skill-chip'>{s.title()}</span>" for s in missing]) if missing else "None", unsafe_allow_html=True)

        with score_col:
            st.markdown("<h3 style='text-align:center'>Resume Match Score</h3>", unsafe_allow_html=True)
            st.markdown("<div style='height:45px'></div>", unsafe_allow_html=True)
            inner_col1, inner_col2, inner_col3 = st.columns([1,2,1])
            with inner_col2:
                st.plotly_chart(create_resume_score_viz(score, len(skills), total, experience, category_fit="Excellent"), use_container_width=False)

        st.markdown("<div style='height:75px'></div>", unsafe_allow_html=True)
        with st.expander("Show Resume Text"):
            st.text_area("Resume", text, height=300)

    # Footer banner
    st.markdown("""
    <div class="footer-banner">
        <p class="footer-text">Developed by Sana Usman © 2025 | Intelligent Resume Screening Solutions </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
