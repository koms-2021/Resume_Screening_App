import json
import os
import sys

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import (  # pyrefly: ignore [missing-import]
    generate_all_summaries,
    get_resume_chunk_count,
    get_top_candidates,
    process_uploaded_resume,
)

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") or ""
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") or ""

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
ALL_JD_PATH = os.path.join(BASE_DIR, "output_json_jd", "all_jd.json")
ALL_RESUME_PATH = os.path.join(BASE_DIR, "output_jsons_resume", "all_resumes.json")
JD_FOLDER_PATH = os.path.join(BASE_DIR, "output_json_jd")
RESUME_FOLDER_PATH = os.path.join(BASE_DIR, "data")

JD_DISPLAY_NAMES = {
    "DS_JD.txt": "Data Scientist",
    "SWE_JD.txt": "Software Engineer",
    "PM_JD.txt": "Product Manager",
}

st.set_page_config(
    page_title="Resume Screening System",
    layout="wide",
)

st.markdown(
    """
    <style>
        div.stButton > button[kind="primary"] {
            background-color: #2f9e8f;
            color: white;
            border: 1px solid #2f9e8f;
            border-radius: 10px;
            font-weight: 600;
        }

        div.stButton > button[kind="primary"]:hover {
            background-color: #26877a;
            border-color: #26877a;
            color: white;
        }

        div.stButton > button[kind="secondary"] {
            background-color: #e8f7f1;
            color: #13795b;
            border: 1px solid #6fcf97;
            border-radius: 10px;
            font-weight: 600;
        }

        div.stButton > button[kind="secondary"]:hover {
            background-color: #d3f0e4;
            color: #0f684d;
            border-color: #48b884;
        }

        .match-score {
            font-size: 46px;
            font-weight: 500;
            line-height: 1.1;
            margin-top: 4px;
            margin-bottom: 18px;
        }

        .match-label {
            font-size: 16px;
            color: #4a4f5c;
            margin-bottom: 4px;
        }

        .block-container {
            padding-top: 1.25rem;
        }

        h1 {
            margin-bottom: 0.2rem;
        }

        .app-subtitle {
            color: #5f6673;
            margin-bottom: 1rem;
        }

        .upload-panel {
            margin: 1.25rem auto 1.5rem auto;
            padding: 1.25rem 1.5rem;
            border: 1px solid #e2e6ea;
            border-radius: 10px;
            background: #ffffff;
            max-width: 620px;
        }

        .upload-title {
            font-size: 1.1rem;
            font-weight: 650;
            margin-bottom: 0.25rem;
            color: #2b2f3a;
        }

        .upload-copy {
            color: #6b7280;
            font-size: 0.92rem;
            margin-bottom: 0.8rem;
        }

        .compact-gap {
            height: 0.75rem;
        }

        .experience-text {
            color: #858895;
            font-size: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "summaries" not in st.session_state:
    st.session_state.summaries = None

if "jd_parsed" not in st.session_state:
    st.session_state.jd_parsed = None

if "selected_jd" not in st.session_state:
    st.session_state.selected_jd = None

if "top_n" not in st.session_state:
    st.session_state.top_n = 3

if "open_candidate" not in st.session_state:
    st.session_state.open_candidate = None

if "upload_result" not in st.session_state:
    st.session_state.upload_result = None

if "show_upload" not in st.session_state:
    st.session_state.show_upload = True


@st.cache_resource
def load_chromadb():
    raw_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    resume_collection = Chroma(
        client=raw_client,
        collection_name="resumes",
        collection_metadata={"hnsw:space": "cosine"},
    )

    jd_collection = Chroma(
        client=raw_client,
        collection_name="jds",
        collection_metadata={"hnsw:space": "cosine"},
    )

    return resume_collection, jd_collection


@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )


@st.cache_data
def load_json_data():
    with open(ALL_JD_PATH, "r", encoding="utf-8") as f:
        all_jds = json.load(f)

    with open(ALL_RESUME_PATH, "r", encoding="utf-8") as f:
        all_resumes = json.load(f)

    return all_jds, all_resumes


def get_jd_options():
    return [
        f.replace(".json", ".txt")
        for f in os.listdir(JD_FOLDER_PATH)
        if f.endswith(".json") and f != "all_jd.json"
    ]


def get_score_value(score):
    try:
        return float(str(score).replace("%", "").strip())
    except ValueError:
        return 0


def get_score_color(score):
    score_value = get_score_value(score)

    if score_value >= 60:
        return "#1f9d55"

    if score_value >= 30:
        return "#d6a400"

    return "#d64545"


def toggle_candidate_info(filename):
    if st.session_state.open_candidate == filename:
        st.session_state.open_candidate = None
    else:
        st.session_state.open_candidate = filename


def show_upload_form():
    st.session_state.summaries = None
    st.session_state.open_candidate = None
    st.session_state.show_upload = True


def save_uploaded_resume(uploaded_resume):
    os.makedirs(RESUME_FOLDER_PATH, exist_ok=True)
    resume_path = os.path.join(RESUME_FOLDER_PATH, uploaded_resume.name)

    existing_resumes = {}
    if os.path.exists(ALL_RESUME_PATH):
        with open(ALL_RESUME_PATH, "r", encoding="utf-8") as f:
            existing_resumes = json.load(f)

    if uploaded_resume.name not in existing_resumes:
        with open(resume_path, "wb") as f:
            f.write(uploaded_resume.getbuffer())

    return resume_path


st.title("Resume Screening System")

with st.sidebar:
    st.markdown("**Configuration**")

    jd_options = get_jd_options()

    if not jd_options:
        st.error("No JD files found.")
        st.stop()

    jd_label_to_file = {
        JD_DISPLAY_NAMES.get(filename, filename.replace(".txt", "").replace("_", " ")): filename
        for filename in jd_options
    }

    selected_jd_label = st.selectbox(
        label="Select Job Description",
        options=list(jd_label_to_file.keys()),
        index=0,
    )
    selected_jd = jd_label_to_file[selected_jd_label]

    top_n = st.slider(
        label="Number of candidates to show",
        min_value=1,
        max_value=5,
        value=3,
    )

    st.caption("Model: Llama 3.1 8B Instant")
    st.caption("Embeddings: BAAI/bge-large-en-v1.5")
    st.caption("Vector DB: ChromaDB")

    run_button = st.button(
        label="Find Top Candidates",
        use_container_width=True,
        type="primary",
    )

if run_button:
    st.session_state.show_upload = False
    st.session_state.summaries = None
    st.session_state.open_candidate = None
    st.session_state.selected_jd = selected_jd
    st.session_state.top_n = top_n

    with st.spinner("Loading models and database..."):
        resume_collection, jd_collection = load_chromadb()
        llm = load_llm()
        all_jds, all_resumes = load_json_data()

    jd_parsed = all_jds.get(selected_jd, {})
    st.session_state.jd_parsed = jd_parsed

    with st.spinner("Finding best candidates..."):
        resume_chunk_count = get_resume_chunk_count(resume_collection)

        top_candidates = get_top_candidates(
            jd_filename=selected_jd,
            resume_collection=resume_collection,
            jd_collection=jd_collection,
            top_n=top_n,
            top_k=resume_chunk_count,
        )

    if not top_candidates:
        st.error("No candidates found. Check your ChromaDB collections.")
        st.stop()

    with st.spinner("Generating candidate summaries..."):
        st.session_state.summaries = generate_all_summaries(
            top_candidates=top_candidates,
            jd_filename=selected_jd,
            all_resumes=all_resumes,
            all_jds=all_jds,
            llm=llm,
        )

if st.session_state.show_upload and not st.session_state.summaries:
    st.info("Select a job description from the sidebar and click Find Top Candidates to get started.")
    upload_left, upload_center, upload_right = st.columns([1.2, 2.2, 1.2])
    with upload_center:
        with st.container(border=True):
            st.markdown("**Upload Resume**")
            st.caption("Add a new PDF resume to the screening database.")

            uploaded_resume = st.file_uploader(
                "Choose a PDF resume",
                type=["pdf"],
                accept_multiple_files=False,
            )

            upload_button = st.button(
                "Process Uploaded Resume",
                use_container_width=True,
            )

            if upload_button:
                if uploaded_resume is None:
                    st.warning("Please choose a PDF resume first.")
                else:
                    with st.spinner("Processing uploaded resume..."):
                        resume_collection, _ = load_chromadb()
                        llm = load_llm()
                        resume_path = save_uploaded_resume(uploaded_resume)
                        st.session_state.upload_result = process_uploaded_resume(
                            file_path=resume_path,
                            llm=llm,
                            resume_collection=resume_collection,
                            force=False,
                        )
                        load_json_data.clear()
                        st.session_state.summaries = None
                        st.session_state.open_candidate = None

            if st.session_state.upload_result:
                parse_result = st.session_state.upload_result.get("parse", {})
                vector_result = st.session_state.upload_result.get("vector_store", {})
                status = parse_result.get("status")
                message = parse_result.get("message", "Upload completed.")

                if status == "processed":
                    st.success(message)
                elif status == "skipped":
                    st.info(message)
                else:
                    st.error(message)

                stored_chunks = vector_result.get("stored_chunks")
                if stored_chunks is not None:
                    st.caption(f"Vector chunks added: {stored_chunks}")

if st.session_state.summaries:
    if not st.session_state.show_upload:
        action_left, action_right = st.columns([5, 1.4])
        with action_right:
            st.button(
                "Upload Another Resume",
                use_container_width=True,
                on_click=show_upload_form,
            )

    jd_parsed = st.session_state.jd_parsed or {}

    if jd_parsed:
        st.subheader(jd_parsed.get("job_title", "Job"))
        st.caption(
            f"Location: {jd_parsed.get('location', 'N/A')} | "
            f"Experience: {jd_parsed.get('experience_required', 'N/A')} | "
            f"Employment Type: {jd_parsed.get('employment_type', 'N/A')}"
        )
        st.markdown('<div class="compact-gap"></div>', unsafe_allow_html=True)

    st.subheader(f"Top {st.session_state.top_n} Candidates")
    st.caption("A concise shortlist of the best matching resumes.")

    for filename, data in st.session_state.summaries.items():
        rank = data["rank"]
        scores = data["scores"]
        summary = data["summary"]

        candidate_name = summary.get("candidate_name", filename)
        match_score = scores["percentage"]
        score_color = get_score_color(match_score)

        with st.container(border=True):
            col_rank, col_candidate, col_score, col_action = st.columns(
                [0.7, 3.2, 1.2, 1]
            )

            with col_rank:
                st.markdown(f"### #{rank}")

            with col_candidate:
                st.markdown(f"**{candidate_name}**")
                st.write(summary.get("match_summary", "N/A"))

            with col_score:
                st.markdown(
                    f"""
                    <div class="match-label">Matching Score</div>
                    <div class="match-score" style="color: {score_color};">
                        {match_score}
                    </div>
                    <div class="experience-text">
                        Experience: {scores['resume_years']} yrs
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_action:
                button_label = (
                    "Hide" if st.session_state.open_candidate == filename else "Info"
                )

                st.button(
                    button_label,
                    key=f"info_{filename}",
                    use_container_width=True,
                    on_click=toggle_candidate_info,
                    args=(filename,),
                )

            if st.session_state.open_candidate == filename:
                st.divider()

                st.markdown("**Experience Summary**")
                st.write(summary.get("experience_summary", "N/A"))

                st.markdown("")

                strengths_col, gaps_col = st.columns(2)

                with strengths_col:
                    st.markdown("**Key Strengths**")
                    strengths = summary.get("key_strengths") or []

                    if strengths:
                        for strength in strengths[:3]:
                            st.markdown(f"- {strength}")
                    else:
                        st.markdown("- No clear strengths found.")

                with gaps_col:
                    st.markdown("**Skill Gaps**")
                    gaps = summary.get("skill_gaps") or []

                    if gaps:
                        for gap in gaps[:3]:
                            st.markdown(f"- {gap}")
                    else:
                        st.markdown("- No major skill gaps found.")

                st.markdown("**Recommendation**")
                st.info(summary.get("recommendation", "N/A"))

