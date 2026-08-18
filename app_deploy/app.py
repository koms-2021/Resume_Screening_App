import json
import os
import re
import sys
import tempfile
import importlib

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from supabase import create_client

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pipeline as pipeline_module  # pyrefly: ignore [missing-import]

# Streamlit can rerun this file while retaining an older imported pipeline in
# sys.modules. Reload it so parser/scoring fixes are applied without requiring
# the user to kill the existing Streamlit process manually.
pipeline_module = importlib.reload(pipeline_module)

from pipeline import (  # pyrefly: ignore [missing-import]
    build_resume_chunks_dict,
    generate_all_summaries,
    get_resume_chunk_count,
    get_top_candidates,
    process_uploaded_resume_for_session,
)

load_dotenv(override=True)


def get_secret(*names, default=""):
    for name in names:
        try:
            value = st.secrets.get(name)
        except Exception:
            value = None

        if value:
            return str(value)

        value = os.getenv(name)
        if value:
            return value

    return default

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

JD_DISPLAY_NAMES = {
    "DS_JD.txt": "Data Scientist",
    "SWE_JD.txt": "Software Engineer",
    "PM_JD.txt": "Product Manager",
}

# Increment whenever stored score/summary output becomes incompatible with the
# current pipeline. This prevents Streamlit from displaying stale session cards.
RESULT_SCHEMA_VERSION = 3

PERMANENT_RESUME_NAME_SET = {
    "abhishek_shaurya_resume",
    "jay kumar behera_cv_ds",
    "komal_kamble_1",
    "meetlad_resume",
    "prity-kumari-resume-devops-2025",
    "soumyadeep_sen_cv",
    "manandaxini-updated-2026-t",
}


def normalize_uploaded_resume_name(filename):
    resume_name = os.path.splitext(filename)[0]
    resume_name = re.sub(r"(?:\s*\(\d+\))+$", "", resume_name)
    return resume_name.strip().lower()

st.set_page_config(
    page_title="Resume Screening System",
    layout="wide",
)

os.environ["GROQ_API_KEY"] = get_secret("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = get_secret("GOOGLE_API_KEY")

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

if "uploaded_session_resume" not in st.session_state:
    st.session_state.uploaded_session_resume = None

if "show_upload" not in st.session_state:
    st.session_state.show_upload = True

if st.session_state.get("result_schema_version") != RESULT_SCHEMA_VERSION:
    st.session_state.summaries = None
    st.session_state.open_candidate = None
    st.session_state.result_schema_version = RESULT_SCHEMA_VERSION


@st.cache_resource
def load_chromadb():
    chroma_api_key = get_secret("CHROMA_API_KEY", "Chroma_API_KEY")
    chroma_tenant = (
        get_secret("CHROMA_TENANT_ID")
        or get_secret("Chroma_tenant_Id")
        or get_secret("Tenant_Id")
    )
    chroma_database = (
        get_secret("CHROMA_DATABASE_ID")
        or get_secret("Chroma_database_Id")
        or get_secret("Chroma_Database")
        or get_secret("Database_Name")
        or get_secret("Database")
    )

    missing = [
        name
        for name, value in {
            "CHROMA_API_KEY": chroma_api_key,
            "CHROMA_TENANT_ID": chroma_tenant,
            "CHROMA_DATABASE_ID": chroma_database,
        }.items()
        if not value
    ]

    if missing:
        raise ValueError(
            "Missing Chroma Cloud secret(s) or environment variable(s): "
            + ", ".join(missing)
        )

    raw_client = chromadb.CloudClient(
        api_key=chroma_api_key,
        tenant=chroma_tenant,
        database=chroma_database,
    )

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
def load_supabase_client():
    supabase_url = get_secret("SUPABASE_URL")
    supabase_key = (
        get_secret("SUPABASE_ANON_KEY")
        or get_secret("SUPABASE_SERVICE_ROLE_KEY")
        or get_secret("SUPABASE_KEY")
    )

    if not supabase_url or not supabase_key:
        raise ValueError(
            "Missing SUPABASE_URL or Supabase API key in Streamlit secrets "
            "or environment variables"
        )

    return create_client(supabase_url, supabase_key)


def fetch_all_rows(supabase, table_name, page_size=1000):
    rows = []
    start = 0

    while True:
        end = start + page_size - 1
        response = supabase.table(table_name).select("*").range(start, end).execute()
        batch = response.data or []
        rows.extend(batch)

        if len(batch) < page_size:
            break

        start += page_size

    return rows


@st.cache_resource
def load_summary_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )


@st.cache_resource
def load_resume_parser_llm():
    return ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        max_tokens=2000,
    )


@st.cache_data
def load_json_data():
    supabase = load_supabase_client()

    resume_rows = fetch_all_rows(supabase, "resumes")
    jd_rows = fetch_all_rows(supabase, "job_descriptions")
    chunk_rows = fetch_all_rows(supabase, "resume_chunks")

    all_resumes = {
        row["filename"]: row["resume_json"]
        for row in resume_rows
        if row.get("filename")
    }
    all_jds = {
        row["filename"]: row["jd_json"]
        for row in jd_rows
        if row.get("filename")
    }
    all_resume_chunks = build_resume_chunks_dict(chunk_rows)

    return all_jds, all_resumes, all_resume_chunks


def get_jd_options(all_jds):
    return sorted(all_jds.keys())


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
    st.session_state.upload_result = None
    st.session_state.uploaded_session_resume = None
    st.session_state.show_upload = True


def process_session_upload(uploaded_resume, llm):
    uploaded_resume_name = normalize_uploaded_resume_name(uploaded_resume.name)

    if uploaded_resume_name in PERMANENT_RESUME_NAME_SET:
        return {
            "status": "skipped",
            "message": (
                f"{uploaded_resume_name} is already in the system, "
                "so this resume does not need to be processed again."
            ),
            "filename": uploaded_resume.name,
            "resume": None,
            "chunks": [],
            "embedded_chunks": [],
        }

    suffix = os.path.splitext(uploaded_resume.name)[1] or ".pdf"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_resume.getbuffer())
            temp_path = temp_file.name

        try:
            result = process_uploaded_resume_for_session(
                file_path=temp_path,
                llm=llm,
            )
        except Exception as exc:
            error_text = str(exc)
            if "rate_limit_exceeded" in error_text or "tokens per minute" in error_text:
                message = (
                    "The uploaded resume is too large for the current Groq token "
                    "limit. Please try a shorter resume or upgrade the Groq tier."
                )
            else:
                message = f"Could not process uploaded resume: {exc}"

            return {
                "status": "failed",
                "message": message,
                "filename": uploaded_resume.name,
                "resume": None,
                "chunks": [],
                "embedded_chunks": [],
            }

        if result.get("filename"):
            result["filename"] = uploaded_resume.name

        if result.get("status") == "processed":
            candidate_name = str(
                (result.get("resume") or {}).get("name") or "Candidate"
            ).strip()
            if candidate_name.isupper():
                candidate_name = candidate_name.title()

            result["message"] = (
                f"{candidate_name} ({uploaded_resume.name}) "
                "processed for this session."
            )

        return result
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


st.title("Resume Screening System")

try:
    all_jds_data, all_resumes_data, all_resume_chunks_data = load_json_data()
except Exception as exc:
    st.error(f"Could not load Supabase data: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("**Configuration**")

    jd_options = get_jd_options(all_jds_data)

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
    st.caption("Vector DB: Chroma Cloud")

    run_button = st.button(
        label=f"Get Top {top_n} Candidates",
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
        summary_llm = load_summary_llm()
        all_jds, all_resumes, all_resume_chunks = load_json_data()

    jd_parsed = all_jds.get(selected_jd, {})
    st.session_state.jd_parsed = jd_parsed

    with st.spinner("Finding best candidates..."):
        resume_chunk_count = get_resume_chunk_count(resume_collection)
        session_resume = st.session_state.uploaded_session_resume

        top_candidates = get_top_candidates(
            jd_filename=selected_jd,
            resume_collection=resume_collection,
            jd_collection=jd_collection,
            top_n=top_n,
            top_k=resume_chunk_count,
            all_jds=all_jds,
            all_resumes=all_resumes,
            all_resume_chunks=all_resume_chunks,
            session_resume=session_resume,
        )

    if not top_candidates:
        st.error("No candidates found. Check your Chroma Cloud collections.")
        st.stop()

    with st.spinner("Generating candidate summaries..."):
        if st.session_state.uploaded_session_resume:
            all_resumes = dict(all_resumes)
            all_resumes[
                st.session_state.uploaded_session_resume["filename"]
            ] = st.session_state.uploaded_session_resume["resume"]

        st.session_state.summaries = generate_all_summaries(
            top_candidates=top_candidates,
            jd_filename=selected_jd,
            all_resumes=all_resumes,
            all_jds=all_jds,
            llm=summary_llm,
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
                        resume_parser_llm = load_resume_parser_llm()
                        st.session_state.upload_result = process_session_upload(
                            uploaded_resume=uploaded_resume,
                            llm=resume_parser_llm,
                        )
                        if st.session_state.upload_result.get("status") == "processed":
                            st.session_state.uploaded_session_resume = st.session_state.upload_result
                        load_json_data.clear()
                        st.session_state.summaries = None
                        st.session_state.open_candidate = None

            if st.session_state.upload_result:
                parse_result = st.session_state.upload_result.get("parse", {})
                status = (
                    parse_result.get("status")
                    or st.session_state.upload_result.get("status")
                )
                message = (
                    parse_result.get("message")
                    or st.session_state.upload_result.get("message")
                    or "Upload completed."
                )

                if status == "processed":
                    st.success(message)
                elif status == "skipped":
                    st.info(message)
                else:
                    st.error(message)

                embedded_chunks = st.session_state.upload_result.get("embedded_chunks")
                if embedded_chunks is not None:
                    st.caption(f"Session chunks ready: {len(embedded_chunks)}")

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
        resume_experience_text = scores.get(
            "resume_experience_text",
            f"{scores['resume_years']} yrs",
        )

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
                        Experience: {resume_experience_text}
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

