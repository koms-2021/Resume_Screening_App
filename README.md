# Resume Screening System

A Streamlit-based resume screening application that ranks candidates against a selected job description using semantic search, keyword relevance, role-fit scoring, experience matching, and LLM-generated recruiter summaries.

The system is designed to help recruiters and hiring teams quickly shortlist the most relevant candidates from a resume database while still providing transparent match details such as relevance score, role fit, experience fit, strengths, gaps, and recommendations.

The system follows a Retrieval-Augmented Generation (RAG) approach: relevant resume chunks are retrieved from Chroma Cloud and used with structured resume/JD data to generate recruiter-friendly candidate evaluations.

## Features

- Select a job description and retrieve the top matching candidates.
- Rank resumes using a hybrid retrieval approach:
  - Chroma Cloud vector similarity
  - BM25 keyword matching
  - Role-family fit scoring
  - Experience fit scoring
- Generate concise candidate summaries with Groq-hosted Llama 3.1.
- Upload a PDF resume during a Streamlit session and evaluate it immediately.
- Display candidate match score, experience, strengths, skill gaps, and recruiter recommendation.
- Load structured resume, job description, and chunk data from Supabase.

## Tech Stack

- **Frontend:** Streamlit
- **LLM:** Groq `llama-3.1-8b-instant`/ Google `gemini-2.5-flash`
- **Embeddings:** HuggingFace `BAAI/bge-large-en-v1.5`
- **Vector Database:** Chroma Cloud
- **Structured Data Store:** Supabase
- **Retrieval:** LangChain, Chroma, BM25
- **PDF Parsing:** PyMuPDF
- **Language:** Python 3.11

## Project Structure

```text
.
|-- app_deploy/
|   |-- app.py              # Streamlit application
|   `-- pipeline.py         # Resume parsing, chunking, scoring, ranking, and summary logic
|-- Notebooks/              # Experimentation notebooks for extraction, embeddings, and vector search
|-- requirements.txt        # Python dependencies
|-- pyproject.toml          # Project metadata
|-- runtime.txt             # Python runtime for deployment platforms
|-- main.py                 # Minimal project entry point
`-- README.md
```

## How It Works

1. Resume and job description data are stored in Supabase as structured JSON.
2. Resume and JD chunks are indexed in Chroma Cloud.
3. The user selects a job description in the Streamlit sidebar.
4. The app retrieves matching resume chunks using vector search and BM25.
5. Candidate-level scores are aggregated using:
   - 70% semantic and keyword relevance
   - 20% role fit
   - 10% experience fit
6. The top candidates are summarized by the LLM for recruiter-friendly review.
7. Optional PDF uploads are parsed, chunked, embedded, and evaluated in the current session.

## Prerequisites

- Python 3.11
- Supabase project with the required tables
- Chroma Cloud account and database
- Groq API key/ Google API key
- Optional Google API key if you use the related notebook workflows

## Supabase Data Requirements

The Streamlit app expects these Supabase tables:

### `resumes`

| Column | Description |
| --- | --- |
| `filename` | Resume file name |
| `resume_json` | Parsed resume JSON |

### `job_descriptions`

| Column | Description |
| --- | --- |
| `filename` | Job description file name |
| `jd_json` | Parsed job description JSON |

### `resume_chunks`

| Column | Description |
| --- | --- |
| `resume_filename` | Source resume file name |
| `chunk_index` | Chunk order |
| `chunk_json` | Chunk JSON containing type and content |
| `chunk_type` | Fallback chunk type |
| `content` | Fallback chunk text |

Chroma Cloud should contain two collections:

- `resumes`
- `jds`

## Installation

Clone the repository and install dependencies:

```bash
git clone <your-repository-url>
cd Resume_Screening_Project_Updated
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the App

Start the Streamlit application:

```bash
streamlit run app_deploy/app.py
```

Then open the local Streamlit URL shown in the terminal.

## Usage

1. Select a job description from the sidebar.
2. Choose the number of candidates to display.
3. Click **Get Top Candidates**.
4. Review ranked candidates, match scores, strengths, skill gaps, and recommendations.
5. Optionally upload a PDF resume to process and evaluate it for the current session.

## Notebooks

The `Notebooks/` directory contains exploratory workflows for:

- Raw text extraction from resumes
- Resume chunking and embedding generation
- Local Chroma vector search
- Chroma Cloud vector search

These notebooks are useful for experimentation, data preparation, and validating retrieval quality before deployment.

## Notes

- Do not commit `.env`, API keys, resumes, generated embeddings, or parsed candidate data.
- Make sure Chroma Cloud and Supabase are populated before running the deployed app.
- The default scoring weights are defined in `app_deploy/pipeline.py`.
- The app currently uses Python 3.11 as specified in `runtime.txt` and `pyproject.toml`.
