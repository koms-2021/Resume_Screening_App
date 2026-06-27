import json
import re
import os
import math
from datetime import datetime
from collections import defaultdict

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

# Hybrid chunk score blend weights
VECTOR_CHUNK_WEIGHT = 0.80
BM25_CHUNK_WEIGHT = 0.20

# Final candidate score blend weights
RELEVANCE_WEIGHT = 0.70
ROLE_FIT_WEIGHT = 0.20
EXPERIENCE_FIT_WEIGHT = 0.10

# JD chunk type weights
# Weights applied to JD requirements (not resume sections)
JD_CHUNK_WEIGHTS = {
    "skills"          : 0.35,
    "responsibilities": 0.30,
    "qualifications"  : 0.20,
    "overview"        : 0.10,
    "summary"         : 0.10,
    "experience"      : 0.10,
    "education"       : 0.05
}

# File paths
ALL_JD_PATH     = os.path.join(os.path.dirname(__file__), "..", "output_json_jd", "all_jd.json")
ALL_RESUME_PATH = os.path.join(os.path.dirname(__file__), "..", "output_jsons_resume", "all_resumes.json")

# File paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESUME_DATA_DIR = os.path.join(BASE_DIR, "data")
RESUME_OUTPUT_DIR = os.path.join(BASE_DIR, "output_jsons_resume")
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")

ALL_JD_PATH = os.path.join(BASE_DIR, "output_json_jd", "all_jd.json")
ALL_RESUME_PATH = os.path.join(RESUME_OUTPUT_DIR, "all_resumes.json")
ALL_RESUME_CHUNKS_PATH = os.path.join(CHUNKS_DIR, "all_resume_chunks.json")
ALL_RESUME_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "all_resume_embeddings.json")
RESUME_EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EXPECTED_RESUME_EMBEDDING_DIM = 1024


# Upload processing helpers
RESUME_EXTRACTION_TEMPLATE = """
You are an expert resume parser. Your job is to extract structured information from the raw resume text provided below.

RESUME TEXT:
{extracted_text}

Extract the following fields and return ONLY valid JSON matching this schema:
{
  "name": string or null,
  "contact": {"email": string or null, "phone": string or null, "location": string or null, "linkedin": string or null},
  "summary": string or null,
  "skills": [string] or null,
  "experience": [{"company": string or null, "title": string or null, "start_date": string or null, "end_date": string or null, "responsibilities": [string]}] or null,
  "education": [{"degree": string or null, "institution": string or null, "year": string or null}] or null,
  "projects": [{"name": string or null, "description": string or null, "tech": [string]}] or null,
  "certifications": [string] or null
}

---

FEW SHOT EXAMPLES:

Example 1 — Software Engineer Resume:

Input:
John Doe
john.doe@gmail.com | +1-9876543210 | New York, USA | linkedin.com/in/johndoe

Profile:
Software Engineer with 3 years of experience building scalable backend systems.

Skills: Python, Django, PostgreSQL, Docker, AWS, Git

Experience:
Software Engineer, Google
06/2021 – Present | New York, USA
- Designed and deployed RESTful APIs serving 1M+ requests/day
- Reduced system latency by 40% through query optimization

Education:
B.Tech - Computer Science, MIT | 2017 – 2021

Projects:
Smart Inventory System
Built an automated inventory tracking tool using IoT sensors and Python.
Tech: Python, MQTT, PostgreSQL, Docker

Certifications:
AWS Certified Developer – Associate

Output:
{{
  "name": "John Doe",
  "contact": {{
    "email": "john.doe@gmail.com",
    "phone": "+1-9876543210",
    "location": "New York, USA",
    "linkedin": "linkedin.com/in/johndoe"
  }},
  "summary": "Software Engineer with 3 years of experience building scalable backend systems.",
  "skills": ["Python", "Django", "PostgreSQL", "Docker", "AWS", "Git"],
  "experience": [
    {{
      "company": "Google",
      "title": "Software Engineer",
      "start_date": "06/2021",
      "end_date": "Present",
      "responsibilities": [
        "Designed and deployed RESTful APIs serving 1M+ requests/day",
        "Reduced system latency by 40% through query optimization"
      ]
    }}
  ],
  "education": [
    {{
      "degree": "B.Tech - Computer Science",
      "institution": "MIT",
      "year": "2017 – 2021"
    }}
  ],
  "projects": [
    {{
      "name": "Smart Inventory System",
      "description": "Built an automated inventory tracking tool using IoT sensors and Python.",
      "tech": ["Python", "MQTT", "PostgreSQL", "Docker"]
    }}
  ],
  "certifications": ["AWS Certified Developer – Associate"]
}}
---

RULES (strictly follow these):
1. If a field is missing from the resume, set it to null. Do NOT guess or assume.
2. Do NOT mix projects with experiences. Experience = paid roles at companies. Projects = personal/academic/side work.
3. Only include a project if it has both a name AND a description. If description is missing, exclude the project entirely.
4. Responsibilities must be extracted as-is from the resume. Do not rephrase or summarize them.
5. Return ONLY the JSON. No explanation, no preamble, no markdown code block, no extra text.
6. Dates must be in DD/MM/YYYY format wherever possible. If only year is available, use YYYY.
"""


def load_json_file(path, default=None):
    if default is None:
        default = {}

    if not os.path.exists(path):
        return default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_resume_chunks_dict(rows):
    all_resume_chunks = defaultdict(list)

    for row in rows or []:
        filename = row.get("resume_filename")
        if not filename:
            continue

        chunk = row.get("chunk_json") or {
            "type": row.get("chunk_type"),
            "content": row.get("content", ""),
        }
        all_resume_chunks[filename].append((row.get("chunk_index", 0), chunk))

    return {
        filename: [chunk for _, chunk in sorted(chunks, key=lambda item: item[0])]
        for filename, chunks in all_resume_chunks.items()
    }


def save_json_file(path, data):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def extract_json_from_llm_response(response_text):
    response_text = response_text.strip()

    if "```json" in response_text:
        json_str = response_text.split("```json")[-1].split("```")[0].strip()
    elif "```" in response_text:
        json_str = response_text.split("```")[1].strip()
    elif "</think>" in response_text:
        json_str = response_text.split("</think>")[-1].strip()
    else:
        json_str = response_text

    return json.loads(clean_json_str(json_str))


def get_json(file_path, llm, template=RESUME_EXTRACTION_TEMPLATE):
    from langchain_community.document_loaders import PyMuPDFLoader

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    extracted_text = "\n".join(doc.page_content for doc in documents)

    formatted_prompt = template.replace("{extracted_text}", extracted_text) + "\n/no_think"
    response = llm.invoke(formatted_prompt).content

    try:
        return extract_json_from_llm_response(response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse resume JSON for {os.path.basename(file_path)}: {e}")
        return None


def process_single_resume(file_path, llm, output_folder=RESUME_OUTPUT_DIR, force=False):
    """
    Parse one PDF resume and add it to all_resumes.json.
    Existing filenames are skipped unless force=True.
    """

    os.makedirs(output_folder, exist_ok=True)

    pdf_file = os.path.basename(file_path)
    combined_output = os.path.join(output_folder, "all_resumes.json")
    all_results = load_json_file(combined_output, {})

    if pdf_file in all_results and not force:
        return {
            "status": "skipped",
            "message": f"{pdf_file} is already processed.",
            "filename": pdf_file,
            "data": all_results[pdf_file]
        }

    parsed = get_json(file_path, llm)

    if not parsed:
        return {
            "status": "failed",
            "message": f"Could not parse {pdf_file}.",
            "filename": pdf_file,
            "data": None
        }

    output_file = os.path.join(output_folder, pdf_file.replace(".pdf", ".json"))
    save_json_file(output_file, parsed)

    all_results[pdf_file] = parsed
    save_json_file(combined_output, all_results)

    return {
        "status": "processed",
        "message": f"{pdf_file} processed and added to all_resumes.json.",
        "filename": pdf_file,
        "data": parsed
    }


def process_multiple_resumes(folder_path, llm, output_folder=RESUME_OUTPUT_DIR):
    """
    Process only new PDF resumes from a folder.
    Internally reuses process_single_resume().
    """

    pdf_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("No PDF files found in the folder.")
        return {}

    results = {}
    already_processed = 0
    new_processed = 0
    failed = 0

    for pdf_file in pdf_files:
        result = process_single_resume(
            file_path=os.path.join(folder_path, pdf_file),
            llm=llm,
            output_folder=output_folder,
            force=False
        )
        results[pdf_file] = result

        if result["status"] == "skipped":
            already_processed += 1
        elif result["status"] == "processed":
            new_processed += 1
        else:
            failed += 1

    print("\nResume Parsing Summary")
    print(f"Already processed: {already_processed}")
    print(f"New resumes processed: {new_processed}")
    print(f"Failed: {failed}")

    return results


def chunk_resume(parsed_resume):
    chunks = []
    name = parsed_resume.get("name") or "Candidate"

    summary = parsed_resume.get("summary") or ""
    if summary:
        chunks.append({
            "type": "summary",
            "content": f"{name} is a professional with the following background: {summary}"
        })

    skills = parsed_resume.get("skills") or []
    if skills:
        core_skills = skills[:10]
        extra_skills = skills[10:]
        content = (
            f"{name} is proficient in the following core technologies "
            f"and tools: {', '.join(core_skills)}."
        )
        if extra_skills:
            content += f" Additional skills and expertise include: {', '.join(extra_skills)}."
        chunks.append({"type": "skills", "content": content})

    experience = parsed_resume.get("experience") or []
    if experience:
        exp_lines = []
        resp_lines = []

        for exp in experience:
            company = exp.get("company") or "Unknown Company"
            title = exp.get("title") or "Unknown Title"
            start = exp.get("start_date") or ""
            end = exp.get("end_date") or "Present"
            responsibilities = exp.get("responsibilities") or []

            exp_lines.append(f"{name} worked as {title} at {company} from {start} to {end}.")

            for responsibility in responsibilities:
                resp_lines.append(f"- {responsibility}")

        chunks.append({"type": "experience", "content": " ".join(exp_lines)})

        if resp_lines:
            chunks.append({
                "type": "responsibilities",
                "content": (
                    f"{name} has the following professional responsibilities "
                    f"and achievements:\n" + "\n".join(resp_lines)
                )
            })

    projects = parsed_resume.get("projects") or []
    for project in projects:
        project_name = project.get("name") or "Unnamed Project"
        description = project.get("description") or ""
        tech = project.get("tech") or []

        chunks.append({
            "type": "project",
            "content": (
                f"{name} built a project called {project_name}. "
                f"{description} Technologies and tools used: {', '.join(tech)}."
            )
        })

    education = parsed_resume.get("education") or []
    if education:
        edu_lines = []
        for edu in education:
            degree = edu.get("degree") or ""
            institution = edu.get("institution") or ""
            year = edu.get("year") or ""
            edu_lines.append(f"{name} studied {degree} at {institution} ({year}).")
        chunks.append({"type": "education", "content": " ".join(edu_lines)})

    certifications = parsed_resume.get("certifications") or []
    if certifications:
        chunks.append({
            "type": "qualifications",
            "content": (
                f"{name} has demonstrated professional qualifications through "
                f"the following certifications and credentials: {', '.join(certifications)}."
            )
        })

    contacts = parsed_resume.get("contact") or {}
    if contacts:
        email = contacts.get("email") or ""
        phone = contacts.get("phone") or ""
        location = contacts.get("location") or ""
        linkedin = contacts.get("linkedin") or ""

        chunks.append({
            "type": "contacts",
            "content": (
                f"{name} can be contacted at {email} or {phone}. "
                f"Located in {location}. LinkedIn profile: {linkedin}."
            )
        })

    return chunks


def chunk_single_resume(filename, resume_data, all_chunks_path=ALL_RESUME_CHUNKS_PATH, force=False):
    all_resume_chunks = load_json_file(all_chunks_path, {})

    if filename in all_resume_chunks and not force:
        return all_resume_chunks, {}, {
            "status": "skipped",
            "message": f"{filename} is already chunked."
        }

    chunks = chunk_resume(resume_data)
    if not chunks:
        return all_resume_chunks, {}, {
            "status": "failed",
            "message": f"No chunks created for {filename}."
        }

    all_resume_chunks[filename] = chunks
    save_json_file(all_chunks_path, all_resume_chunks)

    return all_resume_chunks, {filename: chunks}, {
        "status": "chunked",
        "message": f"{filename} chunked into {len(chunks)} chunks."
    }


def chunk_new_resumes(all_resumes_path=ALL_RESUME_PATH, all_chunks_path=ALL_RESUME_CHUNKS_PATH):
    all_resumes = load_json_file(all_resumes_path, {})
    all_resume_chunks = load_json_file(all_chunks_path, {})
    new_resume_chunks = {}

    for filename, resume_data in all_resumes.items():
        if filename in all_resume_chunks:
            continue

        chunks = chunk_resume(resume_data)
        if chunks:
            all_resume_chunks[filename] = chunks
            new_resume_chunks[filename] = chunks

    save_json_file(all_chunks_path, all_resume_chunks)
    return all_resume_chunks, new_resume_chunks


def load_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=RESUME_EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def generate_embeddings(all_chunks, embedding_model=None):
    if embedding_model is None:
        embedding_model = load_embedding_model()

    all_embedded_chunks = {}

    for filename, chunks in all_chunks.items():
        embedded = []
        for chunk in chunks:
            content = chunk["content"]
            embedded.append({
                "type": chunk["type"],
                "content": content,
                "embedding": embedding_model.embed_query(content)
            })

        all_embedded_chunks[filename] = embedded
        print(f"Embedded: {filename} -> {len(embedded)} chunks")

    return all_embedded_chunks


def embeddings_have_expected_dimension(embedded_chunks, expected_dim=EXPECTED_RESUME_EMBEDDING_DIM):
    if not embedded_chunks:
        return False

    for chunk in embedded_chunks:
        embedding = chunk.get("embedding") or []
        if len(embedding) != expected_dim:
            return False

    return True


def embed_new_resume_chunks(
    new_chunks,
    all_embeddings_path=ALL_RESUME_EMBEDDINGS_PATH,
    embedding_model=None,
    force=False,
):
    all_resume_embeddings = load_json_file(all_embeddings_path, {})
    new_resume_embeddings = {}

    if not new_chunks:
        return all_resume_embeddings, new_resume_embeddings

    if embedding_model is None:
        embedding_model = load_embedding_model()

    for filename, chunks in new_chunks.items():
        existing_embeddings = all_resume_embeddings.get(filename)
        existing_is_valid = embeddings_have_expected_dimension(existing_embeddings)

        if existing_embeddings and existing_is_valid and not force:
            continue

        embedded = generate_embeddings({filename: chunks}, embedding_model=embedding_model)[filename]
        all_resume_embeddings[filename] = embedded
        new_resume_embeddings[filename] = embedded

    save_json_file(all_embeddings_path, all_resume_embeddings)
    return all_resume_embeddings, new_resume_embeddings


def get_stored_files(collection):
    results = collection._collection.get(include=["metadatas"])
    metadatas = results.get("metadatas") or []

    return {
        metadata.get("source_file")
        for metadata in metadatas
        if metadata and metadata.get("source_file")
    }


def filter_new_embeddings(embeddings_dict, collection):
    already_stored = get_stored_files(collection)
    new_embeddings = {}
    skipped_files = []

    for filename, chunks in embeddings_dict.items():
        if filename in already_stored:
            skipped_files.append(filename)
        else:
            new_embeddings[filename] = chunks

    return new_embeddings, skipped_files


def store_embeddings(embeddings_dict, collection, label="chunks"):
    new_embeddings, skipped_files = filter_new_embeddings(embeddings_dict, collection)

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for filename, chunks in new_embeddings.items():
        for i, chunk in enumerate(chunks):
            ids.append(f"{filename}_chunk_{i}")
            embeddings.append(chunk["embedding"])
            documents.append(chunk["content"])
            metadatas.append({
                "source_file": filename,
                "type": chunk["type"],
                "chunk_index": i
            })

    if ids:
        collection._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    return {
        "stored_files": list(new_embeddings.keys()),
        "skipped_files": skipped_files,
        "stored_chunks": len(ids),
        "label": label
    }


def process_uploaded_resume(file_path, llm, resume_collection, embedding_model=None, force=False):
    """
    End-to-end helper for UI uploads:
    PDF -> all_resumes.json -> all_resume_chunks.json -> all_resume_embeddings.json -> ChromaDB.
    Older resumes are skipped at each stage by filename/source_file.
    """

    parse_result = process_single_resume(file_path, llm, force=force)
    filename = parse_result["filename"]

    if parse_result["status"] == "failed":
        return {"parse": parse_result}

    all_resumes = load_json_file(ALL_RESUME_PATH, {})
    resume_data = all_resumes.get(filename) or parse_result.get("data")

    _, new_chunks, chunk_result = chunk_single_resume(
        filename,
        resume_data,
        force=force
    )

    stored_files = get_stored_files(resume_collection)
    all_chunks = load_json_file(ALL_RESUME_CHUNKS_PATH, {})
    all_embeddings = load_json_file(ALL_RESUME_EMBEDDINGS_PATH, {})
    existing_embeddings = all_embeddings.get(filename)

    needs_vector_store = filename not in stored_files
    needs_embedding_repair = not embeddings_have_expected_dimension(existing_embeddings)

    if not new_chunks and (needs_vector_store or needs_embedding_repair):
        chunks = all_chunks.get(filename)
        if chunks:
            new_chunks = {filename: chunks}

    _, new_embeddings = embed_new_resume_chunks(
        new_chunks,
        embedding_model=embedding_model,
        force=force or needs_embedding_repair,
    )

    embeddings_to_store = new_embeddings
    if needs_vector_store and filename not in embeddings_to_store:
        all_embeddings = load_json_file(ALL_RESUME_EMBEDDINGS_PATH, {})
        existing_embeddings = all_embeddings.get(filename)
        if embeddings_have_expected_dimension(existing_embeddings):
            embeddings_to_store = {filename: existing_embeddings}

    store_result = store_embeddings(
        embeddings_to_store,
        resume_collection,
        label="resume chunks"
    )

    return {
        "parse": parse_result,
        "chunk": chunk_result,
        "embedding_files": list(new_embeddings.keys()),
        "vector_store": store_result
    }


def process_uploaded_resume_for_session(file_path, llm, embedding_model=None):
    """
    Parse, chunk, and embed one uploaded resume for the current app session only.
    This does not write to local JSON files, Supabase, or Chroma Cloud.
    """

    filename = os.path.basename(file_path)
    resume_data = get_json(file_path, llm)

    if not resume_data:
        return {
            "status": "failed",
            "message": f"Could not parse {filename}.",
            "filename": filename,
            "resume": None,
            "chunks": [],
            "embedded_chunks": [],
        }

    chunks = chunk_resume(resume_data)
    embedded_chunks = generate_embeddings(
        {filename: chunks},
        embedding_model=embedding_model,
    )[filename] if chunks else []

    return {
        "status": "processed",
        "message": f"{filename} processed for this session.",
        "filename": filename,
        "resume": resume_data,
        "chunks": chunks,
        "embedded_chunks": embedded_chunks,
    }


# STEP 1 - Fetch JD chunks from ChromaDB

def get_jd_chunks(jd_filename, jd_collection):
    """
    Fetch all chunks of a selected JD from ChromaDB.
    Each chunk gets a unique jd_chunk_id used in aggregation
    to track which hiring requirement each match came from.
    """
    results = jd_collection._collection.get(
        where   = {"source_file": jd_filename},
        include = ["embeddings", "documents", "metadatas"]
    )

    if not results["ids"]:
        print(f"No chunks found for JD: {jd_filename}")
        return []

    chunks = []
    for i in range(len(results["ids"])):
        chunk_type = results["metadatas"][i]["type"]
        chunks.append({
            "jd_chunk_id"  : f"{jd_filename}__{chunk_type}__{i}",
            # Unique per JD chunk - ensures same type appearing
            # twice (e.g. two skills chunks) are tracked separately
            "type"         : chunk_type,
            "content"      : results["documents"][i],
            "embedding"    : results["embeddings"][i]
        })

    print(f"Fetched {len(chunks)} chunks for JD: {jd_filename}")
    return chunks


# STEP 2 - Query resume collection with vector search + BM25 keyword search

def tokenize_for_bm25(text):
    """Tokenize text for BM25 keyword matching."""
    if not text:
        return []
    return re.findall(r"[a-zA-Z0-9+#.]+", text.lower())


def load_resume_chunks_for_bm25(chunks_path=ALL_RESUME_CHUNKS_PATH,
                                all_resume_chunks=None,
                                session_resume=None):
    """Build an in-memory BM25 index over saved resume chunks."""
    if BM25Okapi is None:
        print("rank_bm25 is not installed. Falling back to vector-only search.")
        return None, None

    if all_resume_chunks is None:
        all_resume_chunks = load_json_file(chunks_path, {})

    if session_resume and session_resume.get("chunks"):
        all_resume_chunks = dict(all_resume_chunks)
        all_resume_chunks[session_resume["filename"]] = session_resume["chunks"]

    documents = []

    for filename, chunks in all_resume_chunks.items():
        for i, chunk in enumerate(chunks):
            documents.append({
                "source_file": filename,
                "chunk_type": chunk.get("type"),
                "chunk_index": i,
                "content": chunk.get("content", ""),
            })

    if not documents:
        return None, None

    tokenized_corpus = [tokenize_for_bm25(doc["content"]) for doc in documents]
    bm25_index = BM25Okapi(tokenized_corpus)
    print(f"BM25 index built with {len(documents)} resume chunks.")
    return bm25_index, documents


def normalize_bm25_scores(raw_scores):
    """Normalize BM25 scores to a 0-1 scale for blending with vector similarity."""
    if raw_scores is None:
        return []

    raw_scores = list(raw_scores)
    if not raw_scores:
        return []

    min_score = min(raw_scores)
    max_score = max(raw_scores)

    if max_score == min_score:
        return [0.0 for _ in raw_scores]

    return [
        (score - min_score) / (max_score - min_score)
        for score in raw_scores
    ]


def search_bm25_resume_chunks(jd_chunk, bm25_index, bm25_documents):
    """Return BM25 scores for every resume chunk for one JD chunk."""
    if bm25_index is None or bm25_documents is None:
        return {}

    query_tokens = tokenize_for_bm25(jd_chunk["content"])
    raw_scores = bm25_index.get_scores(query_tokens)
    normalized_scores = normalize_bm25_scores(raw_scores)

    bm25_results = {}
    for doc, raw_score, norm_score in zip(bm25_documents, raw_scores, normalized_scores):
        key = (doc["source_file"], doc["chunk_index"])
        bm25_results[key] = {
            "bm25_score": round(float(raw_score), 4),
            "bm25_score_norm": round(float(norm_score), 4),
        }

    return bm25_results


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if not norm_a or not norm_b:
        return 0.0

    return dot / (norm_a * norm_b)


def search_session_resume_chunks(jd_chunk, session_resume, top_k):
    if not session_resume or not session_resume.get("embedded_chunks"):
        return []

    matches = []
    filename = session_resume["filename"]

    for chunk_index, chunk in enumerate(session_resume["embedded_chunks"]):
        similarity = cosine_similarity(
            jd_chunk["embedding"],
            chunk.get("embedding") or [],
        )
        matches.append({
            "source_file": filename,
            "chunk_type": chunk.get("type"),
            "chunk_index": chunk_index,
            "similarity": round(similarity, 4),
            "distance": round(1 - similarity, 4),
        })

    return sorted(
        matches,
        key=lambda item: item["similarity"],
        reverse=True,
    )[:top_k]


def search_similar_resumes(jd_chunk, resume_collection, top_k,
                           bm25_index=None, bm25_documents=None,
                           session_resume=None):
    """
    Query resume chunks using Chroma vector similarity, then blend each match
    with its BM25 lexical score for the same JD chunk.
    """
    bm25_results = search_bm25_resume_chunks(
        jd_chunk, bm25_index, bm25_documents
    )

    matches = []
    vector_matches = []

    if top_k > 0:
        results = resume_collection._collection.query(
            query_embeddings=[jd_chunk["embedding"]],
            n_results=top_k,
            include=["metadatas", "distances"]
        )

        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            vector_matches.append({
                "metadata": metadata,
                "distance": distance,
                "similarity": round(1 - distance, 4),
            })

    for session_match in search_session_resume_chunks(jd_chunk, session_resume, top_k=1):
        vector_matches.append({
            "metadata": {
                "source_file": session_match["source_file"],
                "type": session_match["chunk_type"],
                "chunk_index": session_match["chunk_index"],
            },
            "distance": session_match["distance"],
            "similarity": session_match["similarity"],
        })

    for vector_match in vector_matches:
        metadata = vector_match["metadata"]
        vector_similarity = vector_match["similarity"]

        source_file = metadata["source_file"]
        chunk_index = metadata.get("chunk_index")
        bm25_match = bm25_results.get((source_file, chunk_index), {})
        bm25_score = bm25_match.get("bm25_score", 0.0)
        bm25_score_norm = bm25_match.get("bm25_score_norm", 0.0)

        hybrid_score = round(
            (VECTOR_CHUNK_WEIGHT * vector_similarity) +
            (BM25_CHUNK_WEIGHT * bm25_score_norm),
            4
        )

        matches.append({
            "source_file": source_file,
            "chunk_type": metadata["type"],
            "chunk_index": chunk_index,
            "similarity": vector_similarity,
            "bm25_score": bm25_score,
            "bm25_score_norm": bm25_score_norm,
            "hybrid_score": hybrid_score,
            "jd_chunk_id": jd_chunk["jd_chunk_id"],
            "jd_chunk_type": jd_chunk["type"]
        })

    return matches


# STEP 3 - JD-Centric Aggregation

def aggregate_scores(all_matches):
    """
    For each candidate and each JD chunk, keep the best matching resume chunk.
    Final relevance score is a weighted average of hybrid vector/BM25 scores
    across JD requirements.
    """
    candidate_jd_scores = defaultdict(dict)

    for match in all_matches:
        filename = match["source_file"]
        jd_chunk_id = match["jd_chunk_id"]
        hybrid_score = match["hybrid_score"]

        current_best = candidate_jd_scores[filename].get(jd_chunk_id)
        if current_best is None or hybrid_score > current_best["hybrid_score"]:
            candidate_jd_scores[filename][jd_chunk_id] = {
                "hybrid_score": hybrid_score,
                "similarity": match["similarity"],
                "bm25_score": match.get("bm25_score", 0.0),
                "bm25_score_norm": match.get("bm25_score_norm", 0.0),
                "resume_chunk_type": match["chunk_type"],
                "jd_chunk_type": match["jd_chunk_type"]
            }

    relevance_scores = {}

    for filename, jd_scores in candidate_jd_scores.items():
        weighted_sum = 0.0
        total_weight = 0.0
        chunk_scores = []

        for jd_chunk_id, v in jd_scores.items():
            jd_type = v["jd_chunk_type"]
            hybrid_score = v["hybrid_score"]
            weight = JD_CHUNK_WEIGHTS.get(jd_type, 0.05)

            weighted_sum += hybrid_score * weight
            total_weight += weight

            chunk_scores.append({
                "jd_chunk_id": jd_chunk_id,
                "jd_chunk_type": jd_type,
                "resume_chunk_type": v["resume_chunk_type"],
                "similarity": v["similarity"],
                "bm25_score": v["bm25_score"],
                "bm25_score_norm": v["bm25_score_norm"],
                "hybrid_score": hybrid_score,
                "weight": weight,
                "contribution": round(hybrid_score * weight, 4)
            })

        final_score = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0

        relevance_scores[filename] = {
            "final_score": final_score,
            "chunk_scores": sorted(
                chunk_scores,
                key=lambda x: x["hybrid_score"],
                reverse=True
            )
        }

    return relevance_scores


# STEP 4 - Experience penalty (low impact)
CURRENT_DATE_LABELS = {
    "present", "current", "currently", "now",
    "ongoing", "till date", "to date"
}


def parse_date(date_str, as_of=None, prefer_end=False):
    if date_str is None or not str(date_str).strip():
        return None

    as_of = as_of or datetime.today()
    normalized = re.sub(r"\s+", " ", str(date_str).strip())
    normalized_lower = normalized.lower().rstrip(".")

    if normalized_lower in CURRENT_DATE_LABELS:
        return as_of

    if prefer_end and any(
        re.search(rf"\b{re.escape(label)}\b", normalized_lower)
        for label in CURRENT_DATE_LABELS
    ):
        return as_of

    normalized = normalized.replace(",", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    formats = (
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%m/%Y",
        "%m-%Y",
        "%b %Y",
        "%B %Y",
        "%Y",
    )

    for fmt in formats:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            pass

    # Handles a complete range stored in one field:
    # "Jul, 2021 - Present"
    date_tokens = re.findall(
        r"\b(?:"
        r"\d{4}-\d{1,2}-\d{1,2}|"
        r"\d{1,2}/\d{1,2}/\d{4}|"
        r"\d{1,2}[/-]\d{4}|"
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
        r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,.]?\s+\d{4}|"
        r"\d{4}"
        r")\b",
        normalized,
        flags=re.IGNORECASE,
    )

    if date_tokens:
        token = date_tokens[-1] if prefer_end else date_tokens[0]
        token = token.replace(",", " ").replace(".", " ")
        token = re.sub(r"\s+", " ", token).strip()

        for fmt in formats:
            try:
                return datetime.strptime(token, fmt)
            except ValueError:
                pass

    return None


def _month_index(value):
    return value.year * 12 + value.month - 1


def compute_experience_years(experience_list, as_of=None):
    as_of = as_of or datetime.today()
    intervals = []

    for exp in experience_list or []:
        start_value = exp.get("start_date")
        end_value = exp.get("end_date")

        start = parse_date(start_value, as_of=as_of)

        if end_value:
            end = parse_date(
                end_value,
                as_of=as_of,
                prefer_end=True,
            )
        elif re.search(r"\s*[-–—]\s*", str(start_value or "")):
            end = parse_date(
                start_value,
                as_of=as_of,
                prefer_end=True,
            )
        else:
            end = as_of

        # Recover dates if the parser placed them in another field.
        if not start:
            exp_text = " ".join(
                str(value)
                for value in exp.values()
                if value is not None
            )
            start = parse_date(exp_text, as_of=as_of)
            end = (
                parse_date(
                    exp_text,
                    as_of=as_of,
                    prefer_end=True,
                )
                or end
            )

        if not start or not end or end < start:
            continue

        intervals.append(
            (
                _month_index(start),
                _month_index(min(end, as_of)),
            )
        )

    if not intervals:
        return 0.0

    # Merge adjacent and overlapping roles.
    merged = []

    for start_month, end_month in sorted(intervals):
        if not merged or start_month > merged[-1][1] + 1:
            merged.append([start_month, end_month])
        else:
            merged[-1][1] = max(
                merged[-1][1],
                end_month,
            )

    total_months = sum(
        end_month - start_month
        for start_month, end_month in merged
    )

    return round(total_months / 12, 1)


def extract_jd_min_years(jd_parsed):
    """
    Extract minimum years required from JD.
    Handles: "3-5+ years", "3+ years", "5 years"
    Regex finds the first number it sees in experience_required.
    """
    exp_str = jd_parsed.get("experience_required") or ""
    match   = re.search(r"(\d+)", exp_str)
    return float(match.group(1)) if match else 0.0


def compute_experience_penalty(resume_years, jd_min_years):
    """
    Soft penalty - low impact by design (max 10% reduction).
    Meets requirement  -> 1.00 (no penalty)
    Within 1 year short-> 0.95 (5% penalty)
    More than 1yr short-> 0.90 (10% penalty)
    """
    if jd_min_years == 0 or resume_years >= jd_min_years:
        return 1.0
    gap = jd_min_years - resume_years
    return 0.95 if gap <= 1.0 else 0.90

def get_text(value):
    if isinstance(value, list):
        return " ".join(get_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(get_text(v) for v in value.values())
    return str(value or "")


def compute_role_fit_score(jd_parsed, resume_parsed):
    """
    Estimate whether the resume profile fits the JD role family.
    This complements semantic relevance by rewarding role-specific evidence.
    """
    jd_text = get_text(jd_parsed).lower()
    resume_text = get_text(resume_parsed).lower()

    if "software" in jd_text or "backend" in jd_text or "frontend" in jd_text:
        role_groups = {
            "software_title": [
                "software engineer", "software developer", "backend developer",
                "full stack", "full-stack", "developer", "system engineer"
            ],
            "backend": [
                "python", "java", "node", "spring boot", ".net", "c#"
            ],
            "api_microservices": [
                "api", "rest", "microservice", "microservices", "web api"
            ],
            "frontend": [
                "react", "angular", "javascript", "typescript"
            ],
            "database": [
                "sql", "mysql", "postgresql", "mongodb", "dynamodb",
                "nosql", "relational database"
            ],
            "cloud_devops": [
                "aws", "ci/cd", "jenkins", "docker", "kubernetes",
                "gitlab", "github actions"
            ],
            "ai_systems": [
                "llm", "genai", "agent", "agents", "mcp",
                "langchain", "bedrock", "vector db"
            ]
        }
    elif "data scientist" in jd_text or "machine learning" in jd_text:
        role_groups = {
            "data_title": [
                "data scientist", "machine learning", "ml engineer", "data analyst"
            ],
            "programming": ["python", "sql", "pyspark"],
            "ml": ["machine learning", "predictive", "regression", "classification"],
            "analytics": ["eda", "analysis", "visualization", "tableau"],
            "statistics": ["a/b testing", "hypothesis", "statistics"],
            "ai": ["llm", "genai", "langchain", "agents"]
        }
    elif "product" in jd_text:
        role_groups = {
            "product_title": ["product owner", "product manager", "business analyst"],
            "product": ["roadmap", "backlog", "prd", "brd", "mvp"],
            "stakeholder": ["stakeholder", "cross-functional", "scrum", "agile"],
            "analytics": ["sql", "dashboard", "metrics", "experimentation"],
            "ai_product": ["ai", "llm", "genai"]
        }
    else:
        return 0.5, ["generic_role"]

    matched_groups = []
    for group_name, keywords in role_groups.items():
        if any(keyword in resume_text for keyword in keywords):
            matched_groups.append(group_name)

    role_fit_score = len(matched_groups) / len(role_groups)
    return round(role_fit_score, 4), matched_groups


def compute_experience_fit_score(resume_years, jd_min_years):
    if jd_min_years == 0:
        return 1.0
    return round(min(resume_years / jd_min_years, 1.0), 4)


# STEP 6 - Full pipeline

def get_resume_chunk_count(resume_collection):
    """
    Return the number of resume chunks currently stored in ChromaDB.
    This is used as dynamic top_k for JD-to-resume chunk search.
    """
    return resume_collection._collection.count()

def get_top_candidates(jd_filename, resume_collection,
                       jd_collection, top_n=3, top_k=60,
                       all_jds=None, all_resumes=None,
                       all_resume_chunks=None,
                       session_resume=None):
    """
    Full pipeline - given a JD filename, return top N candidates.
    Final score uses hybrid vector/BM25 relevance and a small experience penalty.
    """

    jd_chunks = get_jd_chunks(jd_filename, jd_collection)
    if not jd_chunks:
        return []

    bm25_index, bm25_documents = load_resume_chunks_for_bm25(
        all_resume_chunks=all_resume_chunks,
        session_resume=session_resume,
    )

    all_matches = []
    for jd_chunk in jd_chunks:
        matches = search_similar_resumes(
            jd_chunk=jd_chunk,
            resume_collection=resume_collection,
            top_k=top_k,
            bm25_index=bm25_index,
            bm25_documents=bm25_documents,
            session_resume=session_resume,
        )
        all_matches.extend(matches)

    print(f"Total matches: {len(all_matches)} "
          f"({len(jd_chunks)} JD chunks x top_k={top_k})")

    relevance_scores = aggregate_scores(all_matches)

    if all_jds is None:
        with open(ALL_JD_PATH, "r", encoding="utf-8") as f:
            all_jds = json.load(f)

    if all_resumes is None:
        with open(ALL_RESUME_PATH, "r", encoding="utf-8") as f:
            all_resumes = json.load(f)

    if session_resume and session_resume.get("resume"):
        all_resumes = dict(all_resumes)
        all_resumes[session_resume["filename"]] = session_resume["resume"]

    jd_parsed = all_jds.get(jd_filename, {})
    if not jd_parsed:
        print(f"JD not found in all_jd.json: {jd_filename}")
        return []

    jd_min_years = extract_jd_min_years(jd_parsed)
    print(f"JD requires minimum {jd_min_years} years experience")

    combined_scores = {}

    for filename, score_data in relevance_scores.items():
        resume_parsed = all_resumes.get(filename, {})

        if not resume_parsed:
            print(f"Resume not found: {filename}")
            continue

        relevance_score = score_data["final_score"]
        resume_years = compute_experience_years(
            resume_parsed.get("experience", [])
        )
        role_fit_score, matched_role_groups = compute_role_fit_score(
            jd_parsed,
            resume_parsed
        )
        experience_fit_score = compute_experience_fit_score(
            resume_years,
            jd_min_years
        )
        exp_penalty = compute_experience_penalty(resume_years, jd_min_years)

        final_score = round(
            (RELEVANCE_WEIGHT * relevance_score) +
            (ROLE_FIT_WEIGHT * role_fit_score) +
            (EXPERIENCE_FIT_WEIGHT * experience_fit_score),
            4
        )

        combined_scores[filename] = {
            "final_score": final_score,
            "percentage": f"{round(final_score * 100, 2)}%",
            "relevance_score": f"{round(relevance_score * 100, 2)}%",
            "role_fit_score": f"{round(role_fit_score * 100, 2)}%",
            "experience_fit_score": f"{round(experience_fit_score * 100, 2)}%",
            "resume_years": resume_years,
            "required_years": jd_min_years,
            "exp_penalty": exp_penalty,
            "matched_role_groups": matched_role_groups,
            "chunk_scores": score_data["chunk_scores"]
        }

    ranked = sorted(
        combined_scores.items(),
        key=lambda x: x[1]["final_score"],
        reverse=True
    )

    print(f"\n Top {top_n} Candidates for JD: {jd_filename}")
    print(" Score = 70% relevance + 20% role fit + 10% experience fit")
    print(f" Hybrid chunk score = {int(VECTOR_CHUNK_WEIGHT * 100)}% vector + "
          f"{int(BM25_CHUNK_WEIGHT * 100)}% BM25")
    print("=" * 75)

    for rank, (filename, scores) in enumerate(ranked[:top_n], start=1):
        penalty_note = (
            f"(penalty: {scores['exp_penalty']})"
            if scores["exp_penalty"] < 1.0 else "(no penalty)"
        )
        print(f"\n#{rank} - {filename}")
        print(f"     Final Score     : {scores['percentage']}")
        print(f"     Relevance Score : {scores['relevance_score']}")
        print(f"     Role Fit        : {scores['role_fit_score']}")
        print(f"     Experience Fit  : {scores['experience_fit_score']}")
        print(f"     Experience      : {scores['resume_years']} yrs  {penalty_note}")
        print(f"     Matched Groups  : {', '.join(scores['matched_role_groups'])}")
        print("\n     JD Requirement Breakdown (JD-Centric):")
        print(f"       {'JD Requirement':<20} {'Resume Section':<20} {'Vector':>8} {'BM25':>8} {'Hybrid':>8}")
        print(f"       {'-' * 70}")
        for chunk in scores["chunk_scores"]:
            print(f"       {chunk['jd_chunk_type']:<20} "
                  f"{chunk['resume_chunk_type']:<20} "
                  f"{chunk['similarity']:>8.4f} "
                  f"{chunk['bm25_score_norm']:>8.4f} "
                  f"{chunk['hybrid_score']:>8.4f}")

    print(f"\n Full Ranking (all {len(ranked)} candidates):")
    print("-" * 80)
    for rank, (filename, scores) in enumerate(ranked, start=1):
        bar_len = int(scores["final_score"] * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)
        print(f"  #{rank:<3} {filename:<40} {scores['percentage']:>7}  "
              f"[rel: {scores['relevance_score']} | "
              f"role: {scores['role_fit_score']} | "
              f"exp: {scores['experience_fit_score']}]  {bar}")

    return ranked[:top_n]

def build_compact_resume(resume_parsed):
    experience = resume_parsed.get("experience") or []
    compact_exp = []
    for exp in experience:
        compact_exp.append({
            "company"         : exp.get("company"),
            "title"           : exp.get("title"),
            "start_date"      : exp.get("start_date"),
            "end_date"        : exp.get("end_date"),
            "responsibilities": exp.get("responsibilities", [])[:3]
        })

    return {
        "name"          : resume_parsed.get("name"),
        "summary"       : resume_parsed.get("summary"),
        "skills"        : resume_parsed.get("skills"),
        "experience"    : compact_exp,
        "education"     : resume_parsed.get("education"),
        "certifications": resume_parsed.get("certifications")
    }


def build_compact_jd(jd_parsed):
    return {
        "job_title"          : jd_parsed.get("job_title"),
        "experience_required": jd_parsed.get("experience_required"),
        "skills_required"    : jd_parsed.get("skills_required"),
        "job_summary"        : jd_parsed.get("job_summary")
    }


def clean_json_str(json_str):
    # Remove trailing commas before } or ]
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    return json_str


def normalize_summary_experience(summary, resume_years):
    """
    Replace stale LLM experience claims with the deterministic experience
    calculated from resume employment dates.
    """
    if not isinstance(summary, dict):
        return summary

    years_text = f"{resume_years} years of experience"
    experience_pattern = re.compile(
        r"\b\d+(?:\.\d+)?\s*\+?\s*years?(?:'|’)?(?:\s+of)?\s+experience\b",
        flags=re.IGNORECASE,
    )

    def normalize_value(value):
        if isinstance(value, str):
            return experience_pattern.sub(years_text, value)
        if isinstance(value, list):
            return [normalize_value(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize_value(item) for key, item in value.items()}
        return value

    return normalize_value(summary)


def generate_candidate_summary(candidate_filename, jd_filename,
                                all_resumes, all_jds, llm,
                                scores):
    """
    Generates a structured summary for a single candidate
    against a specific JD using LLM.
    """
    resume_parsed = all_resumes.get(candidate_filename, {})
    jd_parsed     = all_jds.get(jd_filename, {})

    if not resume_parsed or not jd_parsed:
        print(f"Missing data for {candidate_filename}")
        return None

    # Build context for LLM
    summary_template = """
You are an expert Recruiter reviewing a candidate resume against a Job Description.

JOB DESCRIPTION:
{jd_json}

CANDIDATE RESUME:
{resume_json}

MATCH SCORES:
- Overall Match   : {final_score}
- Relevance Score : {relevance_score}
- Role Fit Score  : {role_fit_score}
- Experience Fit  : {experience_fit_score}
- Experience      : {resume_years} years (required: {required_years} years)

Based on the above, generate a structured evaluation in the following JSON format:
{
    "candidate_name"     : string,
    "match_summary"      : string (2-3 sentences overview of why this candidate fits or doesnt fit),
    "key_strengths"      : [string] (top 1-2 reasons this candidate is suitable),
    "skill_gaps"         : [string] (Only missing atomic JD capabilities.
  Do not include broad grouped requirements.
  Do not include capabilities with direct, equivalent, adjacent, or transferable resume evidence.
),
    "experience_summary" : string (brief summary of relevant experience),
    "recommendation"     : string (1-2 sentences final recommendation for the recruiter)
}

Rules:
1. Return ONLY valid JSON. No preamble, no explanation, no markdown blocks.
2. Be specific - reference actual skills, projects, companies from the resume.
3. Do not make anything up. Only use information present in the resume.
4. Keep each string concise and professional.
5. Dont include Company name anywhere in the response.
6. Evaluate JD requirements lexically and semantically.Consider demonstrated outcomes, projects, responsibilities, and equivalent workflows as evidence even when the JD uses different terminology.
7. Before listing a gap, search all resume fields—summary, skills, experience, projects, responsibilities, technologies, and certifications—for direct or semantically equivalent evidence

Example:
JD requirement: "Experience with Tool A, Tool B, workflow automation, and stakeholder communication."

Resume evidence:
- Mentions Tool A or a closely related tool from the same ecosystem.
- Shows projects where the candidate automated manual work.
- Describes presenting insights or collaborating with business teams.

Correct skill gaps:
- No explicit evidence of Tool B.

Incorrect skill gaps:
- No experience with Tool A, Tool B, workflow automation, or stakeholder communication.

Reason:
Do not copy the full JD requirement as a gap. Split it into atomic capabilities, remove anything supported by direct, equivalent, adjacent, or transferable resume evidence, and return only the truly missing capability.
"""

    formatted_prompt = summary_template \
        .replace("{jd_json}",         json.dumps(jd_parsed, indent=2)) \
        .replace("{resume_json}",     json.dumps(resume_parsed, indent=2)) \
        .replace("{final_score}",     scores["percentage"]) \
        .replace("{relevance_score}", scores["relevance_score"]) \
        .replace("{role_fit_score}", scores.get("role_fit_score", "N/A")) \
        .replace("{experience_fit_score}", scores.get("experience_fit_score", "N/A")) \
        .replace("{resume_years}",    str(scores["resume_years"])) \
        .replace("{required_years}",  str(scores.get("required_years", "N/A")))

    response = llm.invoke(formatted_prompt).content

    # Parse response
    if "```json" in response:
        json_str = response.split("```json")[-1].split("```")[0].strip()
    elif "```" in response:
        json_str = response.split("```")[1].strip()
    elif "</think>" in response:
        json_str = response.split("</think>")[-1].strip()
    else:
        json_str = response.strip()

    json_str = clean_json_str(json_str)

    try:
        summary = json.loads(json_str)
        return normalize_summary_experience(summary, scores["resume_years"])
    except json.JSONDecodeError as e:
        print(f"Failed to parse summary for {candidate_filename}: {e}")
        return None


def generate_all_summaries(top_candidates, jd_filename,
                           all_resumes, all_jds, llm):
    """
    Generates summaries for all top candidates.
    top_candidates -> output from get_top_candidates()
    """
    all_summaries = {}

    print(f"\nGenerating summaries for top {len(top_candidates)} candidates...")
    print("=" * 65)

    for rank, (filename, scores) in enumerate(top_candidates, start=1):
        print(f"\nGenerating summary for #{rank}: {filename}")

        summary = generate_candidate_summary(
            candidate_filename = filename,
            jd_filename        = jd_filename,
            all_resumes        = all_resumes,
            all_jds            = all_jds,
            llm                = llm,
            scores             = scores
        )

        if summary:
            all_summaries[filename] = {
                "rank"   : rank,
                "scores" : scores,
                "summary": summary
            }

            # Display summary
            print(f"\n{'='*65}")
            print(f"#{rank} - {summary.get('candidate_name', filename)}")
            print(f"{'='*65}")
            print(f"Verdict     : {summary.get('overall_verdict')}")
            print(f"Match Score : {scores['percentage']}")
            print(f"\nMatch Summary:")
            print(f"  {summary.get('match_summary')}")
            print(f"\nKey Strengths:")
            for s in (summary.get('key_strengths') or []):
                print(f"  {s}")
            print(f"\nSkill Gaps:")
            for s in (summary.get('skill_gaps') or []):
                print(f"   {s}")
            print(f"\nExperience Summary:")
            print(f"  {summary.get('experience_summary')}")
            print(f"\nRecommendation:")
            print(f"  {summary.get('recommendation')}")
        else:
            print(f" Skipped {filename} - summary generation failed")

    return all_summaries
