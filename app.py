
import streamlit as st
import requests
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="StudyMate - Academic RAG Assistant",
    page_icon="🎓",
    layout="wide"
)


# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 18px;
    margin-bottom: 30px;
}

.source-box {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-bottom: 12px;
}

.api-box {
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# LOAD RAG MODELS
# ------------------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():

    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-small"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small"
    )

    return tokenizer, model


embedding_model = load_embedding_model()

tokenizer, llm_model = load_llm()


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

@st.cache_data
def load_dataset():

    with open(
        "/content/academic_notes.txt",
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


text = load_dataset()


# ------------------------------------------------------------
# CREATE CHUNKS
# ------------------------------------------------------------

@st.cache_data
def create_chunks(text):

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []

    current = ""

    for paragraph in paragraphs:

        if len(current) + len(paragraph) <= 900:

            current += paragraph + "\n\n"

        else:

            if current.strip():
                chunks.append(current.strip())

            current = paragraph + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks


chunks = create_chunks(text)


# ------------------------------------------------------------
# CREATE FAISS INDEX
# ------------------------------------------------------------

@st.cache_resource
def create_index(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index


index = create_index(chunks)


# ------------------------------------------------------------
# RAG RETRIEVER
# ------------------------------------------------------------

def retrieve_documents(
    question,
    top_k=3
):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx >= 0:

            results.append({
                "score": float(score),
                "text": chunks[idx]
            })

    return results


# ------------------------------------------------------------
# RAG GENERATION
# ------------------------------------------------------------

def ask_rag(question):

    results = retrieve_documents(
        question,
        3
    )

    context = "\n\n".join(
        result["text"]
        for result in results
    )

    prompt = f"""
You are StudyMate, an academic assistant.

Answer the question using only the provided context.

Give a clear and concise answer.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = llm_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=120,
        num_beams=4,
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer, results


# ------------------------------------------------------------
# WIKIPEDIA API
# ------------------------------------------------------------

def wikipedia_api(topic):

    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(" ", "_")

    headers = {
        "User-Agent": "StudyMateAcademicAssistant/1.0"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        if response.status_code == 200:

            data = response.json()

            return {
                "title": data.get("title"),
                "summary": data.get("extract"),
                "url": data.get("content_urls", {})
                    .get("desktop", {})
                    .get("page")
            }

    except Exception:
        pass

    return None


# ------------------------------------------------------------
# ARXIV API
# ------------------------------------------------------------

def arxiv_api(query):

    try:

        url = "https://export.arxiv.org/api/query"

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": 3
        }

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        if response.status_code == 200:

            import feedparser

            feed = feedparser.parse(
                response.text
            )

            return [
                {
                    "title": e.get("title", "").strip(),
                    "url": e.get("link", "")
                }
                for e in feed.entries
            ]

    except Exception:
        pass

    return []


# ------------------------------------------------------------
# OPENALEX API
# ------------------------------------------------------------

def openalex_api(query):

    try:

        response = requests.get(
            "https://api.openalex.org/works",
            params={
                "search": query,
                "per-page": 3
            },
            timeout=20
        )

        if response.status_code == 200:

            data = response.json()

            return [
                {
                    "title": x.get("title"),
                    "year": x.get("publication_year"),
                    "doi": x.get("doi")
                }
                for x in data.get("results", [])
            ]

    except Exception:
        pass

    return []


# ------------------------------------------------------------
# CROSSREF API
# ------------------------------------------------------------

def crossref_api(query):

    try:

        response = requests.get(
            "https://api.crossref.org/works",
            params={
                "query": query,
                "rows": 3
            },
            timeout=20
        )

        if response.status_code == 200:

            data = response.json()

            return [
                {
                    "title": x.get("title", [""])[0],
                    "doi": x.get("DOI"),
                    "publisher": x.get("publisher")
                }
                for x in data["message"]["items"]
            ]

    except Exception:
        pass

    return []


# ------------------------------------------------------------
# OPEN LIBRARY API
# ------------------------------------------------------------

def openlibrary_api(query):

    try:

        response = requests.get(
            "https://openlibrary.org/search.json",
            params={
                "q": query,
                "limit": 3
            },
            timeout=20
        )

        if response.status_code == 200:

            data = response.json()

            return [
                {
                    "title": x.get("title"),
                    "author": x.get(
                        "author_name",
                        ["Unknown"]
                    )[0]
                }
                for x in data.get("docs", [])
            ]

    except Exception:
        pass

    return []


# ============================================================
# UI
# ============================================================

st.markdown(
    '<div class="main-title">🎓 StudyMate</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Academic Retrieval-Augmented Generation Assistant'
    '</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:

    st.header("📚 StudyMate")

    st.write(
        "AI-powered academic assistant "
        "using Retrieval-Augmented Generation."
    )

    st.divider()

    st.subheader("🔧 Technology")

    st.write("• Sentence Transformers")
    st.write("• FAISS")
    st.write("• FLAN-T5")
    st.write("• Streamlit")
    st.write("• FastAPI")

    st.divider()

    st.subheader("🌐 External APIs")

    st.write("✅ Wikipedia")
    st.write("✅ arXiv")
    st.write("✅ OpenAlex")
    st.write("✅ Crossref")
    st.write("✅ Open Library")


# ------------------------------------------------------------
# CHAT HISTORY
# ------------------------------------------------------------

if "history" not in st.session_state:

    st.session_state.history = []


# ------------------------------------------------------------
# QUESTION INPUT
# ------------------------------------------------------------

question = st.text_input(
    "💬 Enter your academic question",
    placeholder="Example: What is Retrieval-Augmented Generation?"
)


if st.button(
    "🤖 Ask StudyMate",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching knowledge base and generating answer..."
        ):

            try:

                answer, sources = ask_rag(
                    question
                )

                st.session_state.history.append({
                    "question": question,
                    "answer": answer
                })

                st.success("Answer generated!")

                # ANSWER

                st.subheader("🤖 Answer")

                st.write(answer)


                # SOURCES

                st.subheader(
                    "📖 Retrieved Sources"
                )

                for i, source in enumerate(
                    sources,
                    1
                ):

                    with st.expander(
                        f"Source {i} — "
                        f"Similarity: "
                        f"{source['score']:.4f}"
                    ):

                        st.write(
                            source["text"]
                        )


                # EXTERNAL RESEARCH

                st.subheader(
                    "🌐 External Research"
                )

                tabs = st.tabs([
                    "Wikipedia",
                    "arXiv",
                    "OpenAlex",
                    "Crossref",
                    "Open Library"
                ])


                # Wikipedia

                with tabs[0]:

                    result = wikipedia_api(
                        question
                    )

                    if result:

                        st.write(
                            "**" +
                            str(result["title"]) +
                            "**"
                        )

                        st.write(
                            result["summary"]
                        )

                        if result["url"]:

                            st.markdown(
                                f"[Open Wikipedia page]"
                                f"({result['url']})"
                            )

                    else:

                        st.info(
                            "No Wikipedia result."
                        )


                # arXiv

                with tabs[1]:

                    results = arxiv_api(
                        question
                    )

                    for paper in results:

                        st.markdown(
                            f"**{paper['title']}**"
                        )

                        st.markdown(
                            f"[View paper]"
                            f"({paper['url']})"
                        )


                # OpenAlex

                with tabs[2]:

                    results = openalex_api(
                        question
                    )

                    for item in results:

                        st.write(
                            item["title"]
                        )

                        st.write(
                            "Year:",
                            item["year"]
                        )

                        if item["doi"]:

                            st.write(
                                "DOI:",
                                item["doi"]
                            )


                # Crossref

                with tabs[3]:

                    results = crossref_api(
                        question
                    )

                    for item in results:

                        st.write(
                            item["title"]
                        )

                        st.write(
                            "Publisher:",
                            item["publisher"]
                        )

                        st.write(
                            "DOI:",
                            item["doi"]
                        )


                # Open Library

                with tabs[4]:

                    results = openlibrary_api(
                        question
                    )

                    for book in results:

                        st.write(
                            "📚",
                            book["title"]
                        )

                        st.write(
                            "Author:",
                            book["author"]
                        )


                # DOWNLOAD

                download_text = (
                    "StudyMate Answer\n\n"
                    f"Question:\n{question}\n\n"
                    f"Answer:\n{answer}\n"
                )

                st.download_button(
                    "📥 Download Answer",
                    download_text,
                    file_name="studymate_answer.txt"
                )


            except Exception as e:

                st.error(
                    "An error occurred: "
                    + str(e)
                )


# ------------------------------------------------------------
# CONVERSATION HISTORY
# ------------------------------------------------------------

if st.session_state.history:

    st.divider()

    st.subheader(
        "💬 Conversation History"
    )

    for item in reversed(
        st.session_state.history
    ):

        st.markdown(
            f"**You:** {item['question']}"
        )

        st.markdown(
            f"**StudyMate:** {item['answer']}"
        )
