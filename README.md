# 🎓 StudyMate – Academic RAG Assistant

## Student Information

**Student Name:** Arti  
**Roll Number:** MC2505

---

## Project Title

StudyMate – Academic Retrieval-Augmented Generation Assistant

---

## Project Description

StudyMate is an AI-powered academic assistant based on
Retrieval-Augmented Generation (RAG).

The system retrieves relevant information from an academic
knowledge base and uses FLAN-T5 to generate an answer.

---

## Technologies Used

- Python
- Streamlit
- FastAPI
- FAISS
- Sentence Transformers
- FLAN-T5
- Requests
- NumPy

---

## RAG Pipeline

Dataset

↓

Text Preprocessing

↓

Text Chunking

↓

Sentence Transformer Embeddings

↓

FAISS Vector Search

↓

Relevant Context Retrieval

↓

Prompt Engineering

↓

FLAN-T5

↓

Generated Answer

---

## Embedding Model

all-MiniLM-L6-v2

The embedding model converts text into numerical vectors
for semantic similarity search.

---

## Vector Database

FAISS (Facebook AI Similarity Search)

FAISS is used to efficiently search the document embeddings.

---

## Language Model

FLAN-T5-small

The model generates answers using the retrieved context.

---

## External APIs

StudyMate integrates five external APIs:

1. Wikipedia API
2. arXiv API
3. OpenAlex API
4. Crossref API
5. Open Library API

---

## Own REST API

StudyMate also provides a FastAPI endpoint:

POST /ask

Example request:

{
    "question": "What is Retrieval-Augmented Generation?",
    "top_k": 3
}

---

## Features

- Academic question answering
- Retrieval-Augmented Generation
- FAISS similarity search
- Source document retrieval
- Five external research APIs
- FastAPI REST API
- Streamlit interface
- Conversation history
- Download answer functionality

---

## Project Structure

StudyMate-RAG-Academic-Assistant/

├── app.py

├── academic_notes.txt

├── requirements.txt

├── README.md

└── screenshots/

---

## Installation

Install the required packages:

pip install -r requirements.txt

---

## Run the Application

streamlit run app.py

---

## Deployment

The application is intended to be deployed using
Streamlit Community Cloud.

---

## Author

Student Name: Arti

Roll Number: MC2505
