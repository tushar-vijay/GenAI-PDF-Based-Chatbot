# 🤖 AI-Powered PDF Question Answering System (LLM + RAG)

An **AI-driven document question answering system** that allows users to upload PDF files and ask natural language questions.  
Built using **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** to deliver accurate, context-aware answers.

---

## 📌 Problem Statement

Manually searching through lengthy PDF documents is time-consuming and inefficient.  
This project solves that problem by enabling **instant, intelligent querying of PDF documents** using AI.

---

## 🚀 Key Highlights (Resume-Friendly)

- Designed and developed an **end-to-end RAG-based chatbot** for PDF documents  
- Implemented **semantic search** using OpenAI embeddings and FAISS vector database  
- Integrated **GPT-3.5 Turbo** via LangChain for context-aware question answering  
- Built an interactive **Streamlit web interface** for real-time document querying  
- Optimized text chunking with overlap to preserve contextual continuity  
- Achieved low-latency responses through efficient vector similarity search  

---

## 🛠️ Tech Stack & Tools

| Category | Technologies |
|--------|-------------|
| Programming Language | Python |
| LLM | OpenAI GPT-3.5 Turbo |
| Framework | LangChain |
| Vector Database | FAISS |
| UI | Streamlit |
| Document Processing | PyPDF2 |

---

## 🧠 System Architecture (RAG Pipeline)

1. PDF Upload via Streamlit UI  
2. Text extraction from PDF pages  
3. Chunking using RecursiveCharacterTextSplitter  
4. Embedding generation using OpenAI  
5. Vector storage using FAISS  
6. Similarity-based retrieval  
7. Answer generation using LLM  

---

## 📂 Project Structure

```text
├── app.py
├── PDF_Based_ChatBot.png
└── README.md
```
