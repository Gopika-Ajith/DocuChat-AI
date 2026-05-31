# 📚 DocuChat AI

AI-powered document assistant that allows users to upload PDFs, ask questions in natural language, and receive context-aware answers using Retrieval-Augmented Generation (RAG).

## 🚀 Features

* Upload one or multiple PDF files
* Semantic document search using FAISS
* AI-powered answers using Groq LLM
* Document summarization
* Source page tracking
* Download extracted text
* Modern chat interface
* Multi-document support

## 🛠️ Tech Stack

* Python
* Streamlit
* FAISS
* Sentence Transformers
* LangChain
* Groq API
* PyPDF

## 📂 Project Workflow

1. Upload PDF documents
2. Extract text from PDFs
3. Split text into chunks
4. Generate embeddings
5. Store embeddings in FAISS
6. Retrieve relevant content
7. Generate AI-powered answers using Groq

## ⚡ Installation

```bash
git clone https://github.com/Gopika-Ajith/DocuChat-AI.git
cd DocuChat-AI

pip install -r requirements.txt

streamlit run app.py
```
## Live Demo
    
    https://docuchat-ai-agop.streamlit.app/
---
## 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

## 🎯 Future Improvements

* Generate MCQs from PDFs
* Generate study notes
* Export chat history
* PDF summarization dashboard
* Support for DOCX and TXT files

## 👩‍💻 Author

Gopika Ajith

B.Tech Artificial Intelligence & Data Science
