# TA0081
An AI-powered medical chatbot built using Retrieval-Augmented Generation (RAG) architecture to provide citation-backed, context-aware, and hallucination-resistant medical responses

🚀 Project Overview

 This model is designed to reduce medical misinformation by grounding AI responses in verified medical documents such as clinical guidelines, research papers, and drug manuals.

Instead of generating answers from model memory alone, the system:

Retrieves relevant medical documents using vector search

Injects retrieved context into a Large Language Model (LLM)

Generates evidence-based responses

Provides safer and more reliable outputs

🏗 Architecture

User Query
↓
Embedding Generation (HuggingFace)
↓
Vector Search (Pinecone)
↓
Relevant Document Retrieval
↓
Prompt Injection
↓
Groq LLM Response
↓
Citation-backed Answer.

Backend :

Python

Flask

LangChain

🔹 AI Components

HuggingFace Embeddings

Pinecone Vector Database

Groq LLM (ChatGroq)

🔹 Document Processing

PyPDFLoader

RecursiveCharacterTextSplitter

🔹 Frontend

HTML

CSS

Bootstrap

JavaScript (AJAX)
