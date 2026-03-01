# RAG-Based Q&A for Technical Paper Analysis

An end-to-end Retrieval-Augmented Generation (RAG) system that enables
intelligent question answering over technical research papers.

Users can upload PDFs, and the system retrieves relevant content using
semantic search before generating precise, context-aware answers using a
Large Language Model (LLM).

------------------------------------------------------------------------

##  Features

-    Automatic PDF ingestion\
-   Intelligent hybrid chunking\
-   OpenAI embedding generation\
-   FAISS vector database indexing\
-   Top-k similarity-based retrieval\
-   LLM-based grounded answer generation\
-   Streamlit chat interface

------------------------------------------------------------------------

## System Workflow

###  Document Ingestion

-   Converts PDF papers into structured documents
-   Splits content into overlapping semantic chunks
-   Stores metadata (source file information)

###  Embedding & Indexing

-   Generates embeddings using OpenAI embedding model
-   Stores vectors in FAISS for efficient similarity search
-   Saves vector database locally for reuse

### Retrieval-Augmented Generation

-   Retrieves top-k relevant chunks based on query
-   Passes retrieved context + user query to LLM
-   Generates precise, context-grounded answers

### User Interface

-   Streamlit-based interactive chat UI
-   Maintains session history
-   Displays assistant responses in real-time

------------------------------------------------------------------------

## Tech Stack

-   Python\
-   LangChain\
-   OpenAI API\
-   FAISS\
-   Streamlit\
-   Docling

------------------------------------------------------------------------

## Project Structure

    rag-paper-qa/
    │
    ├── papers/                # Input PDF files
    ├── vectorstore/           # Generated FAISS index (ignored in .gitignore)
    ├── rag_v2.py              # PDF processing & indexing
    ├── rag_query_v1.py        # RAG retrieval pipeline
    ├── streamlit_rag_v1.py    # Streamlit interface
    └── README.md

------------------------------------------------------------------------

## Setup Instructions

### 3. Set Environment Variables

Create a `.env` file:

    OPENAI_API_KEY=your_api_key_here

### 4. Run Document Ingestion

``` bash
python rag_main.py
```

### 5. Run RAG query pipeline

``` bash
python rag_query.py
```

### 5. Launch Streamlit App

``` bash
streamlit run streamlit_rag.py
```

------------------------------------------------------------------------



## Sample Outputs

### Chunking Output

The following screenshot shows sample document chunks generated after hybrid chunking:

![Chunking Output](outputs/chunk_output.png)

---

### RAG Query Output

The following screenshot demonstrates the retrieval-augmented answer generation:

![RAG Output](outputs/rag_output.png)

------------------------------------------------------------------------
## Use Cases

-   Research paper analysis\
-   Literature review assistance\
-   Technical concept clarification\
-   Academic Q&A automation

------------------------------------------------------------------------

## Notes

-   The `vectorstore/` directory is excluded from version control.
-   Embeddings are regenerated locally after ingestion.
-   The system reduces hallucination by grounding answers in retrieved
    context.
