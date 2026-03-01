# Import required libraries
import os
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from collections import Counter


# Path to folder containing research papers (PDFs)
papers_dir = r"C:\ml_projects\rag\papers"

# Initialize document converter (PDF -> structured document)
converter = DocumentConverter()

# Initialize hybrid chunker
# max_tokens: maximum tokens per chunk
# overlap: overlapping tokens between chunks for better context continuity
chunker = HybridChunker(max_tokens=480, overlap=50)

# List to store all processed document chunks
all_chunks = []


# Step 1: Loop through all PDF files in the folder
for filename in os.listdir(papers_dir):
    if filename.lower().endswith(".pdf"):   # Process only PDF files
        
        pdf_path = os.path.join(papers_dir, filename)
        print(f"Processing: {filename}")

        # Convert PDF into structured document format
        result = converter.convert(pdf_path)
        doc = result.document

        # ✂ Step 2: Split document into smaller semantic chunks
        chunks = list(chunker.chunk(doc))

        # 📦 Step 3: Convert each chunk into LangChain Document format
        for chunk in chunks:
            all_chunks.append(
                Document(
                    page_content=chunk.text,          # Actual text content
                    metadata={"source": filename}     # Store source PDF name
                )
            )

# Print total number of generated chunks
print(f"Total chunks: {len(all_chunks)}")


#  Step 4: Generate embeddings using OpenAI embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


#  Step 5: Create FAISS vector store from document chunks
vectorstore = FAISS.from_documents(
    documents=all_chunks,
    embedding=embeddings
)

#  Save the vector store locally for later retrieval
vectorstore.save_local("../vectorstore")

print("\nFAISS vector store created and saved!")


#  Step 6: Count number of chunks generated per PDF
sources = [doc.metadata["source"] for doc in all_chunks]
counts = Counter(sources)

print("\nChunks per PDF:")
for pdf, count in counts.items():
    print(f"{pdf}: {count}")
