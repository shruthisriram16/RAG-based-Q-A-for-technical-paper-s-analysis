import os
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
papers_dir = r"C:\ml_projects\rag\papers"

converter = DocumentConverter()
chunker = HybridChunker(max_tokens=480, overlap=50)

all_chunks = []

for filename in os.listdir(papers_dir):
    if filename.lower().endswith(".pdf"):   
        pdf_path = os.path.join(papers_dir, filename)

        print(f"Processing: {filename}")

        result = converter.convert(pdf_path)
        doc = result.document

        chunks = list(chunker.chunk(doc))
        for chunk in chunks:
            all_chunks.append(
                Document(
                    page_content=chunk.text,
                    metadata={"source": filename}
                )
            )
            
       

print(f"Total chunks: {len(all_chunks)}")


embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"

)
vectorstore=FAISS.from_documents(
    documents=all_chunks,
    embedding=embeddings
)
vectorstore.save_local("../vectorstore")
print("\nFAISS vector store created and saved!")

from collections import Counter

sources = [doc.metadata["source"] for doc in all_chunks]
counts = Counter(sources)

print("\nChunks per PDF:")
for pdf, count in counts.items():
    print(f"{pdf}: {count}")



