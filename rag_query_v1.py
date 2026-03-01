from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import json

# STEP 1: embeddings
embeddings = OpenAIEmbeddings()

# STEP 2: load vector database
vectorstore = FAISS.load_local(
    "../vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

def run_rag(query:str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """"Answer the question using ONLY the provided context.
        
        Give a clear, concept-focused explanation.
        Do NOT mention documents, papers, studies, or applications.
        Do NOT add extra topics beyond what is asked.
        Avoid survey-style language.
        Be precise and concise .
        Context:
        {context}
        Question:
        {question}
        """)
    
    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm)
    
    docs = retriever.invoke(query)
    if not docs:
        answer = "I don't know"
    else:
        response = rag_chain.invoke(query)
        answer = response.content

    return{
    "question": query,
    "answer": answer}

