import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
import torch

def build_rag():
    loader = TextLoader("speech.txt")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./db")
    vectordb.persist()

    llm = HuggingFacePipeline(
        pipeline=pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            max_new_tokens=200,
            device=0 if torch.cuda.is_available() else -1
        )
    )

    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa

if __name__ == "__main__":
    qa = build_rag()
    print("\nAmbedkarGPT is ready! Type 'exit' to quit.\n")
    while True:
        q = input("Question: ")
        if q.lower() == "exit":
            break
        result = qa(q)
        print("\nAnswer:", result["result"], "\n")
