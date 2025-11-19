import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# -----------------------------
# LOAD CORPUS
# -----------------------------
def load_documents(corpus_path="corpus"):
    docs = []
    for file in os.listdir(corpus_path):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(corpus_path, file))
            docs.extend(loader.load())
    return docs

# -----------------------------
# CHUNK DOCUMENTS
# -----------------------------
def chunk_documents(docs, chunk_size=300):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    return splitter.split_documents(docs)

# -----------------------------
# CREATE VECTOR DB
# -----------------------------
def build_vector_db(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb, embedding_model

# -----------------------------
# LOAD LANGUAGE MODEL
# -----------------------------
def load_llm():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    return tokenizer, model

# -----------------------------
# GENERATE ANSWER
# -----------------------------
def generate_answer(context, question, tokenizer, model):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# MAIN APPLICATION LOOP
# -----------------------------
def main():
    print("Loading documents...")
    docs = load_documents()

    print("Chunking documents...")
    chunks = chunk_documents(docs)

    print("Building vector database...")
    vectordb, embed = build_vector_db(chunks)

    print("Loading language model...")
    tokenizer, model = load_llm()

    print("\nRAG System Ready! Ask any question (type 'exit' to quit)\n")

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            print("Goodbye!")
            break

        # Retrieve top chunks
        results = vectordb.similarity_search(question, k=2)
        context = " ".join([r.page_content for r in results])

        # Generate answer
        answer = generate_answer(context, question, tokenizer, model)
        print("\nAnswer:", answer, "\n")

if __name__ == "__main__":
    main()
