import json, os
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity

# UPDATED IMPORTS (new packages)
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chains import RetrievalQA

from transformers import pipeline
import numpy as np


def evaluate_chunk_size(chunk_size, embed):

    # Load corpus files
    docs = []
    for file in os.listdir("corpus"):
        loader = TextLoader(f"corpus/{file}")
        docs += loader.load()

    # Split text into chunks
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Create vector store
    vectordb = Chroma.from_documents(
        chunks,
        embed,
        persist_directory=f"db_{chunk_size}"
    )

    # LLM model
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_new_tokens=200
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

    # Load dataset
    dataset = json.load(open("test_dataset.json"))["test_questions"]
    results = []

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for item in dataset:
        q = item["question"]
        gt = item["ground_truth"]
        src_docs = item["source_documents"]

        res = qa(q)
        ans = res["result"]
        retrieved = [doc.metadata["source"] for doc in res["source_documents"]]

        # Evaluate retrieval metrics
        hit = int(any(s in retrieved for s in src_docs))
        mrr = next((1 / (i + 1)
                   for i, d in enumerate(retrieved)
                   if d in src_docs), 0)
        p_at_5 = len([d for d in retrieved if d in src_docs]) / len(retrieved)

        # Evaluate answer quality
        rougeL = scorer.score(ans, gt)["rougeL"].fmeasure
        bleu = sentence_bleu([gt.split()], ans.split())
        cos = cosine_similarity(
            [embed.embed_query(ans)], [embed.embed_query(gt)]
        )[0][0]

        results.append({
            "id": item["id"],
            "hit_rate": hit,
            "mrr": mrr,
            "precision@5": p_at_5,
            "rougeL": rougeL,
            "bleu": bleu,
            "cosine_similarity": float(cos)
        })

    return results



def main():
    embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    final = {}
    for size in [300, 600, 1000]:
        final[size] = evaluate_chunk_size(size, embed)

    json.dump(final, open("test_results.json", "w"), indent=4)
    print("Evaluation complete → test_results.json created!")


if __name__ == "__main__":
    main()
