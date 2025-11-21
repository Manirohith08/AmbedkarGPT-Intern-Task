import json
import os
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline
import numpy as np


def evaluate_chunk_size(chunk_size, embed):

    # Load all corpus files
    docs = []
    for file in os.listdir("corpus"):
        loader = TextLoader(f"corpus/{file}")
        docs.extend(loader.load())

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=60
    )
    chunks = splitter.split_documents(docs)

    # Build vector DB
    vectordb = Chroma.from_documents(
        chunks,
        embed,
        persist_directory=f"db_{chunk_size}"
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # Load FLAN-T5-Base (small + fast)
    llm = pipeline("text2text-generation",
                   model="google/flan-t5-base",
                   max_new_tokens=150)

    # Load dataset
    dataset = json.load(open("test_dataset.json"))["test_questions"]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = []

    for item in dataset:

        q = item["question"]
        gt = item["ground_truth"]
        sources = item["source_documents"]

        # FIX: Invoke correctly (string, not dict!)
        retrieved_docs = retriever.invoke(q)

        retrieved_paths = [doc.metadata["source"] for doc in retrieved_docs]

        # Build context block
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        context = " ".join(context.split()[:350])   # truncate for safety

        prompt = f"""
        Answer the question using ONLY the provided context.

        Context:
        {context}

        Question: {q}
        Answer:
        """

        ans = llm(prompt)[0]["generated_text"]

        # Evaluation metrics
        hit = int(any(s in retrieved_paths for s in sources))
        mrr = next((1/(i+1) for i, s in enumerate(retrieved_paths) if s in sources), 0)
        p5 = len([s for s in retrieved_paths if s in sources]) / 5

        rougeL = scorer.score(ans, gt)["rougeL"].fmeasure
        bleu = sentence_bleu([gt.split()], ans.split())
        cos = cosine_similarity(
            [embed.embed_query(ans)],
            [embed.embed_query(gt)]
        )[0][0]

        results.append({
            "id": item["id"],
            "hit_rate": hit,
            "mrr": mrr,
            "precision@5": p5,
            "rougeL": rougeL,
            "bleu": bleu,
            "cosine_similarity": float(cos)
        })

    return results


def main():

    print("Loading Embeddings...")
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    final = {}

    for size in [300, 600, 1000]:
        print(f"\n=== Evaluating Chunk Size {size} ===")
        final[size] = evaluate_chunk_size(size, embed)

    json.dump(final, open("test_results.json", "w"), indent=4)
    print("\n✔ DONE — test_results.json generated successfully!")


if __name__ == "__main__":
    main()
