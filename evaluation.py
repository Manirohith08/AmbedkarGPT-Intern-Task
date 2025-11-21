import json, os
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline
import numpy as np


def truncate(text, max_len=350):
    """Truncate text to prevent token overflow."""
    words = text.split()
    if len(words) > max_len:
        return " ".join(words[:max_len])
    return text


def evaluate_chunk_size(chunk_size, embed):
    # Load all corpus documents
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

    # Load QA model (FLAN-T5-Base)
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=180
    )

    with open("test_dataset.json", "r") as f:
        dataset = json.load(f)["test_questions"]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = []

    for item in dataset:
        q = item["question"]
        gt = item["ground_truth"]
        src_docs = item["source_documents"]

        # Retrieve relevant documents
        retrieved_docs = retriever.invoke({"query": q})
        retrieved_paths = [d.metadata["source"] for d in retrieved_docs]

        # Build model input
        context = "\n".join([d.page_content for d in retrieved_docs])
        context = truncate(context)

        prompt = f"Answer the question based ONLY on the text.\n\nContext:\n{context}\n\nQuestion: {q}\nAnswer:"

        # Run the model
        ans = llm(prompt)[0]["generated_text"]
        ans = truncate(ans)

        # Ranking metrics
        hit = int(any(s in retrieved_paths for s in src_docs))
        mrr = next((1 / (i + 1) for i, d in enumerate(retrieved_paths) if d in src_docs), 0)
        p_at_5 = len([d for d in retrieved_paths if d in src_docs]) / 5

        # Similarity scores
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
            "precision@5": p_at_5,
            "rougeL": rougeL,
            "bleu": bleu,
            "cosine_similarity": float(cos)
        })

    return results


def main():
    print("Loading embeddings...")
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    final_results = {}

    for size in [300, 600, 1000]:
        print(f"\n=== Evaluating Chunk Size {size} ===")
        final_results[size] = evaluate_chunk_size(size, embed)

    with open("test_results.json", "w") as f:
        json.dump(final_results, f, indent=4)

    print("\n✅ Evaluation Complete — test_results.json Generated Successfully!")


if __name__ == "__main__":
    main()
