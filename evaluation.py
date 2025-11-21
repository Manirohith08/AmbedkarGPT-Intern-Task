import json, os
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity

# Updated imports
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline


def evaluate_chunk_size(chunk_size, embed):

    print(f"\n=== Evaluating Chunk Size {chunk_size} ===")

    # Load all corpus files
    docs = []
    for file in os.listdir("corpus"):
        loader = TextLoader(f"corpus/{file}")
        docs += loader.load()

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Vector store
    vectordb = Chroma.from_documents(
        chunks,
        embed,
        persist_directory=f"db_{chunk_size}"
    )

    # Retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # LLM (Flan T5)
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_new_tokens=200
    )

    # Load test dataset
    dataset = json.load(open("test_dataset.json"))["test_questions"]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    results = []

    for item in dataset:
        q = item["question"]
        ground_truth = item["ground_truth"]
        src_docs = item["source_documents"]

        # Retrieve top documents
        retrieved_docs = retriever.get_relevant_documents(q)
        retrieved_sources = [doc.metadata["source"] for doc in retrieved_docs]

        # Build context
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Generate answer
        prompt = f"Context:\n{context_text}\n\nQuestion: {q}\nAnswer:"
        answer = llm(prompt)[0]["generated_text"]

        # Retrieval metrics
        hit = int(any(s in retrieved_sources for s in src_docs))
        mrr = next((1/(i+1)
                    for i, d in enumerate(retrieved_sources)
                    if d in src_docs), 0)
        precision_at_5 = (
            len([d for d in retrieved_sources if d in src_docs]) / len(retrieved_sources)
        )

        # Answer quality
        rougeL = scorer.score(answer, ground_truth)["rougeL"].fmeasure
        bleu = sentence_bleu([ground_truth.split()], answer.split())
        cosine = float(
            cosine_similarity(
                [embed.embed_query(answer)],
                [embed.embed_query(ground_truth)]
            )[0][0]
        )

        results.append({
            "id": item["id"],
            "hit_rate": hit,
            "mrr": mrr,
            "precision@5": precision_at_5,
            "rougeL": rougeL,
            "bleu": bleu,
            "cosine_similarity": cosine
        })

    return results



def main():
    embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    final_results = {}

    for size in [300, 600, 1000]:
        final_results[size] = evaluate_chunk_size(size, embed)

    json.dump(final_results, open("test_results.json", "w"), indent=4)
    print("\n✅ Evaluation complete — test_results.json created!")


if __name__ == "__main__":
    main()
