import json, os
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity

# UPDATED IMPORTS FOR NEW LANGCHAIN
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline

from langchain.chains import RetrievalQA
from transformers import pipeline
import numpy as np


def evaluate_chunk_size(chunk_size, embed):
    print(f"\n🔹 Evaluating chunk size = {chunk_size}...\n")

    # Load documents
    docs = []
    for file in os.listdir("corpus"):
        loader = TextLoader(f"corpus/{file}")
        docs += loader.load()

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Build vector database
    vectordb = Chroma.from_documents(
        chunks,
        embed,
        persist_directory=f"db_{chunk_size}"
    )

    # Load LLM
    llm = HuggingFacePipeline(
        pipeline=pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            max_new_tokens=200
        )
    )

    # Retrieval-QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

    # Load test dataset
    dataset = json.load(open("test_dataset.json"))["test_questions"]
    results = []
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # Evaluate each question
    for item in dataset:
        q = item["question"]
        gt = item["ground_truth"]
        src_docs = item["source_documents"]

        res = qa(q)
        ans = res["result"]
        retrieved_docs = [doc.metadata["source"] for doc in res["source_documents"]]

        # Metrics
        hit = int(any(s in retrieved_docs for s in src_docs))
        mrr = next((1 / (i + 1) for i, d in enumerate(retrieved_docs) if d in src_docs), 0)
        p_at_5 = len([d for d in retrieved_docs if d in src_docs]) / len(retrieved_docs)

        rougeL = scorer.score(ans, gt)["rougeL"].fmeasure
        bleu = sentence_bleu([gt.split()], ans.split())
        cos = cosine_similarity(
            [embed.embed_query(ans)], 
            [embed.embed_query(gt)]
        )[0][0]

        results.appen
