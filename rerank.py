from sentence_transformers import CrossEncoder
import pandas as pd

reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, results, top_k=3):
    results = results.copy()
    pairs = [[query, text] for text in results["text"].tolist()]
    scores = reranker.predict(pairs)
    results = results.assign(rerank_score=scores)
    # small penalty for late timestamps, prefer earlier explanations
    max_start = results["start"].max() if results["start"].max()>0 else 1
    position_penalty = results["start"] / max_start*0.3
    results = results.assign(rerank_score=results["rerank_score"]-position_penalty)
    results = results.sort_values("rerank_score", ascending=False)
    results = results.drop_duplicates(subset=["text"])
    return results.head(top_k).reset_index(drop=True)