import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from functools import lru_cache
_embed_model=SentenceTransformer("all-MiniLM-L6-v2")
_index=faiss.read_index("faiss.index")

@lru_cache(maxsize=1)
def _get_df()->pd.DataFrame:
    return pd.read_pickle("chunks.pkl")
def retrieve(query:str,k:int=10)->pd.DataFrame:
    embedding=_embed_model.encode(
        [query],normalize_embeddings=True
    ).astype(np.float32)
    scores, indices=_index.search(embedding,k)
    df=_get_df()
    results=df.iloc[indices[0]].copy().reset_index(drop=True)
    results["score"]=scores[0]
    return results[["chunk_id","title","number","start","end","text","score"]]
if __name__=="__main__":
    from rerank import rerank
    query=input("Ask a question: ")
    res=retrieve(query, k=10)
    reranked=rerank(query, res)
    for _,row in reranked.iterrows():
        print(f"\n[{row['title']} | {row['start']:.0f}s]  rerank={row['rerank_score']:.3f}")
        print(row["text"])