import os
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

model=SentenceTransformer("all-MiniLM-L6-v2")
records=[]
for json_file in os.listdir("jsons"):
    with open(f"jsons/{json_file}") as f:
        content=json.load(f)
    for chunk in content["chunks"]:
        records.append({
            "chunk_id": len(records),
            "number": chunk["number"],
            "title": chunk["title"],
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"]
        })
df=pd.DataFrame(records)
texts=df["text"].tolist()
print(f"Embedding {len(texts)} chunks")
embeddings=model.encode(texts,batch_size=64,show_progress_bar=True)
embeddings=embeddings.astype(np.float32)
dim=embeddings.shape[1]
index=faiss.IndexFlatIP(dim)
faiss.normalize_L2(embeddings)
index.add(embeddings)

faiss.write_index(index,"faiss.index")
df.to_pickle("chunks.pkl")
print(f"{len(df)} chunks embedded")    