import os
import time
import pandas as pd
from groq import Groq
from langsmith import traceable
from retrieve import retrieve, _get_df
from rerank import rerank
from dotenv import load_dotenv
load_dotenv()
_groq=Groq(api_key=os.environ.get("GROQ_API_KEY"))

def _build_context(chunks:pd.DataFrame)->list[dict]:
    df=_get_df()
    parts=[]
    for _,row in chunks.iterrows():
        num=row["number"]
        start=row["start"]
        window=df[
            (df["number"]==num)&
            (df["start"]>=start-30)&
            (df["start"]<=start+30)
        ].sort_values("start")
        parts.append({
            "title":row["title"],
            "number":num,
            "timestamp":f"{int(start//60)}m{int(start%60):02d}s",
            "text":" ".join(window["text"].tolist()),
        })
    return parts

def _build_prompt(query:str,context:list[dict])->str:
    ctx="\n\n".join(
        f"[Video {c['number']}: {c['title']} | {c['timestamp']}]\n{c['text']}"
        for c in context
    )
    return f"""You are a teaching assistant for a Hindi Machine Learning course by Krish Naik.
    Rules:
    - Answer using ONLY the course content provided below.
    - Always cite the video number and timestamp so the user knows where to watch.
    - Use only the timestamps explicitly present in the context - never invent them.
    - If the topic is related to machine learning but NOT covered in the course content below, say: "This topic isn't covered in this course. Try searching for it elsewhere."
    - If the question is completely unrelated to machine learning, say: "I can only answer questions about this ML course."
    - Always respond in English, regardless of the language of the course content.

    COURSE CONTENT:
    {ctx}

    QUESTION: {query}

    ANSWER:"""

@traceable(name="retrieve-faiss",run_type="retriever")
def _retrieve(query:str,k:int)->pd.DataFrame:
    return retrieve(query,k=k)

@traceable(name="rerank-crossencoder",run_type="retriever")
def _rerank(query:str,results:pd.DataFrame)->pd.DataFrame:
    return rerank(query,results)

@traceable(name="llm-groq",run_type="llm")
def _llm(prompt:str)->str:
    r=_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
        max_tokens=512,
    )
    return r.choices[0].message.content

@traceable(name="rag-pipeline",run_type="chain")
def answer(query:str,k:int=10)->tuple[str,list[dict],dict]:
    t0=time.perf_counter()
    t=time.perf_counter(); results=_retrieve(query,k); retrieve_ms=round((time.perf_counter()-t)*1000)
    t=time.perf_counter(); reranked=_rerank(query,results); rerank_ms=round((time.perf_counter()-t)*1000)
    context=_build_context(reranked)
    t=time.perf_counter(); ans=_llm(_build_prompt(query,context)); llm_ms=round((time.perf_counter()-t)*1000)
    latency={"retrieve_ms":retrieve_ms,"rerank_ms":rerank_ms,"llm_ms":llm_ms,"total_ms":round((time.perf_counter()-t0)*1000)}
    return ans,context,latency

if __name__=="__main__":
    query=input("Ask a question: ")
    ans,context,latency=answer(query)
    print("\n--- ANSWER ---")
    print(ans)
    print("\n--- SOURCES ---")
    for c in context:
        print(f"  Video {c['number']}: {c['title']} @ {c['timestamp']}")
    print("\n--- LATENCY ---")
    for k,v in latency.items():
        print(f"  {k}: {v}ms")