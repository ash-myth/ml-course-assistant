import time
import threading
from collections import deque
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

_lock=threading.Lock()
_log:deque=deque(maxlen=200)

def _record(latency:dict,error:bool=False):
    with _lock:
        _log.append({**latency,"ts":time.time(),"error":error})

def _stats(vals:list)->dict:
    if not vals: return {}
    s=sorted(vals); n=len(s)
    return {"mean_ms":round(sum(s)/n),"p50_ms":s[n//2],"p95_ms":s[min(int(n*0.95),n-1)],"max_ms":s[-1]}

def _metrics()->dict:
    with _lock: log=list(_log)
    if not log: return {"total_requests":0}
    ok=[r for r in log if not r["error"]]
    return {
        "total_requests":len(log),
        "error_count":len(log)-len(ok),
        "success_rate_pct":round(len(ok)/len(log)*100,1),
        "latency":{
            "total":_stats([r["total_ms"] for r in ok]),
            "retrieve":_stats([r["retrieve_ms"] for r in ok]),
            "rerank":_stats([r["rerank_ms"] for r in ok]),
            "llm":_stats([r["llm_ms"] for r in ok]),
        }
    }

@asynccontextmanager
async def lifespan(app:FastAPI):
    print("[startup] Loading models and index...")
    t0=time.time()
    from retrieve import _get_df
    from answer import answer
    _get_df()
    print(f"[startup] Ready in {time.time()-t0:.1f}s")
    yield

app=FastAPI(title="Krish Naik ML Course RAG",version="1.0.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

class QueryRequest(BaseModel):
    question:str=Field(...,min_length=3,max_length=500)
    k:int=Field(default=20,ge=1,le=25)
class Source(BaseModel):
    title:str; number:str; timestamp:str; text:str
class QueryResponse(BaseModel):
    answer:str; sources:list[Source]; latency:dict

@app.get("/")
def root(): return {"message":"RAG API is running - POST to /ask"}

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/metrics")
def metrics(): return _metrics()

@app.post("/ask",response_model=QueryResponse)
def ask(query:QueryRequest):
    from answer import answer
    try:
        ans,sources,latency=answer(query.question,k=query.k)
        _record(latency,error=False)
    except Exception as e:
        _record({"retrieve_ms":0,"rerank_ms":0,"llm_ms":0,"total_ms":0},error=True)
        raise HTTPException(status_code=500,detail=str(e))
    return QueryResponse(answer=ans,sources=[Source(**s) for s in sources],latency=latency)