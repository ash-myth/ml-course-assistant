import sys
sys.path.insert(0,".")
import json
from retrieve import retrieve
from rerank import rerank

with open("eval/questions.json") as f:
    questions=json.load(f)
hits=0
results_log=[]
for q in questions:
    retrieved=retrieve(q["question"],k=20)
    reranked=rerank(q["question"],retrieved)
    res=reranked["number"].tolist()
    hit=q["expected_video"] in res
    
    if hit: 
        hits+=1
    results_log.append({"question":q["question"],"expected":q["expected_video"],"got":res,"hit":hit})

for r in results_log:
    status="OK" if r["hit"] else "NOT OK"
    print(f"{status} [{r['expected']}] {r['question']}")
    if not r["hit"]:
        print(f"  got: {r['got']}")
print(f"\nhit@3 = {hits}/{len(questions)} ({hits/len(questions)*100:.1f}%)")

    