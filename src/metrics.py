from __future__ import annotations
import math,re,string
from collections import Counter
PUNCT=set(string.punctuation)|set("，。！？；：、（）【】《》“”‘’…—·")
ART=re.compile(r"\b(a|an|the)\b",re.I)
def norm(s,lang):
    s=str(s).lower()
    if lang=="zh": return "".join(c for c in s if not c.isspace() and c not in PUNCT)
    return " ".join(ART.sub(" ","".join(c for c in s if c not in PUNCT)).split())
def f1(pred,gold,lang):
    a,b=norm(pred,lang),norm(gold,lang); x=list(a) if lang=="zh" else a.split(); y=list(b) if lang=="zh" else b.split()
    if not x or not y:return float(x==y)
    c=sum((Counter(x)&Counter(y)).values())
    if not c:return 0.
    p,r=c/len(x),c/len(y);return 2*p*r/(p+r)
def qa_scores(preds,golds,langs):
    em=[float(any(norm(p,l)==norm(g,l) for g in gs)) for p,gs,l in zip(preds,golds,langs)]
    fs=[max(f1(p,g,l) for g in gs) for p,gs,l in zip(preds,golds,langs)]
    return {"exact_match":sum(em)/len(em),"f1":sum(fs)/len(fs)}
def cls_scores(y,p):
    labels=sorted(set(y)|set(p)); fs=[]
    for z in labels:
        tp=sum(a==z and b==z for a,b in zip(y,p));fp=sum(a!=z and b==z for a,b in zip(y,p));fn=sum(a==z and b!=z for a,b in zip(y,p));pr=tp/(tp+fp) if tp+fp else 0;rc=tp/(tp+fn) if tp+fn else 0;fs.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return {"accuracy":sum(a==b for a,b in zip(y,p))/len(y),"macro_f1":sum(fs)/len(fs)}
def rank_scores(ranks):
    n=len(ranks);return {"mrr":sum(1/r for r in ranks)/n,"recall_at_1":sum(r<=1 for r in ranks)/n,"recall_at_5":sum(r<=5 for r in ranks)/n,"recall_at_10":sum(r<=10 for r in ranks)/n,"ndcg_at_10":sum(1/math.log2(r+1) if r<=10 else 0 for r in ranks)/n}
