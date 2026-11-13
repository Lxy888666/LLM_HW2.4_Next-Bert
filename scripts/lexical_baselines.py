"""Low-cost sanity baselines: train-only multinomial NB and fixed BM25."""
import json,math,sys
from pathlib import Path
from collections import Counter
import numpy as np
from tokenizers import Tokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.scaled_data import OUT,ROOT,dump
from src.metrics import cls_scores,rank_scores

def main():
    tok=Tokenizer.from_file(str(OUT/'tokenizer.json'))
    enc=lambda s,n:tok.encode(s,add_special_tokens=False).ids[:n]
    cls=json.loads((OUT/'classification.json').read_text(encoding='utf8'))
    qa=json.loads((OUT/'qa.json').read_text(encoding='utf8'))
    result={'classification_multinomial_nb':{},'retrieval_bm25':{}}
    for l in ['en','zh']:
        counts=np.ones((2,8192),dtype=np.float64)
        for r in cls[l]['train']:
            ids=enc(r['text'],126);np.add.at(counts[r['label']],ids,1)
        logp=np.log(counts/counts.sum(-1,keepdims=True));pred=[]
        for r in cls[l]['test']:pred.append(int(logp[:,enc(r['text'],126)].sum(-1).argmax()))
        result['classification_multinomial_nb'][l]=cls_scores([r['label'] for r in cls[l]['test']],pred)
        rows=qa[l]['test'];docs=[Counter(enc(r['context'],254)) for r in rows]
        lengths=[sum(c.values()) for c in docs];avg=sum(lengths)/len(lengths)
        df=Counter(t for c in docs for t in c);N=len(docs);ranks=[]
        for i,r in enumerate(rows):
            q=Counter(enc(r['question'],126));scores=[]
            for c,n in zip(docs,lengths):
                score=0.
                for t in q:
                    f=c[t]
                    if f:score+=math.log(1+(N-df[t]+.5)/(df[t]+.5))*f*2.2/(f+1.2*(.25+.75*n/avg))
                scores.append(score)
            order=sorted(range(N),key=lambda j:(-scores[j],rows[j]['doc_id']));ranks.append(order.index(i)+1)
        result['retrieval_bm25'][l]=rank_scores(ranks)
    result['random_reference']={'binary_accuracy':.5,'retrieval_128_mrr':sum(1/i for i in range(1,129))/128,
        'retrieval_128_recall10':10/128,'rerank_10_mrr':sum(1/i for i in range(1,11))/10,
        'rerank_10_ndcg10':sum(1/math.log2(i+1) for i in range(1,11))/10}
    dump(ROOT/'results/validation/lexical_baselines.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
