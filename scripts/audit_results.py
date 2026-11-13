"""Check every main score against saved predictions and frozen test labels."""
import json,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.scaled_data import ROOT,OUT,dump
from src.metrics import cls_scores,qa_scores,rank_scores

def load(p):return json.loads(p.read_text(encoding='utf8'))

def close(actual,stored):
    for k,v in actual.items():assert math.isclose(v,stored[k],abs_tol=1e-12),(k,v,stored[k])

def main():
    folder=ROOT/'results/validation';cls=load(OUT/'classification.json');qa=load(OUT/'qa.json')
    env=load(folder/'environment.json');checks=0
    for seed in env['seeds']:
        for variant in ['B','E0','E1','E2','E3']:
            base=folder/f'seed_{seed}';m=load(base/f'{variant}_metrics.json');p=load(base/f'{variant}_predictions.json')
            assert m['parameters']==6921216
            log=load(base/f'{variant}_train.json');assert log['steps']==(3000 if variant=='B' else 500)
            for lang in ['en','zh']:
                c=[r for r in p['classification'] if r['lang']==lang]
                truth={r['id']:r['label'] for r in cls[lang]['test']}
                assert len(c)==256 and {r['id'] for r in c}==set(truth)
                assert all(r['gold']==truth[r['id']] for r in c)
                close(cls_scores([r['gold'] for r in c],[r['pred'] for r in c]),m['classification'][lang])
                q=[r for r in p['extraction'] if r['lang']==lang]
                truth={r['id']:r['answers'] for r in qa[lang]['test']}
                assert len(q)==128 and {r['id'] for r in q}==set(truth)
                assert all(r['gold']==truth[r['id']] for r in q)
                close(qa_scores([r['pred'] for r in q],[r['gold'] for r in q],[lang]*len(q)),m['extraction'][lang])
                for task,maximum in [('retrieval',128),('reranking',10)]:
                    r=[r for r in p[task] if r['lang']==lang]
                    assert len(r)==128 and {x['id'] for x in r}==set(truth)
                    assert all(1<=x['rank']<=maximum for x in r)
                    close(rank_scores([x['rank'] for x in r]),m[task][lang])
                checks+=4
            assert len(p['long_evidence'])==96
            for n in [256,1024,8192]:
                for loc in ['near','middle','far','removed']:
                    r=[x for x in p['long_evidence'] if x['length']==n and x['location']==loc]
                    assert len(r)==8 and sum(x['label'] for x in r)==4
                    acc=sum(x['label']==x['pred'] for x in r)/8
                    pair=sum(all(x['label']==x['pred'] for x in r if x['pair']==i) for i in range(4))/4
                    close({'accuracy':acc,'counterfactual_pair_accuracy':pair},m['long_evidence'][f'{n}_{loc}'])
                    checks+=1
    result={'status':'passed','evaluations':15,'metric_groups_checked':checks,
        'checks':['test IDs and labels match frozen data','main metrics recompute from saved predictions',
            'three seeds and exact training steps','balanced long-probe labels and paired scores']}
    dump(folder/'audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
