"""Reproducible small-scale ablations of the proposed encoder (GPU required)."""
from __future__ import annotations
import os
os.environ.setdefault('USE_TF','0'); os.environ.setdefault('USE_FLAX','0')
import argparse, copy, gc, json, math, random, statistics, sys, time
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from src.scaled_data import ROOT, OUT, dump, file_hash
from src.scaled_model import Config, Encoder, estimated_flops, mean_pool
from src.span_masking import corrupt_batch
from src.metrics import cls_scores, qa_scores, rank_scores

DEST=ROOT/'results/scaled'; DEVICE='cuda'

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False

class Tok:
    all_special_ids=[0,1,2,3,4];mask_token_id=4;vocab_size=8192
    def __init__(self):self.raw=Tokenizer.from_file(str(OUT/'tokenizer.json'))
    def encode(self,text):return self.raw.encode(text,add_special_tokens=False).ids
    def batch(self,texts,length=256,pairs=None):
        ids=[]
        for i,t in enumerate(texts):
            a=self.encode(t)
            if pairs is None: row=[2]+a[:length-2]+[3]
            else:
                a=a[:min(64,length//3)];b=self.encode(pairs[i])[:length-3-len(a)]
                row=[2]+a+[3]+b+[3]
            ids.append(row+[0]*(length-len(row)))
        x=torch.tensor(ids,device=DEVICE)
        return {'input_ids':x,'attention_mask':x.ne(0)}

def amp():return torch.autocast('cuda',dtype=torch.bfloat16)

def load_data(tok):
    cls=json.loads((OUT/'classification.json').read_text(encoding='utf-8'))
    qa=json.loads((OUT/'qa.json').read_text(encoding='utf-8'))
    corp=json.loads((OUT/'corpus.json').read_text(encoding='utf-8'))
    chunks={l:[] for l in ['en','zh']}
    for r in corp:
        ids=tok.encode(r['text'])
        # Each sample stays within one document; no cross-document packing.
        for k in range(0,len(ids),254):
            row=ids[k:k+254]
            if len(row)<32:continue
            row=[2]+row+[3];chunks[r['lang']].append(row+[0]*(256-len(row)))
    return cls,qa,{k:torch.tensor(v,dtype=torch.long) for k,v in chunks.items()}

def sample_batch(chunks,rng,batch=8):
    rows=[]
    for lang in ['en','zh']:
        rows += [chunks[lang][rng.randrange(len(chunks[lang]))] for _ in range(batch//2)]
    x=torch.stack(rows).to(DEVICE)
    return dict(input_ids=x,attention_mask=x.ne(0))

def save_model(path,model,seed,stage):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(config=asdict(model.c),state_dict=model.cpu().state_dict(),seed=seed,stage=stage,
        tokenizer_sha256=file_hash(OUT/'tokenizer.json'),manifest_sha256=file_hash(OUT/'manifest.json')),path)
    model.to(DEVICE)

def load_model(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    model=Encoder(Config(**ck['config']));model.load_state_dict(ck['state_dict'])
    return model.to(DEVICE)

def train(model,tok,chunks,seed,steps,variant,base=False):
    seed_all(seed);rng=random.Random(seed+911)
    model.train();optim=torch.optim.AdamW(model.parameters(),lr=5e-4 if base else 2e-4,weight_decay=.01)
    logs=[];t0=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        batch=sample_batch(chunks,rng)
        mode='span' if variant=='E1' and step%2 else 'token'
        ids,labels,_=corrupt_batch(batch,tok,mode,seed=seed*10000+step)
        optim.zero_grad(set_to_none=True)
        with amp():
            h1=model(ids,batch['attention_mask']);sel=labels.ne(-100)
            mlm1=F.cross_entropy(model.logits(h1[sel]).float(),labels[sel])
            if base:
                mlm=mlm1;con=torch.zeros((),device=DEVICE);loss=mlm
            else:
                # Every branch pays for the same two encoder passes and MLM heads.
                h2=model(ids,batch['attention_mask'])
                mlm2=F.cross_entropy(model.logits(h2[sel]).float(),labels[sel]);mlm=(mlm1+mlm2)/2
                if variant=='E3':
                    a=F.normalize(mean_pool(h1,ids,batch['attention_mask']).float(),dim=-1)
                    b=F.normalize(mean_pool(h2,ids,batch['attention_mask']).float(),dim=-1)
                    scores=a@b.T/.05;target=torch.arange(len(a),device=DEVICE)
                    con=(F.cross_entropy(scores,target)+F.cross_entropy(scores.T,target))/2
                else:con=torch.zeros((),device=DEVICE)
                loss=mlm+.1*con
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        for group in optim.param_groups:group['lr']=(5e-4 if base else 2e-4)*min(1,(step+1)/20)*(.2+.8*(1-step/steps))
        optim.step()
        if (step+1)%20==0 or step==0:
            row=dict(step=step+1,mlm=float(mlm.detach()),contrastive=float(con.detach()))
            logs.append(row);print(json.dumps(dict(seed=seed,variant=variant,**row)),flush=True)
    torch.cuda.synchronize()
    result=dict(steps=steps,encoder_passes=1 if base else 2,batch=8,length=256,
        seconds=time.perf_counter()-t0,
        estimated_matmul_training_flops=estimated_flops(model.c,8,256)*steps*(1 if base else 2),logs=logs)
    del optim;gc.collect();return result

@torch.no_grad()
def mlm_validation(model,tok,qa):
    model.eval();total=0.;count=0
    texts=[]
    for i in range(32):
        for l in ['en','zh']:texts.append(qa[l]['validation'][i]['context'])
    for k in range(0,len(texts),8):
        batch=tok.batch(texts[k:k+8]);ids,labels,sel=corrupt_batch(batch,tok,seed=900000+k)
        with amp():loss=model.mlm(ids,batch['attention_mask'],labels)
        n=int(sel.sum());total+=float(loss)*n;count+=n
    return dict(loss=total/count,masked_tokens=count)

@torch.no_grad()
def features(model,tok,texts,pairs=None,length=128,batch=16):
    model.eval();out=[]
    for k in range(0,len(texts),batch):
        b=tok.batch(texts[k:k+batch],length,None if pairs is None else pairs[k:k+batch])
        with amp():h=model(b['input_ids'],b['attention_mask']);p=mean_pool(h,b['input_ids'],b['attention_mask'])
        out.append((p if pairs is None else torch.cat([h[:,0],p],dim=-1)).float().cpu())
    return torch.cat(out)

def train_head(x,y,vx,vy,seed,classes=2,epochs=100):
    seed_all(seed);x=x.to(DEVICE);y=torch.tensor(y,device=DEVICE)
    vx=vx.to(DEVICE);vy=torch.tensor(vy,device=DEVICE)
    head=nn.Linear(x.shape[-1],classes).to(DEVICE)
    optim=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.01)
    best=None;best_loss=float('inf')
    for _ in range(epochs):
        optim.zero_grad(set_to_none=True);loss=F.cross_entropy(head(x),y);loss.backward();optim.step()
        with torch.no_grad(): val=float(F.cross_entropy(head(vx),vy))
        if val<best_loss:best_loss=val;best=copy.deepcopy(head.state_dict())
    head.load_state_dict(best);return head.cpu()

def classification(model,tok,data,seed,reference=None):
    xs={};ys={};rows={}
    for s in ['train','validation','test']:
        rows[s]=data['en'][s]+data['zh'][s];ys[s]=[r['label'] for r in rows[s]]
        xs[s]=features(model,tok,[r['text'] for r in rows[s]])
    head=train_head(xs['train'],ys['train'],xs['validation'],ys['validation'],seed)
    with torch.no_grad():pred=head(xs['test']).argmax(-1).tolist()
    result={l:cls_scores(ys['test'][i*256:(i+1)*256],pred[i*256:(i+1)*256]) for i,l in enumerate(['en','zh'])}
    if reference is not None:
        with torch.no_grad():p=reference(xs['test']).argmax(-1).tolist()
        result['fixed_reference_head']={l:cls_scores(ys['test'][i*256:(i+1)*256],p[i*256:(i+1)*256]) for i,l in enumerate(['en','zh'])}
    return result,head,[dict(id=r['id'],lang=r['lang'],gold=y,pred=p) for r,y,p in zip(rows['test'],ys['test'],pred)]

def qa_windows(tok,rows,length=256,stride=128):
    windows=[]
    for index,r in enumerate(rows):
        q=tok.encode(r['question'])[:48];enc=tok.raw.encode(r['context'],add_special_tokens=False)
        context,offsets=enc.ids,enc.offsets;capacity=length-len(q)-3
        pos=0
        while pos<len(context):
            end=min(len(context),pos+capacity);prefix=[2]+q+[3];row=prefix+context[pos:end]+[3]
            off=[(-1,-1)]*len(prefix)+offsets[pos:end]+[(-1,-1)]
            start=r['answer_start'];finish=start+len(r['answers'][0]);a=b=0
            candidates=[j for j,(s,e) in enumerate(off) if s>=0 and s<=start<e]
            ends=[j for j,(s,e) in enumerate(off) if s>=0 and s<finish<=e]
            if candidates and ends:a,b=candidates[0],ends[-1]
            windows.append(dict(index=index,ids=row+[0]*(length-len(row)),offsets=off+[(-1,-1)]*(length-len(row)),start=a,end=b))
            if end==len(context):break
            pos+=min(stride,capacity)
    return windows

@torch.no_grad()
def qa_features(model,windows):
    model.eval();out=[]
    for k in range(0,len(windows),8):
        ids=torch.tensor([w['ids'] for w in windows[k:k+8]],device=DEVICE)
        with amp():h=model(ids,ids.ne(0))
        out.append(h.cpu().to(torch.float16))
    return torch.cat(out)

def qa_loss(head,x,windows,idx):
    logits=head(x[idx].float());starts=torch.tensor([windows[i]['start'] for i in idx.tolist()],device=DEVICE)
    ends=torch.tensor([windows[i]['end'] for i in idx.tolist()],device=DEVICE)
    valid=torch.tensor([[s>=0 for s,e in windows[i]['offsets']] for i in idx.tolist()],device=DEVICE)
    valid[:,0]=True;logits=logits.masked_fill(~valid[:,:,None],-10000.)
    return (F.cross_entropy(logits[:,:,0],starts)+F.cross_entropy(logits[:,:,1],ends))/2

@torch.no_grad()
def qa_decode(head,x,windows,rows):
    best=[(-float('inf'),'') for _ in rows]
    for k in range(0,len(windows),32):
        logits=head(x[k:k+32].float()).numpy()
        for j,w in enumerate(windows[k:k+32]):
            off=w['offsets'];scores=logits[j];valid=np.array([s>=0 for s,e in off])
            scores[~valid]=-1e9
            aa=np.argsort(-scores[:,0])[:12];bb=np.argsort(-scores[:,1])[:12]
            for a in aa:
                for b in bb:
                    if a<=b and b-a<=40 and valid[a] and valid[b]:
                        score=float(scores[a,0]+scores[b,1])
                        if score>best[w['index']][0]:best[w['index']]=(score,rows[w['index']]['context'][off[a][0]:off[b][1]])
    return [s for _,s in best]

def extraction(model,tok,data,seed,reference=None):
    rows={s:data['en'][s]+data['zh'][s] for s in ['train','validation','test']}
    windows={s:qa_windows(tok,r) for s,r in rows.items()}
    xs={s:qa_features(model,ws) for s,ws in windows.items()}
    seed_all(seed);head=nn.Linear(256,2).to(DEVICE);optim=torch.optim.AdamW(head.parameters(),lr=.003)
    x=xs['train'].to(DEVICE);vx=xs['validation'].to(DEVICE);best=None;best_loss=float('inf')
    # Full windows are retained, including windows not containing the answer.
    for epoch in range(8):
        order=torch.randperm(len(x))
        for idx in order.split(32):
            optim.zero_grad(set_to_none=True);loss=qa_loss(head,x,windows['train'],idx)
            loss.backward();optim.step()
        with torch.no_grad():
            val=sum(float(qa_loss(head,vx,windows['validation'],i))*len(i) for i in torch.arange(len(vx)).split(32))/len(vx)
        if val<best_loss:best_loss=val;best=copy.deepcopy(head.state_dict())
    head.load_state_dict(best);head=head.cpu();del x,vx,optim;gc.collect()
    pred=qa_decode(head,xs['test'],windows['test'],rows['test'])
    def score(p):return {l:qa_scores(p[i*128:(i+1)*128],[r['answers'] for r in data[l]['test']],[l]*128) for i,l in enumerate(['en','zh'])}
    result=score(pred);result['windows']={s:len(v) for s,v in windows.items()}
    if reference is not None:result['fixed_reference_head']=score(qa_decode(reference,xs['test'],windows['test'],rows['test']))
    return result,head,[dict(id=r['id'],lang=r['lang'],gold=r['answers'],pred=p) for r,p in zip(rows['test'],pred)]

def retrieval(model,tok,data):
    result={};pred=[]
    for lang in ['en','zh']:
        rows=data[lang]['test']
        a=F.normalize(features(model,tok,[r['question'] for r in rows]),dim=-1)
        b=F.normalize(features(model,tok,[r['context'] for r in rows],length=256),dim=-1)
        scores=(a@b.T).numpy();ranks=[]
        for i,r in enumerate(rows):
            order=sorted(range(len(rows)),key=lambda j:(-float(scores[i,j]),rows[j]['doc_id']))
            rank=order.index(i)+1;ranks.append(rank)
            pred.append(dict(id=r['id'],lang=lang,rank=rank,top10=[rows[j]['doc_id'] for j in order[:10]]))
        result[lang]=rank_scores(ranks)
    return result,pred

def rank_pairs(rows,n,seed):
    rng=random.Random(seed);texts=[];docs=[];labels=[];groups=[]
    for i,r in enumerate(rows):
        negatives=[j for j in range(len(rows)) if j!=i];rng.shuffle(negatives)
        ids=[i]+negatives[:n-1];rng.shuffle(ids);groups.append((len(texts),ids.index(i),ids))
        for j in ids:texts.append(r['question']);docs.append(rows[j]['context']);labels.append(int(i==j))
    return texts,docs,labels,groups

def reranking(model,tok,data,seed,reference=None):
    xs={};ys={};groups={};allrows={}
    for s in ['train','validation','test']:
        allrows[s]=data['en'][s]+data['zh'][s]
        parts=[rank_pairs(data[l][s],10 if s=='test' else 4,1234) for l in ['en','zh']]
        texts=parts[0][0]+parts[1][0];docs=parts[0][1]+parts[1][1];ys[s]=parts[0][2]+parts[1][2]
        offset=len(parts[0][0]);groups[s]=parts[0][3]+[(k+offset,p,ids) for k,p,ids in parts[1][3]]
        xs[s]=features(model,tok,texts,docs,length=256)
    head=train_head(xs['train'],ys['train'],xs['validation'],ys['validation'],seed)
    def scores(h):
        with torch.no_grad():s=(h(xs['test'])[:,1]-h(xs['test'])[:,0]).tolist()
        ranks=[]
        for k,p,ids in groups['test']:
            order=sorted(range(10),key=lambda j:(-s[k+j],j));ranks.append(order.index(p)+1)
        return ranks
    ranks=scores(head);result={l:rank_scores(ranks[i*128:(i+1)*128]) for i,l in enumerate(['en','zh'])}
    if reference is not None:
        rr=scores(reference);result['fixed_reference_head']={l:rank_scores(rr[i*128:(i+1)*128]) for i,l in enumerate(['en','zh'])}
    return result,head,[dict(id=r['id'],lang=r['lang'],rank=rank) for r,rank in zip(allrows['test'],ranks)]

@torch.no_grad()
def long_diagnostic(model,tok):
    model.eval();out={};predictions=[]
    # Artificial sequences: balanced A/B counterfactual pairs; neither answer is privileged.
    a,b=tok.encode(' A'),tok.encode(' B')
    assert len(a)==len(b)==1,(a,b)
    answer_ids=[a[0],b[0]]
    filler=tok.encode(' Background facts are unrelated. ')
    for length in [256,1024,8192]:
        for location in ['near','middle','far','removed']:
            preds=[];labels=[];pairs=[]
            for pair in range(4):
                pp=[]
                for label in [0,1]:
                    question=tok.encode(f' The code for item {pair} is')
                    evidence=tok.encode(f' Item {pair} has code')+[answer_ids[label]]+tok.encode('.')
                    seq=([2]+(filler*((length//len(filler))+1)))[:length]
                    query=[3]+question+[4,3];seq[-len(query):]=query
                    pos=8 if location=='far' else (length//2 if location=='middle' else length-len(query)-len(evidence)-4)
                    if location!='removed':seq[pos:pos+len(evidence)]=evidence
                    ids=torch.tensor([seq],device=DEVICE);valid=torch.ones_like(ids,dtype=torch.bool)
                    with amp():h=model(ids,valid);logits=model.logits(h[:,-2])[:,answer_ids]
                    p=int(logits.argmax(-1));preds.append(p);labels.append(label);pp.append(p)
                    predictions.append(dict(length=length,location=location,pair=pair,label=label,pred=p))
                pairs.append(int(pp==[0,1]))
            out[f'{length}_{location}']=dict(accuracy=sum(p==y for p,y in zip(preds,labels))/len(labels),
                counterfactual_pair_accuracy=sum(pairs)/len(pairs),n=len(labels))
    return out,predictions

@torch.no_grad()
def benchmark(model):
    model.eval();result={}
    for n in [256,1024,8192]:
        ids=torch.full((1,n),42,device=DEVICE,dtype=torch.long);valid=ids.ne(0)
        with amp():model(ids,valid)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(3):
            start=time.perf_counter()
            with amp():model(ids,valid)
            torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
        ms=statistics.median(times)
        result[str(n)]=dict(batch=1,median_ms=ms,tokens_per_second=n*1000/ms,
            output_shape=[1,n,model.c.hidden])
    return result

def evaluate(model,tok,cls,qa,seed,label,refs=None):
    start=time.perf_counter();record=dict(seed=seed,variant=label,parameters=sum(p.numel() for p in model.parameters()))
    record['mlm_validation']=mlm_validation(model,tok,qa)
    c,ch,cp=classification(model,tok,cls,seed,None if refs is None else refs[0]);record['classification']=c
    e,eh,ep=extraction(model,tok,qa,seed,None if refs is None else refs[1]);record['extraction']=e
    record['retrieval'],rp=retrieval(model,tok,qa)
    r,rh,rrp=reranking(model,tok,qa,seed,None if refs is None else refs[2]);record['reranking']=r
    record['long_evidence'],lp=long_diagnostic(model,tok)
    record['deployment']=benchmark(model);record['evaluation_seconds']=time.perf_counter()-start
    dump(DEST/f'seed_{seed}/{label}_predictions.json',dict(classification=cp,extraction=ep,retrieval=rp,reranking=rrp,long_evidence=lp))
    dump(DEST/f'seed_{seed}/{label}_metrics.json',record)
    torch.save(dict(classification=ch.state_dict(),extraction=eh.state_dict(),reranking=rh.state_dict()),DEST/f'seed_{seed}/{label}_heads.pt')
    print(json.dumps(dict(seed=seed,variant=label,status='evaluated',seconds=record['evaluation_seconds']),ensure_ascii=False),flush=True)
    return record,(ch,eh,rh)

def main():
    global DEST
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[17,42,71])
    p.add_argument('--base-steps',type=int,default=3000);p.add_argument('--branch-steps',type=int,default=500)
    p.add_argument('--output',default='results/reproduction')
    p.add_argument('--train-only',action='store_true');args=p.parse_args()
    DEST=ROOT/args.output
    for seed in args.seeds:
        folder=DEST/f'seed_{seed}'
        for variant in ['B','E0','E1','E2','E3']:
            if (folder/f'{variant}_metrics.json').exists() and not (folder/f'{variant}.pt').exists():
                raise ValueError('Saved results have no checkpoints. Use a new --output directory to reproduce them.')
    assert torch.cuda.is_available(),'This protocol requires a CUDA GPU.'
    DEST.mkdir(parents=True,exist_ok=True);tok=Tok();cls,qa,chunks=load_data(tok)
    if (DEST/'environment.json').exists():
        previous=json.loads((DEST/'environment.json').read_text(encoding='utf8'))
        assert previous['tokenizer_sha256']==file_hash(OUT/'tokenizer.json'),'Tokenizer changed; use a new output directory.'
        assert previous['dataset_manifest_sha256']==file_hash(OUT/'manifest.json'),'Data changed; use a new output directory.'
    for seed in args.seeds:
        folder=DEST/f'seed_{seed}';folder.mkdir(exist_ok=True)
        for variant in ['B','E0','E1','E2','E3']:
            checkpoint=folder/f'{variant}.pt'
            if not checkpoint.exists():continue
            log_path=folder/f'{variant}_train.json'
            assert log_path.exists(),f'Missing training metadata: {log_path}; use a new output directory.'
            previous=json.loads(log_path.read_text(encoding='utf8'))
            assert previous['steps']==(args.base_steps if variant=='B' else args.branch_steps),'Training steps changed; use a new output directory.'
            ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
            expected=asdict(Config(window=128 if variant=='E2' else 64))
            assert ck['config']==expected and ck['seed']==seed and ck['stage']==variant,'Checkpoint configuration does not match.'
            if 'tokenizer_sha256' in ck:
                assert ck['tokenizer_sha256']==file_hash(OUT/'tokenizer.json') and ck['manifest_sha256']==file_hash(OUT/'manifest.json')
            del ck
        bp=folder/'B.pt'
        if not bp.exists():
            seed_all(seed);model=Encoder().to(DEVICE)
            stat=train(model,tok,chunks,seed,args.base_steps,'B',base=True)
            save_model(bp,model,seed,'B');dump(folder/'B_train.json',stat);del model;torch.cuda.empty_cache()
        for variant in ['E0','E1','E2','E3']:
            path=folder/f'{variant}.pt'
            if path.exists():continue
            model=load_model(bp)
            if variant=='E2':
                model.c.window=128
                for i,block in enumerate(model.blocks):block.window=None if (i+1)%3==0 else 128
            stat=train(model,tok,chunks,seed+100,args.branch_steps,variant)
            save_model(path,model,seed,variant);dump(folder/f'{variant}_train.json',stat)
            del model;gc.collect();torch.cuda.empty_cache()
        if args.train_only:continue
        refs=None
        for variant in ['B','E0','E1','E2','E3']:
            metrics_path=folder/f'{variant}_metrics.json'
            if metrics_path.exists():
                record=json.loads(metrics_path.read_text(encoding='utf-8'))
                if variant=='B':
                    h=torch.load(folder/'B_heads.pt',weights_only=True)
                    refs=(nn.Linear(256,2),nn.Linear(256,2),nn.Linear(512,2))
                    for head,key in zip(refs,['classification','extraction','reranking']):head.load_state_dict(h[key])
            else:
                seed_all(seed);model=load_model(folder/f'{variant}.pt')
                record,heads=evaluate(model,tok,cls,qa,seed,variant,refs)
                if variant=='B':refs=heads
                del model;gc.collect();torch.cuda.empty_cache()
    dump(DEST/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,python=sys.version,
        seeds=args.seeds,base_steps=args.base_steps,branch_steps=args.branch_steps,config=asdict(Config()),
        tokenizer_sha256=file_hash(OUT/'tokenizer.json'),dataset_manifest_sha256=file_hash(OUT/'manifest.json'),
        precision='FP32 parameters and AdamW states; BF16 autocast',
        implementation='PyTorch SDPA with exact masks and query chunks; no FlashAttention kernel enabled'))

if __name__=='__main__':main()
