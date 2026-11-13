from __future__ import annotations
import random, torch

def make_span_mask(ids, protected, probability=.15, min_len=3, max_len=8, seed=0):
    rng=random.Random(seed); out=torch.zeros_like(ids,dtype=torch.bool)
    for b in range(ids.size(0)):
        eligible=(~protected[b]).nonzero(as_tuple=False).flatten().tolist(); target=min(len(eligible),max(1,round(len(eligible)*probability))) if eligible else 0
        remaining=set(eligible); selected=set()
        while len(selected)<target and remaining:
            start=rng.choice(sorted(remaining)); span=rng.randint(min_len,max_len)
            for pos in range(start,min(ids.size(1),start+span)):
                if protected[b,pos]: break
                if pos in remaining: selected.add(pos)
                if len(selected)>=target: break
            remaining.difference_update(selected)
        if selected: out[b,sorted(selected)]=True
    return out

def corrupt_batch(batch, tokenizer, mode="token", seed=0, probability=.15):
    ids=batch["input_ids"].clone(); protected=batch["attention_mask"].eq(0)
    for sid in tokenizer.all_special_ids: protected |= ids.eq(int(sid))
    if mode=="span": selected=make_span_mask(ids,protected,probability,3,8,seed)
    elif mode=="token":
        g=torch.Generator(device=ids.device).manual_seed(seed); selected=torch.zeros_like(ids,dtype=torch.bool)
        for b in range(ids.size(0)):
            pos=(~protected[b]).nonzero(as_tuple=False).flatten(); n=min(len(pos),max(1,round(len(pos)*probability))) if len(pos) else 0
            if n: selected[b,pos[torch.randperm(len(pos),generator=g,device=ids.device)[:n]]]=True
    else: raise ValueError(mode)
    labels=ids.masked_fill(~selected,-100); g=torch.Generator(device=ids.device).manual_seed(seed+104729)
    draw=torch.rand(ids.shape,generator=g,device=ids.device); masked=selected&(draw<.8); ids[masked]=int(tokenizer.mask_token_id)
    random_ids=selected&(draw>=.8)&(draw<.9)
    if random_ids.any(): ids[random_ids]=torch.randint(tokenizer.vocab_size,ids.shape,generator=g,device=ids.device,dtype=ids.dtype)[random_ids]
    return ids,labels,selected
