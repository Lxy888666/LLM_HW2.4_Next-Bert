"""Untrained full-size native architecture: parameter count and forward smoke test."""
import os
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0')
import gc,json,sys,time,statistics
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.scaled_model import Config,Encoder
from src.scaled_data import ROOT,dump

def main():
    torch.manual_seed(17);torch.cuda.manual_seed_all(17);torch.set_num_threads(4)
    c=Config(vocab=64000,layers=24,hidden=896,heads=14,intermediate=2400,window=512)
    model=Encoder(c).cuda().eval();result=dict(config=asdict(c),parameters=sum(p.numel() for p in model.parameters()),
        seed=17,trained=False,optimizer_step=False,scope='Random initialization, encoder-only forward, no full-vocabulary logits',forward={})
    with torch.no_grad():
        for n in [256,1024,8192]:
            ids=torch.randint(5,64000,(1,n),device='cuda');valid=ids.ne(0)
            with torch.autocast('cuda',dtype=torch.bfloat16):model(ids,valid)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(3):
                start=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(ids,valid)
                torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
                assert list(out.shape)==[1,n,896] and torch.isfinite(out).all()
                del out
            result['forward'][str(n)]=dict(median_ms=statistics.median(times))
    dump(ROOT/'results/validation/target_smoke.json',result);print(json.dumps(result,indent=2),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
