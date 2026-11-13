"""Download only public data used by the final experiment; verify saved hashes."""
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'data/raw'
DATA={
 'sst2_train.parquet':'https://hf-mirror.com/datasets/nyu-mll/glue/resolve/main/sst2/train-00000-of-00001.parquet',
 'sst2_validation.parquet':'https://hf-mirror.com/datasets/nyu-mll/glue/resolve/main/sst2/validation-00000-of-00001.parquet',
 'chn_train.arrow':'https://hf-mirror.com/datasets/seamew/ChnSentiCorp/resolve/main/chn_senti_corp-train.arrow',
 'chn_validation.arrow':'https://hf-mirror.com/datasets/seamew/ChnSentiCorp/resolve/main/chn_senti_corp-validation.arrow',
 'chn_test.arrow':'https://hf-mirror.com/datasets/seamew/ChnSentiCorp/resolve/main/chn_senti_corp-test.arrow',
 'squad_train.json':'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
 'squad_dev.json':'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json',
 'cmrc_train.json':'https://raw.githubusercontent.com/ymcui/cmrc2018/master/data/cmrc2018_train.json',
}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def main():
 RAW.mkdir(parents=True,exist_ok=True)
 manifest=ROOT/'data/scaled/manifest.json'
 expected=json.loads(manifest.read_text(encoding='utf8'))['raw_sha256'] if manifest.exists() else {}
 for name,url in DATA.items():
  path=RAW/name
  if path.exists() and (name not in expected or sha(path)==expected[name]):
   print('verified existing',name,flush=True);continue
  tmp=path.with_suffix(path.suffix+'.part')
  for attempt in range(3):
   try:
    with urlopen(Request(url,headers={'User-Agent':'next-bert-homework/2.0'}),timeout=60) as r,tmp.open('wb') as f:
     for block in iter(lambda:r.read(1048576),b''):f.write(block)
    if name in expected and sha(tmp)!=expected[name]:raise RuntimeError('SHA256 mismatch: upstream data differs from the submitted experiment')
    tmp.replace(path);print('downloaded',name,flush=True);break
   except Exception as e:
    print(name,str(e),flush=True)
    if attempt==2:raise
    time.sleep(2)
 (RAW/'sources.json').write_text(json.dumps(DATA,indent=2),encoding='utf8')
if __name__=='__main__':main()
