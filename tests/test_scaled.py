import torch
import json
from torch.nn import functional as F
from src.scaled_model import attention, rope
from src.scaled_data import OUT, file_hash
from tokenizers import Tokenizer

def test_chunked_attention_matches_dense_and_gradient():
    torch.manual_seed(17)
    q,k,v=[torch.randn(2,2,19,8,requires_grad=True) for _ in range(3)]
    valid=torch.ones(2,19,dtype=torch.bool); valid[1,15:]=False
    for window in [None,8]:
        qr,kr=rope(q),rope(k)
        mask=valid[:,None,None,:]
        if window:
            delta=torch.arange(19)[None,:]-torch.arange(19)[:,None]
            mask=mask & ((delta>=-4)&(delta<4))[None,None]
        a=attention(qr,kr,v,valid,window,chunk=5)
        b=F.scaled_dot_product_attention(qr,kr,v,attn_mask=mask)
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        ga=torch.autograd.grad(a.square().sum(),(q,k,v),retain_graph=True)
        gb=torch.autograd.grad(b.square().sum(),(q,k,v),retain_graph=True)
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=2e-5,rtol=2e-5)

def test_frozen_data_hashes_and_document_separation():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf8'))
    for name,sha in manifest['prepared_sha256'].items():assert file_hash(OUT/name)==sha
    qa=json.loads((OUT/'qa.json').read_text(encoding='utf8'))
    cls=json.loads((OUT/'classification.json').read_text(encoding='utf8'))
    for lang in ['en','zh']:
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            assert not ({x['group'] for x in qa[lang][a]} & {x['group'] for x in qa[lang][b]})
            assert not ({x['doc_id'] for x in qa[lang][a]} & {x['doc_id'] for x in qa[lang][b]})
            assert not ({x['id'] for x in cls[lang][a]} & {x['id'] for x in cls[lang][b]})

def test_byte_tokenizer_roundtrip_for_unseen_unicode():
    tok=Tokenizer.from_file(str(OUT/'tokenizer.json'))
    text='你好，世界。Rare𠮷🙂 café\n42'
    encoded=tok.encode(text,add_special_tokens=False)
    assert 1 not in encoded.ids
    assert tok.decode(encoded.ids)==text
