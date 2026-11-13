"""Deterministic document-held-out data and train-only byte-level BPE."""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT/'data/raw', ROOT/'data/scaled'
SEED = 20261114
SPECIAL = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def digest(text):
    return hashlib.sha256(' '.join(text.split()).encode('utf-8')).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def qa_read(path):
    raw = json.loads(path.read_text(encoding='utf-8')); out = []
    if isinstance(raw, list):
        for p in raw:
            out.append(dict(context=p['context_text'], group=p.get('title') or p['context_id'],
                qas=[dict(id=q['query_id'], question=q['query_text'], answers=q['answers']) for q in p['qas']]))
    else:
        for a in raw['data']:
            for p in a['paragraphs']:
                out.append(dict(context=p['context'], group=a['title'], qas=[dict(id=q['id'],
                    question=q['question'], answers=[v['text'] for v in q['answers']]) for q in p['qas']]))
    return out

def prepare():
    rng = random.Random(SEED); classifications, questions, corpus, audit = {}, {}, [], {}
    for lang in ['en', 'zh']:
        if lang == 'en':
            tr = pq.ParquetFile(RAW/'sst2_train.parquet').read().to_pylist()
            te = pq.ParquetFile(RAW/'sst2_validation.parquet').read().to_pylist()
            tr = [dict(text=r['sentence'], label=r['label']) for r in tr]
            te = [dict(text=r['sentence'], label=r['label']) for r in te]
            rng.shuffle(tr); va, tr = tr[:2000], tr[2000:]
        else:
            sets = []
            for s in ['train', 'validation', 'test']:
                with pa.memory_map(str(RAW/f'chn_{s}.arrow'), 'r') as f:
                    sets.append(pa.ipc.open_stream(f).read_all().to_pylist())
            tr, va, te = sets
        used, selected = set(), {}
        for s, rows, n in [('test', te, 256), ('validation', va, 128), ('train', tr, 512)]:
            bins = {0:[], 1:[]}
            for r in rows:
                h = digest(r['text'])
                if h in used: continue
                used.add(h)
                bins[int(r['label'])].append(dict(id=h, text=r['text'], label=int(r['label']), lang=lang))
            selected[s] = []
            for b in bins.values():
                rng.shuffle(b); selected[s] += b[:n//2]
            rng.shuffle(selected[s]); assert len(selected[s]) == n
        classifications[lang] = selected
        name = 'squad' if lang == 'en' else 'cmrc'
        pars = qa_read(RAW/f'{name}_train.json')
        groups = sorted({p['group'] for p in pars}); rng.shuffle(groups)
        ng = max(1, len(groups)//10)
        if lang == 'en':
            test = qa_read(RAW/'squad_dev.json')
            blocked = {p['group'] for p in test}
            pars = [p for p in pars if p['group'] not in blocked]
            valgroups = set(groups[:ng]); val = [p for p in pars if p['group'] in valgroups]
            train = [p for p in pars if p['group'] not in valgroups]
        else:
            tg, vg = set(groups[:ng]), set(groups[ng:2*ng])
            test = [p for p in pars if p['group'] in tg]
            val = [p for p in pars if p['group'] in vg]
            train = [p for p in pars if p['group'] not in tg|vg]
        used = set(); selected = {}
        for s, rows, n in [('test',test,128), ('validation',val,64), ('train',train,256)]:
            rows = list(rows); rng.shuffle(rows); chosen = []
            for p in rows:
                h = digest(p['context'])
                if h in used: continue
                used.add(h)
                qs = [q for q in p['qas'] if q['answers'] and p['context'].find(q['answers'][0]) >= 0]
                if not qs: continue
                q = qs[0]
                chosen.append(dict(id=q['id'], doc_id=h, group=p['group'], context=p['context'],
                    question=q['question'], answers=q['answers'], answer_start=p['context'].find(q['answers'][0]), lang=lang))
                if len(chosen) == n: break
            assert len(chosen) == n, (lang,s,len(chosen))
            selected[s] = chosen
        # Exclude every held-out paragraph, including paragraphs not sampled for a task.
        blocked = {digest(p['context']) for p in test+val} | {r['doc_id'] for rs in selected.values() for r in rs}
        for p in train:
            h = digest(p['context'])
            if h in blocked: continue
            blocked.add(h); corpus.append(dict(id=h, lang=lang, text=p['context']))
            if sum(c['lang']==lang for c in corpus) >= 3000: break
        questions[lang] = selected
        audit[lang] = dict(qa_source='SQuAD train/dev' if lang=='en' else 'CMRC train, title-disjoint 80/10/10 split',
            corpus_documents=sum(c['lang']==lang for c in corpus))
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            assert not ({r['id'] for r in classifications[lang][a]} & {r['id'] for r in classifications[lang][b]})
            assert not ({r['doc_id'] for r in selected[a]} & {r['doc_id'] for r in selected[b]})
            assert not ({r['group'] for r in selected[a]} & {r['group'] for r in selected[b]})
    dump(OUT/'classification.json', classifications); dump(OUT/'qa.json', questions); dump(OUT/'corpus.json', corpus)
    # The tokenizer never sees validation/test questions, paragraphs or classification texts.
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=8192, min_frequency=2, special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    texts = [c['text'] for c in corpus]
    for lang in ['en','zh']:
        texts += [r['text'] for r in classifications[lang]['train']]
        texts += [r['context'] for r in questions[lang]['train']] + [r['question'] for r in questions[lang]['train']]
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.post_processor = processors.TemplateProcessing(single='[CLS] $A [SEP]',
        pair='[CLS] $A [SEP] $B:1 [SEP]:1', special_tokens=[('[CLS]',2),('[SEP]',3)])
    tokenizer.save(str(OUT/'tokenizer.json'))
    tok_stats = {}
    for lang in ['en','zh']:
        raw = [c['text'] for c in corpus if c['lang']==lang]
        sizes = [len(tokenizer.encode(t,add_special_tokens=False).ids) for t in raw]
        tok_stats[lang] = dict(tokens=sum(sizes), characters=sum(map(len,raw)), mean_tokens=sum(sizes)/len(sizes),
            unk_tokens=sum(tokenizer.encode(t).ids.count(1) for t in raw))
    dump(OUT/'manifest.json', dict(seed=SEED, raw_sha256={p.name:file_hash(p) for p in sorted(RAW.iterdir()) if p.is_file()},
        prepared_sha256={p.name:file_hash(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'},
        classification_counts={l:{s:len(rs) for s,rs in ss.items()} for l,ss in classifications.items()},
        qa_counts={l:{s:len(rs) for s,rs in ss.items()} for l,ss in questions.items()}, audit=audit,
        tokenizer=tok_stats, vocab_size=tokenizer.get_vocab_size(), checks='classification text, QA paragraph/title split intersections empty; tokenizer train-only',
        limitations=['Exact normalized text deduplication; near-duplicate leakage not fully excluded.',
            'CMRC uses a local held-out split, not its official leaderboard test set.']))
    print(json.dumps({'audit':audit,'tokenizer':tok_stats},ensure_ascii=False), flush=True)

if __name__ == '__main__': prepare()
