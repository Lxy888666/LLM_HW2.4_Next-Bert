"""Checks for masking and metrics used by the current reduced experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.span_masking import make_span_mask
from src.metrics import cls_scores, rank_scores

def test_span_budget_and_protected_tokens():
    ids = torch.arange(40).view(2, 20)
    protected = torch.zeros_like(ids, dtype=torch.bool)
    protected[:, :2] = True; protected[:, -2:] = True
    mask = make_span_mask(ids, protected, .15, 3, 8, 7)
    assert mask.sum(1).tolist() == [2, 2]
    assert not (mask & protected).any()
    assert torch.equal(mask, make_span_mask(ids, protected, .15, 3, 8, 7))

def test_metrics_known_values():
    assert cls_scores([0, 1, 1, 0], [0, 1, 0, 0])["accuracy"] == .75
    r = rank_scores([1, 2, 11])
    assert r["recall_at_5"] == 2 / 3
    assert abs(r["mrr"] - (1 + .5 + 1/11) / 3) < 1e-9
