"""Strict schema inference. Explicit overrides in config.json take precedence.

Only columns known at the instant of routing can enter the model. A resolver's
notes, outcome, routed team, ticket resolution, or post-contact actions cannot.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re
import pandas as pd

TEXT_TOKENS = ("subject", "description", "issue", "problem", "complaint", "message", "symptom", "details", "customer_note", "body", "query", "request_text", "remarks", "text")
CAT_TOKENS = ("product", "category", "sku", "model", "warranty", "shield", "plan", "channel", "purchase_channel", "priority", "issue_type", "service_type", "request_type", "language", "city", "state", "region", "symptom_code")
FORBIDDEN = ("resolved", "resolution", "assigned", "routing", "routed", "agent", "closed", "closure", "refund_approved", "outcome", "final_team", "sla", "answer", "reply", "reviewed", "post_", "after_")
TEAM_CANDIDATES = ("assigned_team", "routed_to", "routed_team", "original_team", "routing_team", "team", "team_name", "queue", "department", "target_team")
RES_TEAM_CANDIDATES = ("resolved_team", "resolving_team", "resolved_by_team", "final_resolving_team", "final_team", "resolution_team", "team_resolved")
ID_CANDIDATES = ("request_id", "ticket_id", "case_id", "service_request_id", "sr_id", "id")
DATE_CANDIDATES = ("created_at", "request_date", "created_date", "submitted_at", "opened_at", "date_created", "timestamp", "created_on")


def find_col(df, candidates, preferred=None):
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"Configured column {preferred!r} missing. Available: {list(df.columns)}")
        return preferred
    names = {str(c).casefold(): c for c in df.columns}
    for c in candidates:
        if c in names:
            return names[c]
    return None


def is_safe_feature(name: str) -> bool:
    key = name.casefold().replace(' ', '_')
    return not any(token in key for token in FORBIDDEN) and not re.search(r"(^|_)(id|email|phone|customer_id|address|serial|imei|invoice|tracking|payment)(_|$)", key)


def select_fields(train: pd.DataFrame, test: pd.DataFrame, config: dict):
    both = [c for c in train.columns if c in test.columns]
    def default_list(tokens):
        return [c for c in both if is_safe_feature(c) and any(t in str(c).casefold() for t in tokens)]
    texts = config.get('text_columns') if config.get('text_columns') is not None else default_list(TEXT_TOKENS)
    cats = config.get('category_columns') if config.get('category_columns') is not None else default_list(CAT_TOKENS)
    for c in list(texts) + list(cats):
        if c not in both:
            raise ValueError(f"Feature {c!r} must be in both train and test")
        if not is_safe_feature(c):
            raise ValueError(f"Post-routing/identifying column {c!r} cannot be used as a feature")
    cats = [c for c in cats if c not in texts]
    # Limit cardinality: free-text-like columns do not become one-hot identifiers.
    cats = [c for c in cats if train[c].fillna('').astype(str).nunique() <= min(150, max(15, len(train)//5))]
    if not texts and not cats:
        raise ValueError('No valid pre-routing features discovered. Set text_columns/category_columns in config.json after reading data_dictionary.')
    return list(texts), list(cats)


def choose_data_files(folder: Path, cfg: dict):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Data folder not found: {folder}")
    csvs = sorted(folder.glob('*.csv'))
    sample = folder / (cfg.get('sample_file') or 'sample_submission.csv')
    if not sample.exists():
        raise FileNotFoundError(f"Expected sample_submission.csv under {folder}. Include it to guarantee the exact scoring format.")
    sample_df = pd.read_csv(sample, nrows=10)
    if len(sample_df.columns) != 2:
        raise ValueError(f"Expected 2 columns in sample_submission.csv, got {list(sample_df.columns)}. Inspect and update importer.")
    other = [x for x in csvs if x.name not in {sample.name, (cfg.get('resolution_file') or 'resolution_log.csv')}]
    def explicit(key):
        name = cfg.get(key)
        return folder / name if name else None
    train_p, test_p = explicit('train_file'), explicit('test_file')
    if not train_p:
        patterns = ('train', 'historical', 'past', 'labelled', 'labeled')
        train_p = next((p for p in other if any(t in p.stem.lower() for t in patterns)), None)
    if not test_p:
        patterns = ('test', 'unlabelled', 'unlabeled', 'new_requests', 'recent', 'predict', 'incoming')
        test_p = next((p for p in other if p != train_p and any(t in p.stem.lower() for t in patterns)), None)
    # If generic filenames, identify test by matching sample submission ID values.
    if not test_p:
        sample_id = sample_df.columns[0]
        ids = set(sample_df.iloc[:,0].astype(str).head(10))
        for p in other:
            if p == train_p: continue
            head = pd.read_csv(p, nrows=30, low_memory=False)
            col = find_col(head, [sample_id, *ID_CANDIDATES])
            if col and len(ids.intersection(set(head[col].astype(str)))) >= min(2,len(ids)):
                test_p = p; break
    if not train_p and test_p:
        remain = [p for p in other if p != test_p]
        if len(remain)==1: train_p=remain[0]
    if not test_p and train_p:
        remain = [p for p in other if p != train_p]
        if len(remain)==1: test_p=remain[0]
    if not train_p or not test_p or train_p==test_p:
        raise ValueError(f"Cannot safely identify train/test files from {[p.name for p in csvs]}; set train_file and test_file in config.json.")
    if not train_p.exists() or not test_p.exists():
        raise FileNotFoundError(f"Missing data file: train={train_p} test={test_p}")
    return train_p,test_p,sample


def load_config(path='config.json'):
    p=Path(path)
    return json.loads(p.read_text(encoding='utf8')) if p.exists() else {}


def compose_text(frame: pd.DataFrame, text_cols, cat_cols):
    # Scikit input is a data frame with a single text plus categorical cells.
    out = pd.DataFrame(index=frame.index)
    if text_cols:
        out['text'] = frame[text_cols].fillna('').astype(str).apply(
            lambda row: ' '.join(f'{name}: {str(v)[:2500]}' for name,v in row.items() if str(v).strip()), axis=1)
    else:
        out['text'] = ''
    for col in cat_cols:
        out[col]=frame[col].fillna('UNKNOWN').astype(str).str.slice(0,100).replace('', 'UNKNOWN')
    return out


def format_issue(row, texts):
    return ' '.join(str(row.get(c, '')) for c in texts if str(row.get(c,''))!='nan')[:3000]
