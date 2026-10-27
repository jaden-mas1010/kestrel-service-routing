"""Deterministic, local ticket router. No external services, no paid keys.

Nothing here embeds Kestrel policies or team names: these must come from the
actual supplied data and the signed-off operations policy.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from .schema import compose_text, format_issue


def build_pipeline(cat_cols: list[str], *, chars: bool = True):
    transformers = [('words', TfidfVectorizer(ngram_range=(1,2), min_df=1,
        max_df=0.99, max_features=24000, sublinear_tf=True, strip_accents='unicode'), 'text')]
    if chars:
        transformers.append(('chars', TfidfVectorizer(analyzer='char_wb',
            ngram_range=(3,5), min_df=1, max_features=36000, sublinear_tf=True), 'text'))
    if cat_cols:
        transformers.append(('categories', OneHotEncoder(handle_unknown='ignore',
            min_frequency=2), cat_cols))
    features = ColumnTransformer(transformers, remainder='drop', sparse_threshold=1.0)
    return Pipeline([('features', features),
                     ('classifier', LogisticRegression(C=3.0, max_iter=1200))])


def baseline_pipeline():
    # Useful only as a point of comparison; never shipped as a hidden-score solution.
    return Pipeline([('features', ColumnTransformer([
        ('words', TfidfVectorizer(ngram_range=(1,2), min_df=1), 'text')
    ])), ('classifier', LogisticRegression(max_iter=700))])


def score_model(model, X, y):
    predicted = model.predict(X)
    return {'accuracy': float(accuracy_score(y,predicted)),
            'macro_f1': float(f1_score(y,predicted,average='macro',zero_division=0)),
            'weighted_f1': float(f1_score(y,predicted,average='weighted',zero_division=0))}


def evaluation_details(model, X: pd.DataFrame, y: pd.Series, frame=None, id_column=None):
    pred = model.predict(X)
    probs = model.predict_proba(X)
    confidence = probs.max(axis=1)
    correctness = np.array(pred)==np.array(y)
    labels = list(model.named_steps['classifier'].classes_)
    rows = []
    for label in labels:
        mask = np.array(y)==label
        rows.append({'team':str(label),'count':int(mask.sum()),
                     'recall':float((np.array(pred)[mask]==label).mean()) if mask.sum() else None})
    m = score_model(model,X,y)
    m.update({'sample_size':len(y), 'correct':int(correctness.sum()),
              'errors':int((~correctness).sum()),
              'review_coverage_at_0_75':float((confidence<0.75).mean()),
              'auto_accuracy_at_0_75':float(correctness[confidence>=0.75].mean()) if (confidence>=0.75).any() else None,
              'auto_volume_at_0_75':int((confidence>=0.75).sum()),
              'per_team':rows,
              'confusion_labels':labels,
              'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist(),
              'classification_report':classification_report(y,pred,labels=labels,zero_division=0)})
    errors = pd.DataFrame({'actual_team':np.array(y),'predicted_team':pred,
                           'confidence':np.round(confidence,3), 'correct':correctness})
    if frame is not None:
        if id_column in frame: errors.insert(0,'request_id',frame[id_column].values)
        errors['issue_excerpt'] = [str(s)[:160] for s in X['text'].values]
    return m, errors


def _safe_signals(model, X, best_idx: int, top=5):
    """Inspect weighted *features*, not invented LLM explanations."""
    classifier = model.named_steps['classifier']
    if not hasattr(classifier, 'coef_'):
        return []
    features = model.named_steps['features']
    vector = features.transform(X)
    names = features.get_feature_names_out()
    weights = classifier.coef_
    # LogisticRegression's binary coefficients represent the second class.
    if len(classifier.classes_) == 2 and weights.shape[0] == 1:
        coef = weights[0] if best_idx==1 else -weights[0]
    else:
        coef = weights[best_idx]
    contrib = vector.multiply(coef).tocsr()
    if contrib.nnz == 0: return []
    indices = contrib.indices
    values = contrib.data
    order = np.argsort(values)[::-1]
    signals = []
    for pos in order:
        if values[pos] <= 0: break
        raw = str(names[indices[pos]])
        if raw.startswith('chars__'): continue  # Character n-grams aren't readable.
        raw=raw.replace('words__','').replace('categories__','').replace('_',' ')
        if len(raw)>75: raw=raw[:72]+'...'
        if re.search(r'\b(?:\d{10,}|\S+@\S+)\b',raw): continue
        if raw not in signals:
            signals.append(raw)
        if len(signals)>=top: break
    return signals


class Router:
    def __init__(self, model, text_cols, cat_cols, id_col, threshold=0.75, policy=None):
        self.model=model;self.text_cols=text_cols;self.cat_cols=cat_cols
        self.id_col=id_col;self.threshold=threshold;self.policy=policy or []
        self.teams=[str(x) for x in model.named_steps['classifier'].classes_]

    def route(self, record: dict):
        if not isinstance(record,dict): raise ValueError('Expected a JSON object')
        if not any(str(record.get(c,'')).strip() for c in (self.text_cols+self.cat_cols)):
            raise ValueError('Add at least one issue description or product field to route this request.')
        safe_record={c:record.get(c, '') for c in (self.text_cols+self.cat_cols)}
        X=compose_text(pd.DataFrame([safe_record]),self.text_cols,self.cat_cols)
        probabilities=self.model.predict_proba(X)[0]
        winning=int(np.argmax(probabilities))
        model_team=self.teams[winning]
        confidence=float(probabilities[winning])
        signals=_safe_signals(self.model,X,winning)
        rules=[]; team=model_team
        # Policy rules are opt-in only after verification from the supplied PDF.
        for rule in self.policy:
            if not rule.get('verified',False) or not rule.get('enabled',False): continue
            col=rule.get('field'); actual=str(record.get(col,'')).casefold().strip()
            expected=str(rule.get('value','')).casefold().strip()
            operation=rule.get('operator')
            triggered=(operation=='equals' and actual==expected) or (operation=='contains' and expected and expected in actual)
            if triggered and rule.get('team') in self.teams:
                team=rule['team']
                rules.append(str(rule.get('reason') or rule.get('id') or 'Verified operational policy'))
                break
        state='POLICY' if rules else ('REVIEW' if confidence < self.threshold else 'AUTO_ROUTE')
        reasons=([f'Policy: {r}' for r in rules] if rules else [])
        if signals:
            reasons.append('Evidence in request: '+', '.join(signals[:5])+'.')
        reasons.append(f'Model probability for its recommended team: {confidence:.0%}. This is a statistical estimate, not a guarantee.')
        if state=='REVIEW': reasons.append('Low-confidence case: confirm with a human before dispatch.')
        return {'team':team,'model_team':model_team,'confidence':round(confidence,4),
                'status':state,'reasons':reasons,
                'top_teams':[{'team':self.teams[i],'probability':round(float(probabilities[i]),4)}
                    for i in np.argsort(probabilities)[::-1][:3]],
                'policy_applied':rules,'features_used':{'text':self.text_cols,'categories':self.cat_cols}}

    def save(self,path:Path):
        joblib.dump({'model':self.model,'text_cols':self.text_cols,
            'cat_cols':self.cat_cols,'id_col':self.id_col,
            'threshold':self.threshold,'policy':self.policy},path)

    @classmethod
    def load(cls,path:Path):
        d=joblib.load(path)
        return cls(d['model'],d['text_cols'],d['cat_cols'],d['id_col'],d['threshold'],d.get('policy',[]))
