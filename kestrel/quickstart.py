"""Fast, local-first RouteDesk training: one reproducible model and an honest chronological audit.

Run from the project root: py -m kestrel.quickstart
No API keys, synthetic training records, or hidden labels required.
"""
from pathlib import Path
import json
import pandas as pd
from .engine import baseline_pipeline, evaluation_details, Router
from .schema import compose_text

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'


def main():
    original = DATA / 'train_cleaned.csv'
    if not original.exists():
        raise FileNotFoundError('data/train_cleaned.csv missing; first run py explore.py')
    train = pd.read_csv(original, low_memory=False)
    test = pd.read_csv(DATA/'test_unlabelled.csv', low_memory=False)
    sample = pd.read_csv(DATA/'sample_submission.csv', low_memory=False)
    needed = {'request_id', 'request_text', 'team_label', 'created_at_ist'}
    if not needed.issubset(train.columns):
        raise ValueError(f'Missing training fields: {needed-set(train.columns)}')
    if len(train.team_label.unique())!=7:
        raise ValueError('Expected exactly 7 cleaned team labels; run explore.py and inspect names')
    if train.request_id.duplicated().any() or test.request_id.duplicated().any():
        raise ValueError('Duplicate request IDs')
    if list(sample.columns)!=['request_id','team']:
        raise ValueError(f'Unexpected template columns: {list(sample.columns)}')
    if set(test.request_id)!=set(sample.request_id):
        raise ValueError('Test and template request IDs do not match')
    # A chronological holdout better approximates forecasting future support requests.
    sorted_train=train.sort_values('created_at_ist', kind='stable').reset_index(drop=True)
    text_cols=['request_text']
    features=compose_text(sorted_train,text_cols,[])
    labels=sorted_train.team_label.astype(str)
    split=int(len(sorted_train)*.8)
    model=baseline_pipeline()
    model.fit(features.iloc[:split],labels.iloc[:split])
    audit, errors=evaluation_details(model,features.iloc[split:],labels.iloc[split:])
    print(f"Chronological audit: {audit['correct']}/{audit['sample_size']} correct; {audit['accuracy']:.2%} accuracy, {audit['macro_f1']:.3f} macro-F1")
    # Fit the same model design on all labelled records after the audit.
    final=baseline_pipeline()
    final.fit(features,labels)
    (ROOT/'models').mkdir(exist_ok=True)
    router=Router(final,text_cols,[],'request_id',threshold=.75)
    router.save(ROOT/'models'/'router.joblib')
    new_features=compose_text(test,text_cols,[])
    predicted=final.predict(new_features)
    by_id=dict(zip(test.request_id.astype(str),predicted))
    out=pd.DataFrame({'request_id':sample.request_id, 'team':[by_id[k] for k in sample.request_id.astype(str)]})
    if out.team.isna().any() or not set(out.team).issubset(set(labels)):
        raise ValueError('Invalid predictions; not writing final CSV')
    out.to_csv(ROOT/'predictions.csv',index=False)
    (ROOT/'reports').mkdir(exist_ok=True)
    metrics={k:audit[k] for k in ('accuracy','macro_f1','weighted_f1','correct','errors','sample_size','per_team','review_coverage_at_0_75','auto_accuracy_at_0_75','auto_volume_at_0_75')}
    metrics.update({'validation_type':'chronological last 20%', 'selected_model':'TF-IDF word 1-2gram + logistic regression', 'train_size':split})
    (ROOT/'reports'/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    perteam='\n'.join(f"- {r['team']}: {r['count']} cases, recall {r['recall']:.1%}" for r in audit['per_team'] if r['recall'] is not None)
    (ROOT/'reports'/'EVIDENCE.md').write_text(f"""# Routing validation evidence

- **Data:** {len(train):,} historical labelled requests; {len(test):,} new requests.
- **Holdout:** chronologically latest {audit['sample_size']:,} rows (20%). Earlier {split:,} used to train the evaluated model.
- **Holdout first-assignment accuracy:** {audit['accuracy']:.2%} ({audit['correct']:,} correct / {audit['errors']:,} wrong).
- **Macro-F1:** {audit['macro_f1']:.3f}; an additional check against weak minority-team performance.
- **Low-confidence review flag (<75% model probability):** {audit['review_coverage_at_0_75']:.1%} of holdout. These scores are not probability-calibrated.
- **Model:** Word unigram/bigram TF-IDF + Logistic Regression, retrained on all historical labels for prediction.
- **Caution:** Metrics measure historical *initial team labels*, NOT correctness of final resolution. Repeated messages and operational drift can bias the estimate. Route by policy and track actual transfers in a shadow pilot.

## Recall by team
{perteam}

## Known failure modes
Ambiguous “please call back” messages, payment mentioned as background rather than a Billing issue, purifier breakdown versus replacement filter, damaged delivery versus technical fault, and renamed teams. Review these cases manually rather than trusting a high model score.
""",encoding='utf-8')
    (ROOT/'RITU_MEMO.md').write_text(f"""# Decision memo | Ritu Deshpande, Head of D2C Operations

**Decision:** Pilot RouteDesk alongside the current bot; do not cancel the licence yet.

**The number:** The local classifier matched {audit['accuracy']:.1%} of original routing labels on {audit['sample_size']:,} later historical requests ({audit['errors']:,} mismatches). This is a test of *bot-label matching*, not evidence that it prevents incorrect transfers. The hidden-file score and live accuracy remain unknown.

**The rupees:** Replacing the vendor licence could avoid at most **₹3,20,000 gross annually**, or ~₹26,667 a month before costs. Each team transfer costs ₹305 and the associated extra contact averages ₹260, so reducing one avoidable misroute could save ~₹565, but no verified reduction is claimed yet. Actual net savings require staff-review effort, hosting, upkeep and errors to be measured. No paid model API is used.

**Next week:** Run a one-week shadow pilot. Log the recommended team, human correction, actual resolving team, number of transfers, and handling costs. Separately monitor warranty claims, installation paid-but-not-done messages, and purifier repair versus filter requests. Obtain a written go/no-go decision based on resolution outcomes and operational cost rather than historical label matching alone.

**Known risk:** Training labels are the original bot's initial assignments, including misroutes. RouteDesk may accurately reproduce those historical errors. Low-information tickets need human review.
""",encoding='utf-8')
    cfg_path=ROOT/'config.json'
    cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
    cfg.update({'train_file':'train_cleaned.csv','test_file':'test_unlabelled.csv','sample_file':'sample_submission.csv', 'target_column':'team_label', 'id_column':'request_id','date_column':'created_at_ist','text_columns':['request_text'],'category_columns':[]})
    cfg_path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    print(f'Export verified: {len(out)} rows, columns {list(out.columns)} -> {ROOT/"predictions.csv"}')
    print('Saved models/router.joblib, reports/EVIDENCE.md, RITU_MEMO.md, config.json')
    print('Next: py -m kestrel.verify --data data --submission predictions.csv')
    print('Then: py -m uvicorn kestrel.app:app --host 127.0.0.1 --port 8000')

if __name__=='__main__':
    main()
