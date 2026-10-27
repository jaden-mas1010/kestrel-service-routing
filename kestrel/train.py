"""Local training, chronological validation, honest audit, and submission export."""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from .engine import Router, build_pipeline, baseline_pipeline, evaluation_details, score_model
from .schema import choose_data_files, select_fields, find_col, load_config, compose_text, TEAM_CANDIDATES, RES_TEAM_CANDIDATES, ID_CANDIDATES, DATE_CANDIDATES


def split_three(frame: pd.DataFrame, labels: pd.Series, date_col=None):
    n=len(frame)
    if n<45: raise ValueError('Need at least 45 labelled records for a credible 3-way split. Inspect your data and choose another validation scheme.')
    mode='chronological'
    if date_col:
        date=pd.to_datetime(frame[date_col], errors='coerce',utc=True)
        if date.notna().mean()<.90:
            warnings.warn('More than 10% missing/unparseable dates. Using stratified random split instead.')
            mode='stratified random (missing timestamps)'
        else:
            order=np.argsort(date.fillna(pd.Timestamp('1970-01-01',tz='UTC')).to_numpy(),kind='stable')
            build=order[:int(n*.70)]; dev=order[int(n*.70):int(n*.85)]; hold=order[int(n*.85):]
            if set(labels.iloc[build])==set(labels):
                return build,dev,hold,mode
            warnings.warn('New routing teams appear after training cutoff; time split cannot learn them. Using stratified random split and disclosing this limitation.')
            mode='stratified random (new classes in late records)'
    else: mode='stratified random (no timestamp)'
    try:
        indices=np.arange(n)
        build,remaining=train_test_split(indices,test_size=.30,random_state=42,stratify=labels)
        dev,hold=train_test_split(remaining,test_size=.5,random_state=42,stratify=labels.iloc[remaining])
        return np.sort(build),np.sort(dev),np.sort(hold),mode
    except ValueError:
        warnings.warn('Rare teams make stratification impossible. Using seeded random split. Per-class estimates may be unstable.')
        rng=np.random.default_rng(42); ix=rng.permutation(n)
        return np.sort(ix[:int(.70*n)]),np.sort(ix[int(.70*n):int(.85*n)]),np.sort(ix[int(.85*n):]),'seeded random (rare classes)'


def train(args):
    home=Path(args.out).resolve(); data=Path(args.data).resolve()
    home.mkdir(parents=True,exist_ok=True)
    (home/'reports').mkdir(exist_ok=True)
    cfg=load_config(args.config)
    train_p,test_p,sample_p=choose_data_files(data,cfg)
    labelled=pd.read_csv(train_p,low_memory=False)
    unknown=pd.read_csv(test_p,low_memory=False)
    sample=pd.read_csv(sample_p,low_memory=False)
    target=find_col(labelled,TEAM_CANDIDATES,cfg.get('target_column'))
    rid=find_col(labelled,ID_CANDIDATES,cfg.get('id_column'))
    date_col=find_col(labelled,DATE_CANDIDATES,cfg.get('date_column'))
    if not target or not rid: raise ValueError('Cannot identify ID and routing target columns; provide id_column and target_column in config.json.')
    if rid not in unknown: raise ValueError(f'Test data missing train ID column {rid!r}; adjust config to real field names.')
    if labelled[rid].isna().any() or labelled[rid].astype(str).duplicated().any():
        raise ValueError('Train IDs must be present and unique (duplicate service requests could leak across validation).')
    if unknown[rid].isna().any() or unknown[rid].astype(str).duplicated().any():
        raise ValueError('Unlabeled IDs must be present and unique.')
    sample_id,sample_target=sample.columns.tolist()
    if sample_target not in [target, 'team', 'assigned_team', 'routed_team', 'team_name', 'predicted_team']:
        warnings.warn(f'Output team column {sample_target!r} differs from train target {target!r}; retaining supplied sample submission schema.')
    if sample[sample_id].astype(str).duplicated().any(): raise ValueError('Sample IDs must be unique.')
    if len(sample)!=len(unknown):
        raise ValueError(f'Sample has {len(sample)} rows, unlabeled data has {len(unknown)}; do not export misaligned predictions.')
    if set(sample[sample_id].astype(str))!=set(unknown[rid].astype(str)):
        raise ValueError('Sample and unlabeled IDs do not match exactly; do not produce a misaligned submission.')
    if labelled[target].isna().any(): raise ValueError(f'Label column {target!r} contains missing values.')
    y=labelled[target].astype(str).str.strip()
    if len(y.unique())!=7:
        warnings.warn(f'Found {len(y.unique())} classes, not the stated seven. Check team mapping and labels before submitting.')
    if (y=='').any(): raise ValueError('Blank class labels present')
    texts,cats=select_fields(labelled,unknown,cfg)
    X=compose_text(labelled,texts,cats)
    if not X['text'].str.strip().any():
        raise ValueError('No usable issue description text found. Inspect mappings before training.')
    build,dev,hold,method=split_three(labelled,y,date_col)
    print(f'Data: {len(labelled):,} labelled; {len(unknown):,} unlabelled; 7 expected teams')
    print(f'Validation: {method}; build={len(build)}, dev={len(dev)}, audit={len(hold)}')
    print(f'Features: text={texts} category={cats}')
    scores={}
    candidates={'word-only':baseline_pipeline(), 'word+character+metadata':build_pipeline(cats,chars=True),
                'word+metadata':build_pipeline(cats,chars=False)}
    for name,model in candidates.items():
        print(f'Training {name}...',flush=True)
        model.fit(X.iloc[build],y.iloc[build])
        scores[name]=score_model(model,X.iloc[dev],y.iloc[dev])
    majority=y.iloc[build].value_counts().idxmax()
    majority_dev_accuracy=float((y.iloc[dev]==majority).mean())
    best_name=max(scores,key=lambda name:(scores[name]['accuracy'],scores[name]['macro_f1']))
    best=candidates[best_name]
    # Selection finished: audit is touched once, not used to tune the model.
    best.fit(X.iloc[np.r_[build,dev]],y.iloc[np.r_[build,dev]])
    audit, audit_rows=evaluation_details(best,X.iloc[hold],y.iloc[hold],labelled.iloc[hold],rid)
    audit_rows.to_csv(home/'reports'/'audit_cases.csv',index=False)
    audit['validation_type']=method
    audit['selection_dev_sample_size']=len(dev)
    audit['audit_sample_size']=len(hold)
    audit['selection_scores']=scores
    audit['majority_baseline_dev_accuracy']=majority_dev_accuracy
    audit['selected_model']=best_name
    audit['teams']=sorted(y.unique().tolist())
    # Wilson score interval: calibrated as binomial sampling uncertainty only.
    from math import sqrt
    z=1.96
    n_a=audit['sample_size']; p_a=audit['accuracy']; denom=1+z*z/n_a
    mid=(p_a+z*z/(2*n_a))/denom
    margin=z*sqrt(p_a*(1-p_a)/n_a+z*z/(4*n_a*n_a))/denom
    audit['accuracy_95pct_wilson_interval']=[round(mid-margin,4),round(mid+margin,4)]
    audit['training_file']=train_p.name
    audit['unlabelled_file']=test_p.name
    audit['sample_file']=sample_p.name
    audit['date_column']=date_col
    audit['text_columns']=texts
    audit['category_columns']=cats
    audit['generated_on_data_not_synthetic']=True
    audit['duplicate_texts_across_build_audit']=int(len(set(X['text'].iloc[build]) & set(X['text'].iloc[hold])))
    if audit['duplicate_texts_across_build_audit']:
        warnings.warn('Identical issue texts appear in build and audit. Inspect whether repeat cases inflate accuracy.')
    # Historical vs actual resolved owner is a diagnostic ONLY, never an input feature.
    resolution_file=data/(cfg.get('resolution_file') or 'resolution_log.csv')
    if resolution_file.exists():
        resolved=pd.read_csv(resolution_file,low_memory=False)
        res_id=find_col(resolved,ID_CANDIDATES,cfg.get('resolution_id_column'))
        resolved_col=find_col(resolved,RES_TEAM_CANDIDATES,cfg.get('resolution_team_column'))
        if res_id and resolved_col:
            unique=resolved[[res_id,resolved_col]].drop_duplicates(subset=[res_id])
            mapped=labelled[[rid,target]].merge(unique,how='inner',left_on=rid,right_on=res_id,suffixes=('_assigned','_resolved'))
            if len(mapped):
                matches=(mapped[target].astype(str)==mapped[resolved_col].astype(str))
                audit['historical_vs_resolution']={'linked':len(mapped),'agreement':round(float(matches.mean()),4),
                    'disagreement':int((~matches).sum()),'note':'Resolved owner may differ legitimately from initially assigned owner. Investigate, do not train on post-resolution fields.'}
    policy_json=home/'policy_rules.json'
    policy=json.loads(policy_json.read_text(encoding='utf-8')) if policy_json.exists() else {}
    all_rules=policy.get('rules',[])
    for r in all_rules:
        if r.get('enabled') and r.get('verified') and r.get('team') not in set(y):
            raise ValueError(f'Policy rule {r.get("id")} refers to unknown team {r.get("team")!r}')
    # Final retrain on ALL labels to maximize hidden scoring ability.
    final=build_pipeline(cats,chars=(best_name=='word+character+metadata')) if best_name!='word-only' else baseline_pipeline()
    final.fit(X,y)
    threshold=float(cfg.get('minimum_confidence',0.75))
    router=Router(final,texts,cats,rid,threshold=threshold,policy=all_rules)
    (home/'models').mkdir(exist_ok=True)
    router.save(home/'models'/'router.joblib')
    # Holdout confidence applies to the validated model; auto-route policy overlays
    # were not applied to that benchmark, so do not claim they were evaluated.
    audit['enabled_policy_rules']=[r.get('id',None) for r in all_rules if r.get('enabled') and r.get('verified')]
    if audit['enabled_policy_rules']:
        warnings.warn('Policy overrides are not included in model holdout metrics. Evaluate combined policy+model on held-out cases before deployment.')
    # EXACT formatting and row order of sample_submission.csv.
    records={str(rec[rid]): rec for rec in unknown.to_dict(orient='records')}
    predictions=[]
    for request_id in sample[sample_id].astype(str):
        route=router.route(records[request_id])
        if route['team'] not in router.teams: raise ValueError('Unknown predicted team; abort export')
        predictions.append(route['team'])
    submission=pd.DataFrame({sample_id:sample[sample_id],sample_target:predictions})
    if submission.isna().any().any() or len(submission)!=len(sample):
        raise ValueError('Output failed submission checks')
    submission.to_csv(home/'predictions.csv',index=False)
    (home/'reports'/'metrics.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    (home/'reports'/'EVIDENCE.md').write_text(build_evidence(audit),encoding='utf-8')
    (home/'RITU_MEMO.md').write_text(build_memo(audit),encoding='utf-8')
    print('\nSELECTED MODEL:',best_name)
    print(f'AUDIT: accuracy={audit["accuracy"]:.2%} macro-F1={audit["macro_f1"]:.3f} on {len(hold)} rows; errors={audit["errors"]}')
    print(f'AUTO-ROUTE AT 75%: volume={audit["auto_volume_at_0_75"]}/{len(hold)}, accuracy={audit["auto_accuracy_at_0_75"]}')
    print('Saved predictions.csv, models/router.joblib, reports/EVIDENCE.md, reports/metrics.json, RITU_MEMO.md')
    return audit


def build_evidence(a):
    perteam='\n'.join(f'| {r["team"]} | {r["count"]} | {r["recall"]:.1%} |' for r in a['per_team'] if r['recall'] is not None)
    return f'''# Evaluation evidence (measured, not guessed)\n\n**Audit:** {a['sample_size']} completely held-out requests; {a['correct']} correct; {a['errors']} incorrect.\n\n**Top-1 accuracy:** {a['accuracy']:.2%} (95% Wilson interval: {a['accuracy_95pct_wilson_interval'][0]:.1%}–{a['accuracy_95pct_wilson_interval'][1]:.1%}; sampling uncertainty only). **Macro-F1:** {a['macro_f1']:.3f}. **Weighted F1:** {a['weighted_f1']:.3f}.\n\n**Split:** {a['validation_type']}; selected among 3 candidate architectures using separate {a['selection_dev_sample_size']} development records; audit set had {a['audit_sample_size']} records.\n\n**Selected model:** {a['selected_model']}. **Majority-class baseline accuracy on development set:** {a['majority_baseline_dev_accuracy']:.2%}. **Issue text columns:** {a['text_columns']}. **Categorical columns:** {a['category_columns']}.\n\n**Reason for accuracy metric:** The task scores the exact team for each request; top-1 accuracy measures how often that one team matches the labelled outcome. Macro-F1 is a guardrail for underperforming minority teams.\n\n**Per-team audit recall:**\n\n| Team | Audit cases | Recall |\n|---|---:|---:|\n{perteam}\n\n**Confidence triage (model only):** {a['auto_volume_at_0_75']} of {a['sample_size']} audit cases scored ≥75% model confidence, with {('%.2f%%'%(a['auto_accuracy_at_0_75']*100)) if a['auto_accuracy_at_0_75'] is not None else 'N/A'} accuracy on those cases. Low-confidence share: {a['review_coverage_at_0_75']:.1%}. This threshold was not calibrated separately.\n\n**Data leakage check:** {a['duplicate_texts_across_build_audit']} repeated exact feature-text strings between initial build and audit sets. Inspect repeats for overoptimism.\n\n**Limitation:** The audit is an estimate of future accuracy. The unseen hidden-label score can differ, particularly if current incoming requests differ from historic requests. A real pilot must log reassignment and final resolution outcomes.\n\n## Classification report\n\n```\n{a['classification_report']}\n```\n\n## Candidate development scores (do not misrepresent these as the final audit)\n\n```json\n{json.dumps(a['selection_scores'],indent=2)}\n```\n'''


def build_memo(a):
    acc=a['accuracy']; n=a['sample_size']; miss=a['errors']
    go=acc>=.90 and all((x['recall'] or 0)>=.75 for x in a['per_team'] if x['count']>=5)
    decision=('Proceed with a limited, supervised pilot; do not cancel today.' if go
              else 'Do not cancel the current bot yet; begin a supervised shadow trial.')
    review=a['review_coverage_at_0_75']
    return f'''# Kestrel Home | Decision note for Ritu Deshpande\n\n**Decision — {decision}**\n\nWe built a local service-request router using Kestrel’s own historical request labels. It suggests one of the seven existing teams, explains which request details support that choice, and flags uncertain tickets for human review. No paid AI account is required to run it.\n\n**The number — {acc:.1%} first-choice matches** on {n:,} held-out historical requests ({miss:,} mismatches). This is {('at or above' if acc>=.90 else 'below')} the proposed 90% target **on this test**, not a guarantee about next month’s tickets. A manual-review threshold flags roughly {review:.0%} of this holdout at the chosen cut-off; the cut-off itself still needs operational tuning.\n\n**The rupees — ₹3,20,000/year is the maximum gross licence spend we could avoid**, if the current routing bot can be completely retired. That is **not net savings**. Subtract staff review time, hosting, upkeep, and the cost of mistakes before approving cancellation. We have not been given reliable costs for those items and should not invent a payback number.\n\n**Next week —** Run both systems side by side for one week; log the route proposed, human corrections, ultimate resolving team, warranty/Shield exceptions, and extra handling time. Ask team leads to inspect the worst-performing queues and every policy exception. Set a go/no-go target using first-choice accuracy **and** costly misroutes, then only discuss cancellation after reliable live results.\n\n**Known limitation —** Old assignment labels may reflect past process errors or later handoffs. The historical holdout cannot prove savings or quality of the customer experience.\n'''

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data');p.add_argument('--config',default='config.json');p.add_argument('--out',default='.')
    train(p.parse_args())
