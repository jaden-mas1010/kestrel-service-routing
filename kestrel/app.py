"""Local operator console and JSON route API. Start: uvicorn kestrel.app:app --reload"""
from __future__ import annotations
import json
import os
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from .engine import Router

ROOT=Path(__file__).parent.parent
app=FastAPI(title='Kestrel RouteDesk',version='0.1.0',description='Explainable, locally hosted service-request routing')

class RoutingRequest(BaseModel):
    record: dict = Field(description='A single service-request record with original column names')

@lru_cache(maxsize=1)
def current_router():
    model=ROOT/'models'/'router.joblib'
    if not model.exists(): return None
    return Router.load(model)

@app.get('/',response_class=HTMLResponse)
def homepage():
    return (ROOT/'static'/'index.html').read_text(encoding='utf-8')

@app.get('/api/health')
def health():
    return {'service':'up','model_loaded':current_router() is not None,'no_paid_api_required':True}

@app.get('/api/status')
def status():
    router=current_router()
    if router is None:
        return {'ready':False,'message':'Model not built yet. Add the supplied files in data/ and run python -m kestrel.train --data data --out .','teams':[],'text_columns':[],'category_columns':[]}
    audit=ROOT/'reports'/'metrics.json'
    metrics=json.loads(audit.read_text(encoding='utf-8')) if audit.exists() else {}
    return {'ready':True,'message':'Routing model ready','teams':router.teams,
            'text_columns':router.text_cols,'category_columns':router.cat_cols,
            'threshold':router.threshold,'validation':{key:metrics.get(key) for key in
                ('accuracy','macro_f1','sample_size','errors','selected_model','validation_type','review_coverage_at_0_75')}}

@app.post('/api/route')
def route(payload: RoutingRequest):
    router=current_router()
    if router is None: raise HTTPException(status_code=503,detail='Model is not trained. Follow README.md, then restart the service.')
    try:
        return router.route(payload.record)
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc

@app.get('/api/examples')
def examples():
    """Sample real unlabeled records when local data exists, not bundled mock tickets."""
    router=current_router()
    if router is None: return {'rows':[]}
    try:
        import pandas as pd
        from .schema import choose_data_files, load_config
        cfg=load_config(ROOT/'config.json')
        _,test,_=choose_data_files(Path(os.getenv('KESTREL_DATA_DIR',str(ROOT/'data'))),cfg)
        rows=pd.read_csv(test,nrows=8,low_memory=False).fillna('').to_dict('records')
        # Do not send email/contact numbers or sensitive operational columns to the UI.
        whitelisted=set(router.text_cols+router.cat_cols+[router.id_col])
        return {'rows':[{k:v for k,v in row.items() if k in whitelisted} for row in rows]}
    except (FileNotFoundError,ValueError):
        return {'rows':[]}
