from pathlib import Path
import pandas as pd
import pytest
from kestrel.schema import select_fields, is_safe_feature, compose_text
from kestrel.engine import Router, build_pipeline


def make_data():
    products=['fan','purifier','heater']
    teams=['Fan Team','Water Team','Heat Team']
    data=[]
    for i in range(15):
        for p,t in zip(products,teams):
            data.append({'issue_text':f'{p} is broken and needs repair {i%3}',
                         'product_line':p,'team':t})
    return pd.DataFrame(data)


def test_no_outcome_columns_as_features():
    frame=make_data();test=frame.drop(columns='team').copy()
    frame['resolution_notes']='AFTER THE FACT'
    test['resolution_notes']='AFTER THE FACT'
    text,cats=select_fields(frame,test,{})
    assert 'resolution_notes' not in text+cats
    assert 'issue_text' in text
    assert 'product_line' in cats
    assert not is_safe_feature('assigned_team')


def test_router_gives_legal_label_and_honest_probability(tmp_path):
    frame=make_data();X=compose_text(frame,['issue_text'],['product_line'])
    model=build_pipeline(['product_line'],chars=True)
    model.fit(X,frame['team'])
    router=Router(model,['issue_text'],['product_line'],'request_id')
    routed=router.route({'issue_text':'My fan needs repair','product_line':'fan'})
    assert routed['team'] in frame['team'].unique().tolist()
    assert 0<=routed['confidence']<=1
    assert routed['reasons']
    assert len(routed['top_teams'])==3
    assert pytest.approx(sum(r['probability'] for r in routed['top_teams']),abs=0.0005)==1
    path=tmp_path/'model.joblib';router.save(path)
    assert Router.load(path).route({'issue_text':'fan is broken','product_line':'fan'})['team']=='Fan Team'


def test_verified_policy_override_requires_explicit_opt_in():
    frame=make_data();model=build_pipeline(['product_line']).fit(compose_text(frame,['issue_text'],['product_line']),frame['team'])
    record={'issue_text':'broken fan','product_line':'fan'}
    disabled=[{'id':'r1','field':'product_line','value':'fan','operator':'equals','team':'Heat Team','enabled':True,'verified':False,'reason':'unverified'}]
    enabled=[{**disabled[0],'verified':True}]
    assert Router(model,['issue_text'],['product_line'],'request_id',policy=disabled).route(record)['team']=='Fan Team'
    outcome=Router(model,['issue_text'],['product_line'],'request_id',policy=enabled).route(record)
    assert outcome['team']=='Heat Team' and outcome['status']=='POLICY'


def test_reject_empty_ticket():
    frame=make_data();model=build_pipeline(['product_line']).fit(compose_text(frame,['issue_text'],['product_line']),frame['team'])
    with pytest.raises(ValueError): Router(model,['issue_text'],['product_line'],'request_id').route({})
