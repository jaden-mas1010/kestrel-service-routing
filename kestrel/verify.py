"""Final CSV pre-flight. Run: python -m kestrel.verify --data data"""
import argparse
from pathlib import Path
import pandas as pd
from .schema import choose_data_files, load_config, find_col, TEAM_CANDIDATES, ID_CANDIDATES

def verify(data='data', config='config.json', submission='predictions.csv'):
    data=Path(data);cfg=load_config(config)
    train,test,sample=choose_data_files(data,cfg)
    original=pd.read_csv(sample,low_memory=False)
    output=pd.read_csv(submission,low_memory=False)
    train_df=pd.read_csv(train,low_memory=False)
    team_col=find_col(train_df,TEAM_CANDIDATES,cfg.get('target_column'))
    if not team_col: raise ValueError('Missing label column')
    expected=set(train_df[team_col].dropna().astype(str).str.strip().unique())
    assert list(output.columns)==list(original.columns),f'Header mismatch: {list(output.columns)} vs {list(original.columns)}'
    assert len(output)==len(original), f'Row count mismatch: {len(output)} vs {len(original)}'
    key,team=list(original.columns)
    assert output[key].astype(str).tolist()==original[key].astype(str).tolist(),'ID order changed'
    assert not output.isna().any().any(),'Missing output fields'
    assert set(output[team].astype(str)).issubset(expected),f'Predicted unknown label: {set(output[team].astype(str))-expected}'
    assert output[key].is_unique,'Duplicate IDs'
    print(f'PASS: {len(output):,} predictions, exact submission header+ID order, {len(expected)} permitted labels, no blanks.')
    print('Permitted labels:',sorted(expected))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data');p.add_argument('--config',default='config.json');p.add_argument('--submission',default='predictions.csv')
    a=p.parse_args();verify(a.data,a.config,a.submission)
