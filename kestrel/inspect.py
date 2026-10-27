"""Inventory supplied pack without altering it. python -m kestrel.inspect --data data"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .schema import choose_data_files, load_config, select_fields, find_col, ID_CANDIDATES, TEAM_CANDIDATES, DATE_CANDIDATES

def inspect(data='data', config='config.json'):
    folder=Path(data); cfg=load_config(config)
    print('\nKESTREL / DATA READINESS CHECK')
    print('='*62)
    if not folder.exists():
        print(f'Missing directory: {folder.resolve()}'); return False
    for p in sorted(folder.iterdir()):
        if p.is_file():
            print(f'{p.name:36} {p.stat().st_size/1024:9.1f} KB')
    try: train,test,sample=choose_data_files(folder,cfg)
    except Exception as exc:
        print(f'\nDATA FILE MAPPING NEEDED: {exc}')
        return False
    train_df=pd.read_csv(train,low_memory=False)
    test_df=pd.read_csv(test,low_memory=False)
    sample_df=pd.read_csv(sample,low_memory=False)
    print(f'\nTrain file: {train.name}: {len(train_df):,} rows')
    print(f'Test file:  {test.name}: {len(test_df):,} rows')
    print(f'Sample:     {sample.name}: {list(sample_df.columns)}')
    print('\nTraining columns and missing rate:')
    for col in train_df.columns:
        print(f'  {col:<34} missing {train_df[col].isna().mean():6.1%}  cardinality {train_df[col].nunique():>7}')
    print('\nUnlabeled/new request columns:',list(test_df.columns))
    y_col=find_col(train_df, TEAM_CANDIDATES,cfg.get('target_column'))
    id_col=find_col(train_df,ID_CANDIDATES,cfg.get('id_column'))
    dt_col=find_col(train_df,DATE_CANDIDATES,cfg.get('date_column'))
    print(f'\nProposed target={y_col!r}, ID={id_col!r}, date={dt_col!r}')
    if y_col:
        print('Class distribution:')
        print(train_df[y_col].value_counts(dropna=False).to_string())
    try:
        texts,cats=select_fields(train_df,test_df,cfg)
        print(f'\nProposed TEXT features: {texts}')
        print(f'Proposed CAT features:  {cats}')
        print('Check against data_dictionary and ops-policy before accepting this.')
    except Exception as exc: print('FEATURE MAPPING NEEDED:',exc)
    print('\nNEXT: inspect ops-policy.pdf, team descriptions, email-thread.txt; edit config.json.')
    return True

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data'); parser.add_argument('--config',default='config.json'); a=parser.parse_args()
    inspect(a.data,a.config)
