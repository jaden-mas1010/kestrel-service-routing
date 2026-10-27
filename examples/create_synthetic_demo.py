"""OPTIONAL smoke-test only. Data is FICTIONAL and never belongs in final submission.

Run: python examples/create_synthetic_demo.py /tmp/kestrel_fake_data
"""
import csv
import sys
from pathlib import Path
from datetime import date, timedelta

TEAMS={
 'Helpdesk Alpha':('air fryer','not heating properly','online'),
 'Helpdesk Bravo':('water purifier','water leaking from valve','outlet'),
 'Helpdesk Charlie':('robot vacuum','wheels stuck and device stopped','online'),
 'Helpdesk Delta':('mixer grinder','motor makes loud noise','online'),
 'Helpdesk Echo':('cooktop','ignition button does not work','outlet'),
 'Helpdesk Foxtrot':('ceiling fan','remote no longer responds','online'),
 'Helpdesk Golf':('room heater','heater smells burnt','outlet'),
}

def main(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    labelled=[];new=[];submission=[];resolved=[]
    for group,(team,(product,symptom,channel)) in enumerate(TEAMS.items()):
        for i in range(24):
            rid=f'FAKE-{group+1:02}-{i:03}'
            row={'request_id':rid,'created_at':(date(2025,1,1)+timedelta(days=i*10+group)).isoformat(),
                 'issue_text':f'{product}: {symptom}. Please assist. Ticket variation {i%4}',
                 'product_line':product,'channel':channel}
            if i>=21:
                new.append(row);submission.append({'request_id':rid,'team':''})
            else:
                labelled.append({**row,'assigned_team':team})
                resolved.append({'request_id':rid,'resolved_team':team})
    for name,rows in [('train_requests.csv',labelled),('new_requests.csv',new),('sample_submission.csv',submission),('resolution_log.csv',resolved)]:
        with (folder/name).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print('Created clearly fictional training data in',folder)

if __name__=='__main__':
    main(sys.argv[1] if len(sys.argv)>1 else '/tmp/kestrel_fake_data')
