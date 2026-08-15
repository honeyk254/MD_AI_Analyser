from fastapi.testclient import TestClient
from md_platform.api.app import app
import time, os, sys

client = TestClient(app)

print('Listing demo examples...')
r = client.get('/api/v1/demo/examples')
print('examples status', r.status_code)
print(r.json())

print('\nSubmitting demo: stable')
r = client.post('/api/v1/demo/stable/submit')
print('submit status', r.status_code)
print(r.json())
if r.status_code != 200:
    sys.exit(1)

run_id = r.json().get('run_id')
print('run_id=', run_id)

# Poll for human_review or completion
for i in range(120):
    s = client.get(f'/api/v1/analysis/{run_id}/status')
    js = s.json()
    print(f"poll {i}:", js.get('status'), js.get('message'))
    if js.get('status') in ('human_review','completed','failed'):
        break
    time.sleep(1)

# Approve if human_review
if js.get('status') == 'human_review':
    print('Approving run...')
    rev = client.post(f'/api/v1/analysis/{run_id}/review', json={'reviewer_signoff': 'demo-approved'})
    print('review response', rev.status_code, rev.json())

# Final status
s = client.get(f'/api/v1/analysis/{run_id}/status')
print('final status', s.status_code, s.json())

# Fetch results
res = client.get(f'/api/v1/analysis/{run_id}/results')
print('results status', res.status_code)
if res.status_code == 200:
    bundle = res.json()
    outdir = os.path.join('data','outputs',run_id)
    htmlpath = os.path.join(outdir, 'analysis_report.html')
    print('HTML report path:', htmlpath)
    if os.path.exists(htmlpath):
        with open(htmlpath, encoding='utf-8') as f:
            content = f.read()
            print('\n--- HTML report preview (first 4000 chars) ---\n')
            print(content[:4000])
    else:
        print('No HTML report found at expected path')
else:
    print('Could not fetch results:', res.text)
