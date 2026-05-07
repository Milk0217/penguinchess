import json
d = json.load(open('models/registry_0672d31.json'))
if isinstance(d, list):
    items = d
elif isinstance(d, dict) and 'models' in d:
    items = d['models']
else:
    items = []
has_elo = [m for m in items if m.get('elo', '----') != '----']
print(f'Total: {len(items)}, With ELO: {len(has_elo)}')
for m in has_elo[:15]:
    print(f'  {m["id"]:35s} ELO={m["elo"]}')
