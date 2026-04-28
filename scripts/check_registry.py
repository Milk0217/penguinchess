import json
with open('models/model_registry.json', encoding='utf-8') as f:
    d = json.load(f)
for m in d['models']:
    if 'xl' in m['id']:
        elo = m.get('eval', {}).get('elo', '?')
        it = m.get('iteration', '?')
        print(f'  {m["id"]:35s} ELO={elo}  iter={it}')
