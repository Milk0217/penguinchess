"""Clear old AZ ELO data so MCTS-based evaluation runs fresh."""
import json

with open('models/model_registry.json', encoding='utf-8') as f:
    d = json.load(f)

cleared = 0
for m in d['models']:
    if m['type'] == 'alphazero':
        if 'eval' in m:
            old = m['eval'].get('elo', 'none')
            m['eval'] = {}
            cleared += 1
            print(f'  Cleared {m["id"]} (was ELO={old})')

with open('models/model_registry.json', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)

print(f'Done. Cleared {cleared} AZ models.')
