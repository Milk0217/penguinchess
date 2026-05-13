"""Clean stale registry entries."""
import sys
sys.path.insert(0, '.')
from penguinchess.model_registry import get_registry, save_registry

reg = get_registry()
before = len(reg['models'])

stale_ids = {'alphazero_best'}
reg['models'] = [m for m in reg['models'] if m['id'] not in stale_ids]

# Dedup by file path for alphazero_best.pth
seen_files = {}
keep = []
for m in reg['models']:
    f = m.get('file', '')
    if f == 'alphazero/alphazero_best.pth':
        mid = m['id']
        if mid in ('az_mlp_best', 'az_best'):
            if mid not in seen_files:
                seen_files[mid] = True
                keep.append(m)
            else:
                print(f"  Remove duplicate: {mid} -> {f}")
        else:
            print(f"  Remove stale: {mid} -> {f}")
    else:
        keep.append(m)

reg['models'] = keep
after = len(reg['models'])
save_registry(reg)
print(f"Cleaned {before - after} stale entries. {before} -> {after}")

# Verify
from penguinchess.model_registry import get_registry
reg2 = get_registry()
for m in reg2['models']:
    if 'best' in m['id']:
        ev = m.get('eval') or {}
        print(f"  {m['id']:25s} elo={str(ev.get('elo','-')):>6}  file={m['file']}")
