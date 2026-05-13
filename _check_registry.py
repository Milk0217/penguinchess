"""Check registry for AZ model entries."""
import sys
sys.path.insert(0, '.')
from penguinchess.model_registry import get_registry

reg = get_registry()
for m in reg['models']:
    mid = m['id']
    if 'resnet' in mid or 'alphazero' in mid or mid.startswith('az_'):
        ev = m.get('eval') or {}
        print(f"  {mid:25s} elo={str(ev.get('elo','-')):>6}  arch={m.get('arch','?'):8s}  file={m['file']}")
