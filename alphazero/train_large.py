"""Launch AZ ResNetLarge training."""
import sys; sys.path.insert(0, '.')
from alphazero.train import main
import alphazero.train as t
# Override default net class
import penguinchess.ai.alphazero_net as net_mod
t.AlphaZeroResNet = net_mod.AlphaZeroResNetLarge
t.AlphaZeroResNetLarge = net_mod.AlphaZeroResNetLarge
# Also update detect_net_arch
orig_detect = t.detect_net_arch
def detect_with_large(sd):
    try: return orig_detect(sd)
    except: return net_mod.AlphaZeroResNetLarge
t.detect_net_arch = detect_with_large
main()
