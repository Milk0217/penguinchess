import os
from penguinchess.model_registry import register_model

xl_models = [
    ('az_resnet_xl_iter_5', 5, 'alphazero/alphazero_resnet_xl_iter_5.pth'),
    ('az_resnet_xl_iter_10', 10, 'alphazero/alphazero_resnet_xl_iter_10.pth'),
    ('az_resnet_xl_iter_20', 20, 'alphazero/alphazero_resnet_xl_iter_20.pth'),
    ('az_resnet_xl_iter_30', 30, 'alphazero/alphazero_resnet_xl_iter_30.pth'),
    ('az_resnet_xl_best', 999, 'alphazero/alphazero_resnet_xl_best.pth'),
    ('az_resnet_xl_final', 50, 'alphazero/alphazero_resnet_xl_final.pth'),
    ('az_resnet_xl_iter_50', 50, 'alphazero/alphazero_resnet_xl_iter_50.pth'),
]

for mid, it, fpath in xl_models:
    full = 'models/' + fpath
    if not os.path.exists(full):
        print(f'  SKIP {mid}: {full} not found')
        continue
    sz = os.path.getsize(full)
    try:
        register_model(mid, 'alphazero', fpath, iteration=it, arch='resnet_xl')
        print(f'  OK {mid:35s} {sz/1024/1024:.0f}MB')
    except Exception as e:
        print(f'  FAIL {mid}: {e}')
