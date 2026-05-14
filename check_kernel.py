"""Check Kaggle kernels for penguinchess."""
import requests
h = {'Authorization': 'Bearer CfDJ8KXhJwB4GRRPqJWFH0ywOelZklLenDH5P9E6a_b-7dVHoxfRiPpGd1me_niGXwoHJY27fHrsaizmwpVTGWOxMjuQppccO-2og5ZaPWg3RAJMn36536lfqbXkvDbb7oAHn0lfY43yaI4L'}
r = requests.get('https://www.kaggle.com/api/v1/kernels/list', headers=h)
if r.status_code == 200:
    found = False
    for k in r.json():
        title = k.get('title', '')
        ref = k.get('ref', '')
        if 'penguin' in title.lower():
            found = True
            print('Kernel found!')
            print(f'  Title: {title}')
            print(f'  URL: https://kaggle.com/{ref}')
            print(f'  GPU: {k.get("enableGpu")}')
            print(f'  Status: {k.get("lastRunTime", "?")}')
    if not found:
        print('No penguinchess kernel found in listing')
        print('Kernel titles on your account:')
        for k in r.json()[:10]:
            print(f'  - {k.get("title", "?")}')
else:
    print(f'API Error: {r.status_code}')
