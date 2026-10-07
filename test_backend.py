import requests

# video: hit it repeatedly to prove the PermissionError/500 is gone
for i in range(4):
    with open('videos/real-1.mp4', 'rb') as f:
        r = requests.post('http://127.0.0.1:5000/api/detect/video', files={'file': f}, timeout=600)
    j = r.json()
    print(f'video try{i+1}: {r.status_code} pred={j.get("prediction", "ERR")} '
          f'conf={j.get("confidence")} model={j.get("model")}')

# text polarity re-check
tests = [
    ('REAL', 'The Federal Reserve raised interest rates by a quarter point on Wednesday.'),
    ('FAKE', 'BREAKING!!! doctors HATE this one weird trick, click now for free money!!!'),
]
ok = True
for want, t in tests:
    r = requests.post('http://127.0.0.1:5000/api/detect/text', data={'text': t}, timeout=300).json()
    good = r['prediction'] == want
    ok &= good
    print(f'text want={want:4s} got={r["prediction"]:4s} conf={r["confidence"]} '
          f'{"OK" if good else "WRONG"}')

# image sanity
with open('images/fake_image.jpg', 'rb') as f:
    r = requests.post('http://127.0.0.1:5000/api/detect/image', files={'file': f}, timeout=300)
j = r.json()
print(f'image fake_image.jpg: {r.status_code} pred={j.get("prediction")} conf={j.get("confidence")}')

print('ALL TEXT CHECKS PASS' if ok else 'TEXT CHECKS FAILED')
