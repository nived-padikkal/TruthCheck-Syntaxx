import requests

tests = [
    ("REAL?",  "The Federal Reserve raised interest rates by a quarter percentage point on Wednesday following its two-day policy meeting."),
    ("FAKE?",  "BREAKING!!! You wont believe this miracle cure, doctors HATE this one weird trick, click now for free money!!!"),
    ("FAKE?",  "SHOCKING discovery proves the moon landing was faked, scientists finally admit the truth conspiracy 100% proven"),
    ("REAL?",  "Rainfall in the Amazon basin was 12% below the ten-year average last month, according to data released by the national weather institute."),
    ("FAKE?",  "Miracle weight loss drink melts belly fat overnight, celebrities are furious that this is being hidden from the public."),
    ("REAL?",  "Parliament voted 312 to 189 to approve the budget bill after a debate that lasted nearly nine hours."),
]
for name, t in tests:
    r = requests.post('http://127.0.0.1:5002/api/detect/text', data={'text': t}, timeout=300).json()
    print(f"{name:6s} -> {r['prediction']:11s} conf={r['confidence']:<6} fake_p={r['fake_probability']:<6} {r['model']}")
