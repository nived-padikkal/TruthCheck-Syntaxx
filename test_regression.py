"""Regression suite for the corrected detection pipeline."""
import os, numpy as np, cv2, torch
from PIL import Image
import inference_2 as inf

fails = []

def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)

print("=" * 72)
print("1. Labeled images (ground truth: lady=REAL, fake_image=FAKE)")
print("=" * 72)
for p, want in [('images/lady.jpg', 'REAL'), ('images/fake_image.jpg', 'FAKE')]:
    r = inf.deepfakes_image_predict(p)
    got = 'FAKE' if 'FAKE / DEEPFAKE' in r else 'REAL'
    check(got == want, f"{p}: got {got} (want {want})")

print()
print("=" * 72)
print("2. Labeled videos (aaa=FAKE, real-1=REAL, bbb=REAL)")
print("=" * 72)
for p, want in [('videos/aaa.mp4', 'FAKE'), ('videos/real-1.mp4', 'REAL'),
                ('videos/bbb.mp4', 'REAL')]:
    r = inf.deepfakes_video_predict(p)
    got = 'FAKE' if 'FAKE / DEEPFAKE' in r else ('REAL' if 'REAL / AUTHENTIC' in r else 'ERR')
    check(got == want, f"{p}: got {got} (want {want})")

print()
print("=" * 72)
print("3. Robustness: inputs that used to crash")
print("=" * 72)
rgb = np.array(Image.open('images/lady.jpg').convert('RGB'))

# preprocess_video used to return a bare [] -> ValueError on unpack
r = inf.preprocess_video('temp_uploads/does_not_exist.mp4')
check(isinstance(r, tuple) and len(r) == 3, f"missing video -> 3-tuple (got {type(r).__name__} len {len(r)})")

# RGBA / grayscale / float / palette fed as ndarray (was: RuntimeError in conv)
for name, arr in [
    ('RGBA ndarray', np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])),
    ('grayscale 2D', cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)),
    ('grayscale 3ch', cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[:, :, None].repeat(3, 2)),
    ('float 0..1', rgb.astype(np.float32) / 255.0),
    ('float 0..255', rgb.astype(np.float32)),
    ('bool', (rgb[:, :, 0] > 128)),
]:
    try:
        inf.deepfakes_image_predict(arr)
        check(True, f"image {name} handled")
    except Exception as e:
        check(False, f"image {name} -> {type(e).__name__}: {e}")

# every odd-but-valid image format
os.makedirs('temp_uploads/img', exist_ok=True)
for name, mode in [('rgba.png', 'RGBA'), ('gray.png', 'L'), ('pal.png', 'P'),
                   ('cmyk.tiff', 'CMYK'), ('1bit.png', '1'), ('la.png', 'LA')]:
    im = Image.open('images/lady.jpg').convert(mode)
    fp = f'temp_uploads/img/{name}'
    im.save(fp)
    try:
        inf.deepfakes_image_predict(fp)
        check(True, f"file {name} handled")
    except Exception as e:
        check(False, f"file {name} -> {type(e).__name__}: {e}")

# None inputs
check(inf.deepfakes_image_predict(None) == "Please upload an image.", "image None -> friendly message")
check(inf.deepfakes_video_predict(None) == "Please upload a video.", "video None -> friendly message")
r = inf.deepfakes_video_predict('temp_uploads/nope.mp4')
check(isinstance(r, str) and 'Could not' in r, f"missing video -> message not crash ({r[:45]})")
r = inf.deepfakes_video_predict('images/lady.jpg')
check(isinstance(r, str), f"image-as-video -> message not crash ({r[:45]})")

print()
print("=" * 72)
print("4. Odd video containers / frame counts")
print("=" * 72)
def write_video(path, fourcc, frames, fps=30, size=(640, 480)):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
    if not vw.isOpened():
        return False
    for i in range(frames):
        img = np.zeros((size[1], size[0], 3), np.uint8)
        img[:] = (60, 70, 90)
        cv2.ellipse(img, (size[0]//2, size[1]//2), (95, 125), 0, 0, 360, (175, 165, 155), -1)
        vw.write(img)
    vw.release()
    return os.path.exists(path) and os.path.getsize(path) > 0

for name, fcc, n in [('mp4v.mp4', 'mp4v', 60), ('MJPG.avi', 'MJPG', 60),
                     ('XVID.avi', 'XVID', 60), ('one.mp4', 'mp4v', 1),
                     ('two.mp4', 'mp4v', 2), ('short.mp4', 'mp4v', 10),
                     ('long.mp4', 'mp4v', 400)]:
    p = f'temp_uploads/{name}'
    if not write_video(p, fcc, n):
        print(f"  SKIP {name} (encoder unavailable)"); continue
    try:
        r = inf.deepfakes_video_predict(p)
        ok = isinstance(r, str) and ('VERDICT' in r or 'Could not' in r)
        check(ok, f"{name} ({n}f) -> {r.splitlines()[0][:50]}")
    except Exception as e:
        check(False, f"{name} ({n}f) -> {type(e).__name__}: {e}")

print()
print("=" * 72)
print(f"RESULT: {len(fails)} failure(s)")
for f in fails:
    print("  -", f)
print("=" * 72)
