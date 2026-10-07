"""Prepare Celeb-DF (v2) videos for MesoNet training.

Celeb-DF v2 structure (default):
    Celeb-DF-v2/
      Celeb-real/         <- real videos
      YouTube-real/       <- real videos
      Celeb-synthesis/    <- deepfake videos

This script samples frames, extracts face crops with MTCNN, and writes them to:

    datasets/celebdf/real/
    datasets/celebdf/fake/

Run:
    python prepare_celebdf.py --src "C:/path/to/Celeb-DF-v2" --frames_per_video 5
"""

import argparse
import os
import glob

import cv2

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='Path to Celeb-DF-v2 folder')
    p.add_argument('--out', default='datasets/celebdf')
    p.add_argument('--frames_per_video', type=int, default=5)
    args = p.parse_args()

    import numpy as np
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=False, select_largest=True)

    real_out = os.path.join(args.out, 'real')
    fake_out = os.path.join(args.out, 'fake')
    os.makedirs(real_out, exist_ok=True)
    os.makedirs(fake_out, exist_ok=True)

    groups = {
        'real': [os.path.join(args.src, 'Celeb-real'),
                 os.path.join(args.src, 'YouTube-real')],
        'fake': [os.path.join(args.src, 'Celeb-synthesis')],
    }

    total_saved = 0
    for label, folders in groups.items():
        out_dir = real_out if label == 'real' else fake_out
        for folder in folders:
            videos = []
            for ext in VIDEO_EXTS:
                videos += glob.glob(os.path.join(folder, '**', f'*{ext}'), recursive=True)
            print(f"[{label}] {len(videos)} videos in {folder}")
            for vi, vpath in enumerate(videos):
                cap = cv2.VideoCapture(vpath)
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if n <= 0:
                    cap.release()
                    continue
                idxs = [int(i) for i in np.linspace(0, n - 1, min(args.frames_per_video, n))]
                fi = 0
                saved_from_video = 0
                while cap.isOpened() and saved_from_video < args.frames_per_video:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if fi in idxs:
                        try:
                            boxes, _ = mtcnn.detect(frame[:, :, ::-1])  # BGR->RGB
                            if boxes is not None and len(boxes) > 0:
                                x1, y1, x2, y2 = [int(v) for v in boxes[0]]
                                face = frame[max(0, y1):y2, max(0, x1):x2]
                                if face.size > 0:
                                    name = f"{label}_{vi}_{saved_from_video}.jpg"
                                    cv2.imwrite(os.path.join(out_dir, name), face)
                                    saved_from_video += 1
                                    total_saved += 1
                        except Exception:
                            pass
                    fi += 1
                cap.release()

    print(f"[done] saved {total_saved} face crops to {args.out}")


if __name__ == '__main__':
    main()
