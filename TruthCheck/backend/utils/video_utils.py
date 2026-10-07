"""Video helpers: frame sampling and face cropping (used by video_detector)."""

import cv2


def sample_frame_indexes(n_frames: int, rate: int = 5, max_frames: int = 32):
    idxs = range(0, max(n_frames, 1), max(rate, 1))
    return list(idxs)[:max_frames]


def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
    return cap
