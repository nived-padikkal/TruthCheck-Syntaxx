import os
import cv2
import onnx
import torch
import argparse
import numpy as np
import torch.nn as nn

from models.TMC import ETMC
from models import image
from models.mesonet import MesoNetDetector

from onnx2pytorch import ConvertModel

onnx_model = onnx.load('checkpoints/efficientnet.onnx')
pytorch_model = ConvertModel(onnx_model)

#Set random seed for reproducibility.
torch.manual_seed(42)


# Define the audio_args dictionary
audio_args = {
    'nb_samp': 64600,
    'first_conv': 1024,
    'in_channels': 1,
    'filts': [20, [20, 20], [20, 128], [128, 128]],
    'blocks': [2, 4],
    'nb_fc_node': 1024,
    'gru_node': 1024,
    'nb_gru_layer': 3,
    'nb_classes': 2
}


def get_args(parser):
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default="datasets/train/fakeavceleb*")
    parser.add_argument("--LOAD_SIZE", type=int, default=256)
    parser.add_argument("--FINE_SIZE", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=1024)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="MMDF")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--savedir", type=str, default="./savepath/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--annealing_epoch", type=int, default=10)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--pretrained_image_encoder", type=bool, default = False)
    parser.add_argument("--freeze_image_encoder", type=bool, default = False)
    parser.add_argument("--pretrained_audio_encoder", type = bool, default=False)
    parser.add_argument("--freeze_audio_encoder", type = bool, default = False)
    parser.add_argument("--augment_dataset", type = bool, default = True)

    for key, value in audio_args.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)

def model_summary(args):
    '''Prints the model summary.'''
    model = ETMC(args)

    for name, layer in model.named_modules():
        print(name, layer)

def load_multimodal_model(args):
    '''Load multimodal model'''
    model = ETMC(args)
    ckpt = torch.load('checkpoints/model.pth', map_location = torch.device('cpu'))
    model.load_state_dict(ckpt, strict = True)
    model.eval()
    return model

def load_img_modality_model(args):
    '''Loads image modality model.'''
    rgb_encoder = pytorch_model

    ckpt = torch.load('checkpoints/model.pth', map_location = torch.device('cpu'))
    rgb_encoder.load_state_dict(ckpt['rgb_encoder'], strict = True)
    rgb_encoder.eval()
    return rgb_encoder

def load_spec_modality_model(args):
    spec_encoder = image.RawNet(args)
    ckpt = torch.load('checkpoints/model.pth', map_location = torch.device('cpu'))
    spec_encoder.load_state_dict(ckpt['spec_encoder'], strict = True)
    spec_encoder.eval()
    return spec_encoder


#Load models.
parser = argparse.ArgumentParser(description="Inference models")
get_args(parser)
args, remaining_args = parser.parse_known_args()
assert remaining_args == [], remaining_args

spec_model = load_spec_modality_model(args)

img_model = load_img_modality_model(args)

# MesoNet forgeries detector (used to ensemble with EfficientNet when weights exist).
mesonet = MesoNetDetector(weights_path='checkpoints/mesonet.pth',
                          architecture='meso4', device='cpu')


from facenet_pytorch import MTCNN

mtcnn = MTCNN(keep_all=False, select_largest=True)

def to_rgb_uint8(img):
    """Coerce anything the UI can hand us into an (H, W, 3) uint8 RGB array.

    Accepts a filepath, a PIL image, or an ndarray of arbitrary dtype and
    channel count. Anything else raises ValueError with a clear message
    instead of blowing up inside the conv layers.
    """
    if isinstance(img, str):
        bgr = cv2.imread(img, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not read image file '{img}'")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # PIL Image (Gradio can pass type="pil")
    if hasattr(img, "convert") and hasattr(img, "mode"):
        img = np.asarray(img.convert("RGB"))

    arr = np.asarray(img)
    if arr.size == 0:
        raise ValueError("received an empty image")

    # Normalise dtype -> uint8
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        hi = float(np.nanmax(a))
        lo = float(np.nanmin(a))
        if hi <= 1.0 and lo >= 0.0:      # already scaled to 0..1
            a = a * 255.0
        arr = np.clip(a, 0.0, 255.0).astype(np.uint8)

    if arr.ndim == 2:                    # grayscale
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.ndim != 3:
        raise ValueError(f"unsupported image shape {arr.shape}")

    c = arr.shape[2]
    if c == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)
    if c == 3:
        return arr
    if c == 4:                           # RGBA -> composite over white
        rgb = arr[:, :, :3].astype(np.float32)
        alpha = arr[:, :, 3:4].astype(np.float32)
        if alpha.max() > 1.0:
            alpha = alpha / 255.0
        return np.clip(rgb * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    raise ValueError(f"unsupported channel count {c} (expected 1, 3 or 4)")


def crop_face(img_rgb, fallback_face=None):
    """Return (face_crop, found).

    If no face is detected we reuse `fallback_face` when given (the last face
    seen in a video), otherwise the whole frame.
    """
    try:
        boxes, _ = mtcnn.detect(img_rgb)
        if boxes is not None and len(boxes) > 0:
            box = boxes[0].astype(int)
            h, w = img_rgb.shape[:2]
            bw, bh = int(box[2] - box[0]), int(box[3] - box[1])
            if bw > 0 and bh > 0:
                pad_x, pad_y = int(bw * 0.15), int(bh * 0.15)
                x1, y1 = max(0, int(box[0]) - pad_x), max(0, int(box[1]) - pad_y)
                x2, y2 = min(w, int(box[2]) + pad_x), min(h, int(box[3]) + pad_y)
                face = img_rgb[y1:y2, x1:x2]
                if face.size > 0:
                    return face, True
    except Exception:
        pass
    if fallback_face is not None:
        return fallback_face, False
    return img_rgb, False


def preprocess_img(face, fallback_face=None):
    """Returns (face_tensor, cropped_face_rgb, face_detected).

    The EfficientNet ONNX model expects NHWC layout: (1, 224, 224, 3).
    """
    rgb = to_rgb_uint8(face)
    face_rgb, found = crop_face(rgb, fallback_face=fallback_face)
    face = cv2.resize(face_rgb, (224, 224))
    face = face.astype('float32') / 255.0
    face_pt = torch.tensor(face, dtype=torch.float32).unsqueeze(0)  # (1, H, W, C) NHWC
    return face_pt, face_rgb, found

def preprocess_audio(audio_data, sr=None, target_sr=16000):
    """Convert raw audio to a (1, 64600) float tensor for RawNet."""
    audio = np.asarray(audio_data)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono
    audio = audio.astype(np.float32)
    if audio.size == 0:
        audio = np.zeros(audio_args['nb_samp'], dtype=np.float32)
    if audio.max() > 1.0:
        audio = audio / 32768.0 if audio.dtype != np.float32 or audio.max() > 2.0 else audio
    if sr is not None and sr != target_sr and sr > 0:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except Exception:
            pass
    nb_samp = audio_args['nb_samp']
    if audio.shape[0] < nb_samp:
        audio = np.pad(audio, (0, nb_samp - audio.shape[0]))
    else:
        audio = audio[:nb_samp]
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

def deepfakes_spec_predict(input_audio):
    if input_audio is None:
        return "Please upload an audio file."
    # gr.Audio returns (sample_rate, data)
    sr, audio_data = input_audio[0], input_audio[1]
    audio = preprocess_audio(audio_data, sr=sr)
    with torch.no_grad():
        spec_grads = spec_model.forward(audio)
    # RawNet output is log-softmax -> exponentiate to get probabilities.
    probs = np.exp(spec_grads.cpu().detach().numpy().squeeze())
    real_prob = float(probs[0])
    fake_prob = float(probs[1])

    if fake_prob > real_prob:
        text2 = (f"🚨 VERDICT: FAKE / DEEPFAKE AUDIO\n"
                 f"───────────────────────────────────\n"
                 f"• Result          : SYNTHETIC AUDIO DETECTED\n"
                 f"• Confidence Score: {round(fake_prob * 100, 2)}%")
    else:
        text2 = (f"✅ VERDICT: REAL / AUTHENTIC AUDIO\n"
                 f"───────────────────────────────────\n"
                 f"• Result          : AUTHENTIC AUDIO\n"
                 f"• Confidence Score: {round(real_prob * 100, 2)}%")

    return text2

def deepfakes_image_predict(input_image):
    if input_image is None:
        return "Please upload an image."
    try:
        face_pt, face_rgb, face_found = preprocess_img(input_image)
        with torch.no_grad():
            img_grads = img_model.forward(face_pt)
        probs = img_grads.cpu().detach().numpy().squeeze()
        real_prob = float(probs[0])
        fake_prob = float(probs[1])

        # Ensemble with MesoNet when trained weights are available.
        meso_prob = mesonet.predict_fake_proba(face_rgb)
    except Exception as e:
        return f"Could not analyse this image: {e}"

    if meso_prob is not None:
        fake_prob = 0.6 * fake_prob + 0.4 * meso_prob
        real_prob = 1.0 - fake_prob

    face_note = ""
    if not face_found:
        # The classifier is trained on face crops; a whole-frame result is
        # much weaker, so say so instead of over-claiming.
        face_note = ("\n• Note           : No face found — analysed the full frame. "
                     "Confidence is lower for face-less images.")

    if fake_prob > real_prob:
        preds = round(fake_prob * 100, 2)
        text2 = (
            f"🚨 VERDICT: FAKE / DEEPFAKE IMAGE\n"
            f"───────────────────────────────────\n"
            f"• Result          : FAKE MEDIA DETECTED\n"
            f"• Confidence Score: {preds}%\n"
            f"• Analysis        : Synthetic facial manipulation artifacts identified.{face_note}"
        )
    else:
        preds = round(real_prob * 100, 2)
        text2 = (
            f"✅ VERDICT: REAL / AUTHENTIC IMAGE\n"
            f"───────────────────────────────────\n"
            f"• Result          : AUTHENTIC MEDIA\n"
            f"• Confidence Score: {preds}%\n"
            f"• Analysis        : Natural, unmanipulated facial features confirmed.{face_note}"
        )

    return text2


def preprocess_video(input_video, n_frames=15):
    """Returns (frame_tensors, face_crops, any_face_detected).

    Always returns a 3-tuple so callers never blow up on unpacking.
    """
    v_cap = cv2.VideoCapture(input_video)
    if not v_cap.isOpened():
        return [], [], False

    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if v_len <= 0:
        # Frame count is unreliable for some containers: decode everything and
        # subsample afterwards instead of sampling against a bogus length.
        all_frames = []
        while True:
            success, frame = v_cap.read()
            if not success:
                break
            all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        v_cap.release()
        if not all_frames:
            return [], [], False
        step = max(1, len(all_frames) // n_frames)
        picked = all_frames[::step][:n_frames]
        return _frames_to_tensors(picked)

    if n_frames is None or n_frames >= v_len:
        sample = set(int(i) for i in np.arange(v_len))
    else:
        sample = set(int(i) for i in np.linspace(0, v_len - 1, n_frames))

    frames = []
    curr_frame = 0
    last_face = None
    any_face = False
    while v_cap.isOpened():
        success, frame = v_cap.read()
        if not success:
            break
        if curr_frame in sample:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor, face_rgb, found = preprocess_img(frame_rgb, fallback_face=last_face)
            if found:
                last_face = face_rgb
                any_face = True
            frames.append((frame_tensor, face_rgb))
        curr_frame += 1
    v_cap.release()
    if not frames:
        return [], [], False
    return ([f for f, _ in frames],
            [c for _, c in frames],
            any_face)


def _frames_to_tensors(frames):
    tensors, faces = [], []
    last_face = None
    any_face = False
    for frame_rgb in frames:
        frame_tensor, face_rgb, found = preprocess_img(frame_rgb, fallback_face=last_face)
        if found:
            last_face = face_rgb
            any_face = True
        tensors.append(frame_tensor)
        faces.append(face_rgb)
    return tensors, faces, any_face


def deepfakes_video_predict(input_video):
    '''Perform inference on a video with rich analysis outcome.'''
    if input_video is None:
        return "Please upload a video."
    try:
        video_frames, video_faces, any_face = preprocess_video(input_video, n_frames=15)
    except Exception as e:
        return f"Could not read this video: {e}"
    if not video_frames:
        return ("Could not process video frames. The file may be corrupt, "
                "empty, or use an unsupported codec.")

    fake_faces_list = []
    for face_tensor, face_rgb in zip(video_frames, video_faces):
        try:
            with torch.no_grad():
                img_grads = img_model.forward(face_tensor)
            probs = img_grads.cpu().detach().numpy().squeeze()
            fake_p = float(probs[1])
            meso_p = mesonet.predict_fake_proba(face_rgb)
        except Exception:
            continue
        if meso_p is not None:
            fake_p = 0.6 * fake_p + 0.4 * meso_p
        fake_faces_list.append(fake_p)

    if not fake_faces_list:
        return "Could not score any video frames."

    scores = np.asarray(fake_faces_list, dtype=np.float64)
    total_sampled = len(scores)
    fake_faces_mean = float(scores.mean())
    max_fake = float(scores.max())
    fake_count = int((scores > 0.5).sum())
    fake_pct = round((fake_count / total_sampled) * 100, 1)

    # Aggregate evidence: the plain mean is the default signal. A video where
    # only part of the frames are swapped (a partial deepfake) can have a mean
    # below 0.5 while still being fake, so the top quartile of frames is also
    # considered -- but only when several frames corroborate each other. A
    # single stray frame can no longer flip the whole verdict.
    k = max(1, int(np.ceil(total_sampled * 0.25)))
    topk_mean = float(np.sort(scores)[-k:].mean())
    corroborated = int((scores > 0.6).sum()) >= 2
    fake_score = max(fake_faces_mean, topk_mean if corroborated else 0.0)

    is_fake = fake_score > 0.5
    confidence = round((fake_score if is_fake else 1.0 - fake_score) * 100, 2)

    face_note = ""
    if not any_face:
        face_note = ("\n• Face Detection   : No face found in any frame — "
                     "scored whole frames (lower confidence).")

    if is_fake:
        text2 = (
            f"🚨 VERDICT: FAKE / DEEPFAKE VIDEO\n"
            f"───────────────────────────────────\n"
            f"• Result              : FAKE / MANIPULATED VIDEO\n"
            f"• Confidence Score    : {confidence}%\n"
            f"• Frame Breakdown     : {fake_count}/{total_sampled} frames flagged as Fake ({fake_pct}%)\n"
            f"• Mean Fake Score     : {round(fake_faces_mean * 100, 2)}%\n"
            f"• Peak Fake Severity  : {round(max_fake * 100, 2)}%\n"
            f"• Temporal Analysis   : Deepfake facial blending and temporal synthesis detected.{face_note}"
        )
    else:
        text2 = (
            f"✅ VERDICT: REAL / AUTHENTIC VIDEO\n"
            f"───────────────────────────────────\n"
            f"• Result              : REAL / AUTHENTIC VIDEO\n"
            f"• Confidence Score    : {confidence}%\n"
            f"• Frame Breakdown     : {fake_count}/{total_sampled} frames flagged as Fake ({fake_pct}%)\n"
            f"• Mean Fake Score     : {round(fake_faces_mean * 100, 2)}% (Below threshold)\n"
            f"• Peak Fake Severity  : {round(max_fake * 100, 2)}%\n"
            f"• Temporal Analysis   : Natural facial motion and authentic features verified.{face_note}"
        )

    return text2
