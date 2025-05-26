"""
only pip install two packages
insightface==0.7.3
opencv-python==4.10.0.84
"""

import cv2
import insightface
from insightface.app import FaceAnalysis

# matplotlib only for visualization, don't need in production
# import matplotlib.pyplot as plt

app = FaceAnalysis(providers=["CPUExecutionProvider"])

img_path = "/mnt/d/download/audio_live/f1.png"
# load img
if img_path is not None:
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
else:
    raise ValueError("img_path must be provided")

print(img_bgr.shape)
# adjust det_thresh, liveportrait used 0.5. Given model is deterministic, so we can use 0.5
# anime style often lies between 0.05 to 0.1 det values
app.prepare(ctx_id=0, det_thresh=0.5, det_size=(512, 512))
faces = app.get(img_bgr)  # detected faces, list of faces

print(len(faces))
# if detection is successful, len(faces) > 0,
if len(faces) > 0:  # print the first score, which is detected with the biggest bbox
    print(f"{faces[0].det_score:.4f}")
