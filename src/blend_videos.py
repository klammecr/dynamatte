# Third Party
import argparse
import enum
import os
import numpy as np
import cv2

# In House
from src.compositing_poisson import poisson_blend
from src.utils import read_video, read_img

# Enum to differentiate types of blending
class Blending(enum.Enum):
    ePOISSON = 0,
    eDYNAMATTE_BASE = 1
    # eDYNAMTTE_MOD = 2

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        prog        = "Video-to-Video Blending",
        description = "Take OpenCV accepted videos and extract out frames to a folder like how Omnimatte likes")
    ap.add_argument("-s", "--source_video")
    ap.add_argument("-t", "--target_video")
    ap.add_argument("-i", "--image_size", required=False, default=(448, 256), type = tuple)
    args = ap.parse_args()

    # Read the source video
    source_imgs = read_video(args.source_video, args.image_size, filter = "rgba_l1")

    # Read the target video
    if ".mp4" in args.target_video or os.path.isdir(args.target_video):
        target_imgs = read_video(args.target_video, args.image_size)
    else:
        # TODO: Handle if target is just a single, image, repeat many times
        target_imgs = read_img(args.target_video, args.image_size, len(source_imgs))

    
    # Perform frame by frame compositing
    for src, tgt in zip(source_imgs, target_imgs):
        # cv2.imshow("yo", (src >= 200).astype('uint8')*255)
        # cv2.waitKey()
        mask = np.max(src >= 200, axis=2)
        bgr = [poisson_blend(src[:, :, c], tgt[:, :, c], mask) for c in range(src.shape[-1])]
        blend_img_norm = cv2.normalize(cv2.merge(bgr), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        cv2.imshow("Blended Image", blend_img_norm)
        cv2.waitKey()

