# Third Party
import argparse
import enum
import os
import numpy as np
import cv2

# In House
from src.compositing_poisson import poisson_edit ,poisson_blend, preview
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
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (args.image_size[0], args.image_size[1]))
    for src, tgt in zip(source_imgs, target_imgs):
        alpha     = src[:, :, -1] / 255
        alpha_rz = np.tile(alpha.reshape(src.shape[0], src.shape[1], 1), 3)
        bgr_im = src[:, :, 0:3]
        bgr_src = bgr_im * alpha_rz
        # Debug the source image to be passed in
        # cv2.imshow("fff", bgr_src.astype('uint8'))
        # cv2.waitKey()
        mask = alpha >= 0.5
        #blend = poisson_edit(bgr_im, tgt, mask, (0,0))
        blend = cv2.merge([poisson_blend(bgr_im[:, :, c], tgt[:, :, c], mask, alpha) for c in range(bgr_im.shape[-1])])
        blend[blend > 255] = 255
        blend[blend < 0] = 0
        out.write(blend.astype('uint8'))
        # blend_img_norm = cv2.normalize(blend, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        # cv2.imshow("Blended Image", blend.astype('uint8'))
        # cv2.waitKey()
    out.release()
    cv2.destroyAllWindows()


    # Test for a single frame:
    # mask = cv2.imread("mask.jpg")
    # src  = cv2.imread("source.jpg")
    # tgt  = cv2.imread("target.jpg")
    # mask = np.max(mask >= 200, axis=2)

    # bgr  = [poisson_blend(src[:, :, c], tgt[:, :, c], mask) for c in range(src.shape[-1])]
    # #blend_img_norm = cv2.normalize(cv2.merge(bgr), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    # cv2.imshow("Blended Image", cv2.merge(bgr).astype('uint8'))
    # cv2.waitKey()
