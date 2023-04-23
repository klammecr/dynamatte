# Third Party
import argparse
import os
import cv2
import numpy as np

# In House
from omnimatte.utils import flow_to_image, readFlow

def visualize_flow(forward_file, backward_file):
    f_img = None
    b_img = None
    if forward_file is not None:
        f_flow = readFlow(forward_file)
        f_img  = flow_to_image(f_flow, convert_to_bgr=True)
    if backward_file is not None:
        b_flow = readFlow(backward_file)
        b_img  = flow_to_image(b_flow, convert_to_bgr=True)

    # Visualize the flows
    vis_img = np.hstack((f_img, b_img))
    cv2.imshow("Forward Flow | Backward Flow", vis_img)
    cv2.waitKey()

# Click through script to validate homography and/or flow calculation
if __name__ == "__main__":
    # Parse the arguments
    ap = argparse.ArgumentParser(
        prog        = "Visualizer",
        description = "Visualize homography")
    ap.add_argument("-v", "--video_path")
    ap.add_argument("-hf", "--hom_file", required=True)
    ap.add_argument("-ff", "--for_path", required=False)
    ap.add_argument("-bf", "--bak_path", required=False)
    args = ap.parse_args()

    prev_img = None
    homographies = np.loadtxt(args.hom_file)

    # Read the flows
    ffs = []
    for i, file in enumerate(os.listdir(args.for_path)):
        ffs.append(f"{args.for_path}/{file}")

    bfs = []
    for i, file in enumerate(os.listdir(args.bak_path)):
        bfs.append(f"{args.bak_path}/{file}")

    for ff,bf in zip(ffs, bfs):
        visualize_flow(ff, bf)

    for i, file in enumerate(os.listdir(args.video_path)):
        # Read image and homography
        img = cv2.imread(f"{args.video_path}/{file}")

        if prev_img is not None:
            # Find the frame-to-frame homography from the global
            prev_H = homographies[i - 1].reshape(3,3)
            H      = homographies[i].reshape(3,3)
            final_H = H @ np.linalg.inv(prev_H)

            # Warp the image
            warped_prev = cv2.warpPerspective(prev_img, final_H, (img.shape[1], img.shape[0]))
            warped_prev_gray = cv2.cvtColor(warped_prev, cv2.COLOR_BGR2GRAY)
            img_gray         = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            diff_img = 255 - cv2.absdiff(warped_prev_gray, img_gray)
            cv2.imshow("Difference Image", diff_img)
            cv2.waitKey()

        prev_img = img

