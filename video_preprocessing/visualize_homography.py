# Third Party
import argparse
import os
import cv2
import numpy as np

# Click through script to validate homography calculation
if __name__ == "__main__":
    # Parse the arguments
    ap = argparse.ArgumentParser(
        prog        = "Homography Visualizer",
        description = "Visualize homography")
    ap.add_argument("-v", "--video_path")
    ap.add_argument("-hf", "--hom_file")
    args = ap.parse_args()

    prev_img = None
    homographies = np.loadtxt(args.hom_file)
    for i, file in enumerate(os.listdir(args.video_path)):
        # Read iamge and homography
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

