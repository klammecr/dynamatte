# Third Party
import cv2
import argparse
import numpy as np
import os

# In House


def create_and_validate_out_path(out_path):
    try:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    except:
        raise IOError("Output path could not be validated")
    
    return True

def extract_and_save_frames(vc, out_path, out_size = (256,448)):
    frame_num = -1
    while vc.isOpened():
        frame_num += 1
        ret, frame = vc.read()

        if ret:
            # Resize the frame to [256,448]
            frame_rz = np.resize(frame, out_size)

            # Save the frame to the given folder
            cv2.imwrite(f"{out_path}/{str(frame_num)}.png", frame)


# Entrypoint
if __name__ == "__main__":
    # Parse the arguments
    ap = argparse.ArgumentParser(
        prog        = "Omnimatte Video Preprocessor",
        description = "Take OpenCV accepted videos and extract out frames to a folder like how Omnimatte likes")
    ap.add_argument("-v", "--video_path")
    ap.add_argument("-o", "--output_path")
    args = ap.parse_args()

    # Read the video
    vc = cv2.VideoCapture(args.video_path)

    # Extract out the frames and save them to a specified location at a given size
    create_and_validate_out_path(args.output_path)
    extract_and_save_frames(vc, args.output_path)