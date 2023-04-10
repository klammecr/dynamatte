# Third Party
import cv2
import argparse
import os
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torchvision.models.optical_flow import raft_large
import torch

# In House

def cv_find_homography(prev_img, img):
    # Find the keypoints with ORB
    orb = cv2.ORB_create()
    kp_curr, desc_curr  = orb.detectAndCompute(img)
    kp_prev, desc_prev  = orb.detectAndCompute(prev_img)

    # Find correspondences
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(desc_prev, desc_curr), key = lambda x : x.distance)

    # Get keypoints
    prev_pts  = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    curr_pts  = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # Find the forward homography
    M, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)

def create_and_validate_out_path(out_path):
    try:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    except:
        raise IOError("Output path could not be validated")
    
    return True

def get_sam_model(model_type = "vit_h", device = "cuda"):
    if not os.path.exists("models"):
        os.makedirs("models")

    # Support for all types of ViT and their checkpoints
    if model_type == "vit_h":
        sam_checkpoint = "sam_vit_h_4b8939.pth"
    else:
        ValueError(f"Model type: {model_type} not supported")

    # Grab the model for the internet
    if not os.path.exists(f"models/{sam_checkpoint}"):
        os.system(f"wget -P models/ https://dl.fbaipublicfiles.com/segment_anything/{sam_checkpoint}")

    return sam_model_registry[model_type](checkpoint=f"models/{sam_checkpoint}").to(device)

def get_optical_flow_model(device = "cuda"):
    model = raft_large(pretrained=True).to(device)
    model = model.eval()
    return model

def extract_and_save_frames(vc, out_path, mask_generator, of_model, out_size = (448,256), device = "cuda"):
    frame_num     = -1
    prev_frame_rz = None
    while vc.isOpened():
        frame_num += 1
        frame_str = str(frame_num).zfill(4)
        ret, frame = vc.read()
        
        if ret:
            # Resize the frame to [256,448]
            frame_rz = cv2.resize(frame, out_size, cv2.INTER_NEAREST)

            # Extract out the masks
            masks = mask_generator.generate(frame_rz)
            # TODO: Process and save each mask

            
            if prev_frame_rz is not None:
                # Convert from OpenCV (BGR) to RGB => Add a batch dimension of size 1 => (N, H, W, C) -> (N, C, H, W)
                prev_img = torch.tensor(prev_frame_rz[..., ::-1].copy()).to(device).unsqueeze(0).permute(0, 3, 1, 2)
                img      = torch.tensor(frame_rz[..., ::-1].copy()).to(device).unsqueeze(0).permute(0, 3, 1, 2)

                # Compute forward and backward flow
                forward_flow  = of_model(prev_img.float(), img.float())
                backward_flow = of_model(img.float(), prev_img.float())

                # Save the flows
                torch.save(forward_flow, f"{out_path}/flow/{frame_str}.flo")
                torch.save(backward_flow, f"{out_path}/flow_backward/{frame_str}.flo")

            # Save the frame to the given folder
            cv2.imwrite(f"{out_path}/{str(frame_num)}.png", frame)

            # Set previous frame for optical flow
            prev_frame_rz = frame_rz


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

    # Create segmentation object
    sam_model = get_sam_model()
    mask_generator = SamAutomaticMaskGenerator(sam_model)

    # Get optical flow model
    of_model = get_optical_flow_model()

    # Extract out the frames and save them to a specified location at a given size
    create_and_validate_out_path(args.output_path)
    extract_and_save_frames(vc, args.output_path, mask_generator, of_model)

    # Compute the forward and backward flow

    # Compute the homographies for the frames

