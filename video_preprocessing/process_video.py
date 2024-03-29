# Third Party
import cv2
import argparse
import os
import subprocess
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torchvision.models.optical_flow import raft_large
import torch
import flowpy

# In House
from src.utils import read_video

def cv_find_homography(prev_img, img, out_size, prev_hom = np.eye(3), prev_seg_mask = None, seg_mask = None):
    if prev_img is None:
        return prev_hom
    
    # Resize
    img = cv2.resize(img, out_size, cv2.INTER_NEAREST)
    prev_img = cv2.resize(prev_img, out_size, cv2.INTER_NEAREST)

    # Edge case for masks being none
    if prev_seg_mask is None:
        prev_seg_mask = np.ones(out_size)
    if seg_mask is None:
        seg_mask = np.ones_like(out_size)

    # Find the keypoints with ORB
    orb = cv2.ORB_create()
    kp_curr, desc_curr  = orb.detectAndCompute(img, mask = 255 - seg_mask[:, :, 0])
    kp_prev, desc_prev  = orb.detectAndCompute(prev_img, mask = 255 - prev_seg_mask[:, :, 0])

    # Find correspondences
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(desc_prev, desc_curr), key = lambda x : x.distance)

    # Get keypoints
    prev_pts  = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    curr_pts  = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # Find the forward homography
    M, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)

    # Obtain the homography w.r.t. the first frame
    cascaded_H = M @ prev_hom
    return cascaded_H

def create_and_validate_out_path(out_path):
    # Omnimatte Spec:
    # output_path/
    #   rgb/ (images)
    #   mask/ (seg_masks)
    #   flow/ (optical flow)
    #   confidence/ (confidence images)
    #   homographies.txt
    try:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            os.makedirs(f"{out_path}/rgb")
            os.makedirs(f"{out_path}/mask")
            os.makedirs(f"{out_path}/flow")
            os.makedirs(f"{out_path}/flow_backward")
            os.makedirs(f"{out_path}/confidence")
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

def extract_and_save_frames(frames, out_path, mask_generator, of_model, out_size, seg_masks = None, device = "cuda"):
    # Init variables
    prev_frame    = None
    homographies  = []
    prev_hom      = np.eye(3)
    prev_seg_mask  = None
    for idx, frame in enumerate(frames):
        # Masks
        # masks = mask_generator.generate(frame)

        # for i, mask in enumerate(masks):
        #     # Create the subdir
        #     if not os.path.exists(f"{out_path}/mask/{str(i).zfill(2)}"):
        #         os.makedirs(f"{out_path}/mask/{str(i).zfill(2)}")

        #     # Save the mask
        #     seg_img = mask['segmentation'].astype("float") * 255
        #     cv2.imwrite(f"{out_path}/mask/{str(i).zfill(2)}/{frame_str}.png", seg_img)
        # del masks
        frame_str = str(idx).zfill(4)
        seg_mask  = seg_masks[idx]
        if prev_frame is not None:
            # Resize
            frame = cv2.resize(frame, out_size, cv2.INTER_NEAREST)
            prev_frame = cv2.resize(prev_frame, out_size, cv2.INTER_NEAREST)
            
            # Compute the homography between frames
            prev_hom = homographies[-1].reshape(3,3)
            
            # Convert from OpenCV (BGR) to RGB => Add a batch dimension of size 1 => (N, H, W, C) -> (N, C, H, W)
            prev_img = torch.tensor(prev_frame[..., ::-1].copy()).to(device).unsqueeze(0).permute(0, 3, 1, 2)
            img      = torch.tensor(frame[..., ::-1].copy()).to(device).unsqueeze(0).permute(0, 3, 1, 2)

            # Compute forward and backward flow
            forward_flow  = of_model(prev_img.float(), img.float())
            backward_flow = of_model(img.float(), prev_img.float())
            forward_flow  = forward_flow[-1].cpu()[0].permute(1, 2, 0).detach().numpy()
            backward_flow = backward_flow[-1].cpu()[0].permute(1, 2, 0).detach().numpy()

            # Save the flows
            flowpy.flow_write(f"{out_path}/flow/{frame_str}.flo", forward_flow, format = "flo")
            flowpy.flow_write(f"{out_path}/flow_backward/{frame_str}.flo", backward_flow, format = "flo")

            # Clear up temporary CUDA memory
            del img
            del prev_img
            del forward_flow
            del backward_flow

        # Find the homography
        H_matrix = cv_find_homography(prev_frame, frame, out_size, prev_hom, prev_seg_mask, seg_mask)
        homographies.append(H_matrix.flatten()) 

        # Save the frame to the given folder
        frame_rz = cv2.resize(frame, out_size, cv2.INTER_NEAREST)
        cv2.imwrite(f"{out_path}/rgb/{frame_str}.png", frame_rz)

        # Set previous frame for optical flow
        prev_frame = frame
        prev_seg_mask = seg_mask

        # Clean up GPU memory each iteration
        torch.cuda.empty_cache()

    # Save the homographies
    np.savetxt(f"{out_path}/homographies.txt", np.array(homographies))

    # Run the confidence.py file to generate confidence images as a subprocess
    subprocess.run(["python", "omnimatte/datasets/confidence.py", "--dataroot", out_path])

    # Run the homography.py file to generate the homography file with bounds as a subprocess
    subprocess.run(["python", \
                    "omnimatte/datasets/homography.py", \
                    "--homography_path", f"{out_path}/homographies.txt", \
                    "--width",  str(frames[0].shape[1]), \
                    "--height", str(frames[0].shape[0])])

# Entrypoint
if __name__ == "__main__":
    # Parse the arguments
    ap = argparse.ArgumentParser(
        prog        = "Omnimatte Video Preprocessor",
        description = "Take OpenCV accepted videos and extract out frames to a folder like how Omnimatte likes")
    ap.add_argument("-v", "--video_path")
    ap.add_argument("-o", "--output_path")
    ap.add_argument("-s", "--seg_mask_path")
    ap.add_argument("-i", "--image_size", required=False, default=(448, 256), type = tuple)
    args = ap.parse_args()

    # Read the video
    imgs      = read_video(args.video_path, args.image_size)
    seg_masks = read_video(args.seg_mask_path, args.image_size)

    # Create segmentation object
    sam_model = get_sam_model()
    mask_generator = SamAutomaticMaskGenerator(sam_model)

    # Get optical flow model
    of_model = get_optical_flow_model()

    # Extract out the frames and save them to a specified location at a given size
    create_and_validate_out_path(args.output_path)
    extract_and_save_frames(imgs, args.output_path, mask_generator, of_model, args.image_size, seg_masks)
