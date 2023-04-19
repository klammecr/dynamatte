# Third Party
import cv2
import argparse
import os
import subprocess
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torchvision.models.optical_flow import raft_large
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
import torch
import flowpy

# In House

def cv_find_homography(prev_img, img, prev_hom = np.eye(3), prev_seg_mask = None, seg_mask = None):
    if prev_img is None:
        return prev_hom
    
    # Edge case for masks being none
    if prev_seg_mask is None:
        prev_seg_mask = np.ones_like(prev_img)
    if seg_mask is None:
        seg_mask = np.ones_like(img)

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

def extract_and_save_frames(frames, out_path, mask_generator, of_model, seg_masks = None, device = "cuda"):
    # Init variables
    prev_frame    = None
    homographies  = []
    prev_hom      = np.eye(3)
    prev_seg_mask  = None
    for idx, frame in enumerate(frames):

        # Extract out the masks
        # TODO: Check the temporal consistency between the classes, ideally it should be the same if the camera is still and all 
        # objects do not leave the scene/not objects enter the scene. Certainly a limitation because there would need to be an 
        # association step.
        frame_str = str(idx).zfill(4)

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

        if prev_frame is not None:
            # Compute the homography between frames
            prev_hom = homographies[-1].reshape(3,3)
            
            # Convert from OpenCV (BGR) to RGB => Add a batch dimension of size 1 => (N, H, W, C) -> (N, C, H, W)
            prev_img = torch.tensor(prev_frame[..., ::-1].copy()).to(device).unsqueeze(0).permute(0, 3, 1, 2)
            img      = torch.tensor(frame[..., ::-1].copy()).to(device).unsqueeze(0).permute(0, 3, 1, 2)

            # Compute forward and backward flow
            forward_flow  = of_model(prev_img.float(), img.float())
            backward_flow = of_model(img.float(), prev_img.float())

            # Save the flows
            flowpy.flow_write(f"{out_path}/flow/{frame_str}.flo", forward_flow[-1].cpu()[0].detach().numpy(), format = "flo")
            flowpy.flow_write(f"{out_path}/flow_backward/{frame_str}.flo", backward_flow[-1].cpu()[0].detach().numpy(), format = "flo")

            # Clear up temporary CUDA memory
            del img
            del prev_img
            del forward_flow
            del backward_flow

        # Find the homography
        seg_mask = seg_masks[idx]
        H_matrix = cv_find_homography(prev_frame, frame, prev_hom, prev_seg_mask, seg_mask)
        homographies.append(H_matrix.flatten()) 

        # Save the frame to the given folder
        cv2.imwrite(f"{out_path}/rgb/{frame_str}.png", frame)

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
                    "--width",  str(out_size[0]), \
                    "--height", str(out_size[1])])

# Entrypoint
if __name__ == "__main__":
    # Parse the arguments
    ap = argparse.ArgumentParser(
        prog        = "Omnimatte Video Preprocessor",
        description = "Take OpenCV accepted videos and extract out frames to a folder like how Omnimatte likes")
    ap.add_argument("-v", "--video_path")
    ap.add_argument("-o", "--output_path")
    # TODO: Make an optional argument for image size
    args = ap.parse_args()

    # Output image size
    out_size = (448,256)
    #out_size = (854, 480)

    # Read the video
    imgs      = []
    seg_masks = []
    if ".mp4" in args.video_path.split("/")[-1]:
        vc = cv2.VideoCapture(args.video_path)
        while vc.isOpened():
            ret, frame = vc.read()
            if ret:
                # Resize the frame to [256,448]
                frame_rz = cv2.resize(frame, out_size, cv2.INTER_NEAREST)
                imgs.append(frame_rz)
    else:
        for file in sorted(os.listdir(args.video_path)):
            imgs.append(cv2.imread(f"{args.video_path}/{file}"))
            seg_masks.append(cv2.imread(f"datasets/tennis/mask/01/{file}"))

    # Create segmentation object
    sam_model = get_sam_model()
    mask_generator = SamAutomaticMaskGenerator(sam_model)

    # Get optical flow model
    of_model = get_optical_flow_model()

    # Extract out the frames and save them to a specified location at a given size
    create_and_validate_out_path(args.output_path)
    extract_and_save_frames(imgs, args.output_path, mask_generator, of_model, seg_masks)
