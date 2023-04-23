# Third Party
import cv2
import os


def read_video(video_path, image_size, filter = ""):
    """
    Read the videos for consumption

    Args:
        video_path (str): Filepath to the video file or path to the rgb frames
        image_size (tuple): The size of image (width, height)

    Returns:
        _type_: _description_
    """
    imgs = []
    if ".mp4" in video_path.split("/")[-1]:
        vc = cv2.VideoCapture(video_path)
        while vc.isOpened():
            ret, frame = vc.read()
            if ret:
                imgs.append(frame)
            else:
                vc.release()
    else:
        for file in sorted(os.listdir(video_path)):
            if filter in file:
                frame = cv2.resize(cv2.imread(f"{video_path}/{file}"), image_size, cv2.INTER_NEAREST)
                imgs.append(frame)
    return imgs


def read_img(img_path, image_size, repeats):
    return [cv2.resize(cv2.imread(img_path), image_size, cv2.INTER_NEAREST) for i in range(repeats)]