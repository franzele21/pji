import argparse

if "__main__" == __name__:
    parser = argparse.ArgumentParser(description="Used to blur things from a bag file")
    parser.add_argument("bag_file_path",        help="Path to the .bag file",       type=str)
    parser.add_argument("yolo_model_path",      help="Path to the Yolov8 model",    type=str)
    parser.add_argument("-o", "--output_file",  help="Path to the output mp4 file", type=str)
    parser.add_argument("--frame_rate",         help="Frame sampling rate",         type=int)
    parser.add_argument("--black_box",          help="Replace the blur with a black box", action="store_true")
    parser.add_argument("--keep_orig_mp4",      help="Keep the original mp4 of the bag video", action="store_true")
    parser.add_argument("-v", "--verbose",      help="Program will print its progress", action="store_true")
    args = parser.parse_args()

from ultralytics import YOLO
import ultralytics
import cv2
import math
from tqdm import tqdm
import os
import gc
import numpy as np
import uuid

import bagpy as bg
from cv_bridge import CvBridge

def blur_box(frame: np.ndarray, 
             box: ultralytics.engine.results.Boxes, 
             black_box: bool=False, 
             min_conf: float=0.3):
    """
    Apply a blur or black box to a specified region of an image if the confidence level is above a threshold.

    Parameters
    ----------
    frame : numpy.ndarray
        The image on which the blur or black box is to be applied. It should be a 3D array representing an RGB image.
    box : object
        An object containing the bounding box coordinates and confidence score. It should have attributes `conf` and `xyxy`:
        - `box.conf` : list or numpy.ndarray
            The confidence score(s) of the bounding box, with values between 0 and 1.
        - `box.xyxy` : list or numpy.ndarray
            The coordinates of the bounding box in the format [x1, y1, x2, y2].
    black_box : bool, optional
        If True, the specified region is filled with a black box instead of being blurred. Default is False.
    min_conf : float, optional
        The minimum confidence threshold to apply the blur or black box. Default is 0.3. Must in [0, 1].

    Returns
    -------
    numpy.ndarray
        The modified image with the blur or black box applied to the specified region.

    Notes
    -----
    - The input `frame` must be a 3D NumPy array representing an image with shape (height, width, channels).
    - The bounding box coordinates and confidence score must be provided in the `box` object (already implemented in the Yolov8 results).
    - If `black_box` is set to True, the region within the bounding box will be replaced with black pixels (faster than blurring).
    - The Gaussian blur applied uses a kernel size of (51, 51) with a standard deviation of 0.
    """
    if math.ceil((box.conf[0]*100))/100 > min_conf:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        h, w = y2-y1, x2-x1

        if black_box:
            blur = np.zeros((h, w, 3))
        else:
            ROI = frame[y1:y1+h, x1:x1+w]
            blur = cv2.GaussianBlur(ROI, (51,51), 0) 
        frame[y1:y1+h, x1:x1+w] = blur
    return frame

def read_video_file(video_path: str):
    """
    Open a video file and retrieve its properties.

    Parameters
    ----------
    video_path : str
        The path to the video file to be read.

    Returns
    -------
    vidcap : cv2.VideoCapture
        The VideoCapture object for the video file.
    frame_count : int
        The total number of frames in the video.
    fps : float
        The frames per second (FPS) of the video.
    frame_size : tuple of int
        A tuple representing the width and height of the video frames in pixels.

    Examples
    --------
    >>> vidcap, frame_count, fps, frame_size = read_video_file('example_video.mp4')
    >>> print(f"Total frames: {frame_count}")
    Total frames: 240
    >>> print(f"FPS: {fps}")
    FPS: 30.0
    >>> print(f"Frame size: {frame_size}")
    Frame size: (1920, 1080)

    Notes
    -----
    - The returned `vidcap` object can be used to read frames from the video using methods like `vidcap.read()`.
    """
    vidcap = cv2.VideoCapture(video_path)
    return vidcap, \
            int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), \
            vidcap.get(cv2.CAP_PROP_FPS), \
            (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

def create_video_file(output_path: str,
                      fps: float, 
                      frame_size: tuple[int, int]):
    """
    Create a video file for writing frames.

    Parameters
    ----------
    output_path : str
        The path where the output video file will be saved.
    fps : float
        The frames per second (FPS) for the output video.
    frame_size : tuple of int
        A tuple representing the width and height of the video frames in pixels.

    Returns
    -------
    video : cv2.VideoWriter
        The VideoWriter object for writing frames to the video file.

    Examples
    --------
    >>> video = create_video_file('output_video.mp4', 30.0, (1920, 1080))
    >>> for frame in frames:
    >>>     video.write(frame)
    >>> video.release()

    Notes
    -----
    - The `fps` parameter should be a positive float representing the desired frame rate for the output video.
    - The `frame_size` parameter should be a tuple of two integers representing the width and height of the video frames in pixels.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filename=output_path, 
                        fourcc=fourcc, 
                        fps=fps, 
                        frameSize=frame_size)
    return video

def tmp_video(orig_video: str, 
              output_video: str, 
              frame_rate: str):
    """
    Create a temporary video by sampling frames from an original video at a specified frame rate.

    Parameters
    ----------
    orig_video : str
        The path to the original video file.
    output_video : str
        The path where the output video file will be saved.
    frame_rate : int
        The interval of frames to sample. For example, a `frame_rate` of 2 will sample every second frame.

    Returns
    -------
    None

    Examples
    --------
    >>> tmp_video('input_video.mp4', 'output_video.mp4', 10)
    >>> # This will create an output video that includes every 10th frame from the input video.

    Notes
    -----
    - The input `orig_video` should be a valid path to an existing video file.
    - The `frame_rate` should be a positive integer representing the frame sampling interval.
    """
    vidcap, _, fps, frame_size = read_video_file(orig_video)
    video = create_video_file(output_video, fps, frame_size)
    success, img = vidcap.read()
    i=0
    while success:
        if i%frame_rate == 0:
            video.write(img)
        i+=1
        success, img = vidcap.read()
    # on prends la dernière frame
    if (i-1)%frame_rate != 0:
        video.write(img)
    del i
    del img
    del success
    vidcap.release()
    video.release()
    gc.collect()

def blur_video(model, 
               video_path: str, 
               output_path: str,
               frame_verif_rate: int=5, 
               black_box: bool=False,
               verbose: bool=False):
    """
    Apply blur or black boxes to specified objects in a video based on a detection model.

    Parameters
    ----------
    model : object
        The object detection model used to detect objects in the video frames.
    video_path : str
        The path to the input video file.
    output_path : str
        The path where the output video file with blurred or black boxed objects will be saved.
    frame_verif_rate : int, optional
        The interval of frames at which the object detection model is applied. Default is 5.
    black_box : bool, optional
        If True, black boxes are applied to detected objects instead of blurring. Default is False.
    verbose : bool, optional
        If True, prints progress and status messages. Default is False.

    Returns
    -------
    None

    Notes
    -----
    - The input `model` should be an object detection model capable of detecting objects in video frames.
    - The `video_path` parameter should be a valid path to an existing video file.
    - The `frame_verif_rate` determines how often the object detection model is applied to video frames.
    - If `black_box` is set to True, black boxes will be applied to detected objects instead of blurring them.
    """
    if verbose: print("Création vidéo temporaire")
    tmp_video_name = f"{uuid.uuid4().hex}.mp4"
    tmp_video(video_path, tmp_video_name, frame_verif_rate)
    gc.collect()

    if verbose: print("Floutage")
    vidcap_s, frame_number_s, _, _ = read_video_file(tmp_video_name)
    vidcap_o, _, fps_o, frame_size_o = read_video_file(video_path)
    video = create_video_file(output_path, fps_o, frame_size_o)
    success_s, img_s = vidcap_s.read()
    assert success_s, "Error while loading the temp video"

    for i in tqdm(range(frame_number_s)):
        results_s = model(img_s, verbose=False, stream=True)
        to_blur = len(next(results_s).boxes.conf) > 0
        del results_s ; gc.collect()
        # première frame
        if i == 0:
            for j in range(math.ceil(frame_verif_rate/2)):
                success_o, img_o = vidcap_o.read()
                if to_blur:
                    boxes = next(model(img_o, verbose=False, stream=True)).boxes
                    for box in boxes:
                        img_o = blur_box(img_o, box, black_box)
                video.write(img_o)
        # autre frame
        else:
            for j in range(frame_verif_rate):
                success_o, img_o = vidcap_o.read()
                if to_blur:
                    boxes = next(model(img_o, verbose=False, stream=True)).boxes
                    for box in boxes:
                        img_o = blur_box(img_o, box, black_box)
                video.write(img_o)
            # dernières frames
            if i+1 == frame_number_s:
                success_o, img_o = vidcap_o.read()
                while success_o:
                    video.write(img_o)
                    if to_blur:
                        boxes = next(model(img_o, verbose=False, stream=True)).boxes
                        for box in boxes:
                            img_o = blur_box(img_o, box, black_box)
                    success_o, img_o = vidcap_o.read()

        success_s, img_s= vidcap_s.read()

    vidcap_s.release()
    vidcap_o.release()
    video.release()
    if verbose: print("Suppression vidéo temporaire")
    os.remove(tmp_video_name)

    if verbose: print("========= FIN =========")

def bag_to_mp4(bag_file, output_file, topic="/camera/color/image_raw"):
    """
    Convert a ROS1 bag file containing image messages to an MP4 video file.

    Parameters
    ----------
    bag_file : str
        The path to the ROS1 bag file.
    output_file : str
        The path where the output MP4 video file will be saved.
    topic : str, optional
        The topic containing the image messages in the ROS bag file. Default is "/camera/color/image_raw".

    Returns
    -------
    None

    Examples
    --------
    >>> bag_to_mp4('example.bag', 'output_video.mp4', topic="/camera/depth/image_raw")

    Notes
    -----
    - The input `bag_file` should be a valid path to a ROS bag file.
    - The `topic` parameter specifies the ROS topic containing the image messages.
    - Ensure that the `bagpy` and `cv_bridge ` library is installed and imported before using this function.
    """
    bag_data = bg.bagreader(bag_file)
    bridge = CvBridge()
    images = bag_data.reader.read_messages(topic)

    fps = bag_data.topic_table[bag_data.topic_table["Topics"] == "/camera/color/image_raw"]["Frequency"].item()

    frame = next(images).message

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filename=output_file, 
                        fourcc=fourcc, 
                        fps=fps, 
                        frameSize=(frame.width, frame.height))
    
    while True:
        try:
            video.write(bridge.imgmsg_to_cv2(frame, desired_encoding="bgr8"))
            frame = next(images).message
        except StopIteration:
            break

    video.release()

if __name__ == "__main__":
    model = YOLO(args.yolo_model_path)
    bag_file = args.bag_file_path

    mp4_tmp_file_name = f"{uuid.uuid4().hex}.mp4"
    bag_to_mp4(bag_file, mp4_tmp_file_name)
    
    blur_video(model, mp4_tmp_file_name, 
               "output.mp4" if isinstance(args.output_file, type(None)) else args.output_file, 
               frame_verif_rate=5 if isinstance(args.frame_rate, type(None)) else args.frame_rate, 
               black_box=args.black_box, 
               verbose=args.verbose)
    if not args.keep_orig_mp4:
        os.remove(mp4_tmp_file_name)
    