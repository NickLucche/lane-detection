import argparse
import time

import cv2 as cv
import numpy as np
from hough_transform import *
import torch
import utils.train_utils as tu
from segnet_conv_lstm_model import SegnetConvLSTM
from utils.config import Configs
from utils.cuda_device import device
import matplotlib.pyplot as plt

"""
    This file was used to generate a lane-marked video sample
    to assess visual quality of model output.
    Input file is specified at line 26 while output video
    filename is passed as argument to the script.
    Frames are read from input video using a 'sliding window'
    of size 5 (but any size can be tried), therefore getting a
    prediction for every frame after the first 4.
    Frames are written to video output stream 'live' as they 
    get computed from model.
    
    Note that no particular efficiency method is taken into account
    here, such as storing previous frames feature maps.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--full-video-size", default=False, type=bool)
parser.add_argument("--filename", default='my-video-marked.mp4', type=str)
parser.add_argument("-m", '--model-path', required=True, type=str, help='Pre-trained model filepath')

args = parser.parse_args()

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("../video2.mp4")
cc = Configs()
model = SegnetConvLSTM(cc.hidden_dims, decoder_out_channels=2, lstm_nlayers=len(cc.hidden_dims),
                       vgg_decoder_config=cc.decoder_config)
print("Loading model..")
tu.load_model_checkpoint(model, args.model_path, inference=False,
                         map_location=device)

# frames to feed to the model
inputs = []
model.train()
if args.full_video_size:
    print("Using images at original scale")
    zeros = np.zeros((720, 1280)).astype(np.uint8)
else:
    zeros = np.zeros((128, 256)).astype(np.uint8)

# get a 5 frame 'sliding' window
for i in range(5):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    # reshape frame to model input size
    f = cv.resize(frame, (256, 128)).astype(np.float32).transpose(2, 0, 1)
    f = (torch.from_numpy(f) / 255.).unsqueeze(0)
    inputs.append(f)

# open video for writing lane marked frames
if args.full_video_size:
    video = cv.VideoWriter(args.filename, -1, 15, (1280, 720))
else:
    video = cv.VideoWriter(args.filename, -1, 15, (256, 128))

while cap.isOpened():
    with torch.no_grad():
        # print("Input len:", len(inputs), [(ff.size(), ff.max().item()) for ff in inputs])
        marks = model(inputs)
        # print(marks.max().item(), marks.min().item(), marks.sum().item())
        marks = (marks > 0.).long()

        # overlay line markings to original image
        marks = marks[:, 1, :, :].permute(1, 2, 0).numpy().reshape(128, 256)
        # get red lane markings (multiply then stack on red channel)
        marks = marks*255

        if args.full_video_size:
            # scale prediction back to 720p
            marks = cv.resize(marks, (1280, 720), interpolation=cv.INTER_NEAREST)
        else:
            frame = cv.resize(frame, (256, 128)).astype(np.uint8)
        # plt.imshow(cv.resize(marks.reshape(128, 256), (1280, 720), interpolation=cv.INTER_NEAREST), cmap='gray')
        # plt.show()
        # repeat image to have rgb representation
        # marks = (np.repeat(marks, 3, axis=2)).astype(np.uint8)


        marks = np.stack([zeros, zeros, marks], axis=2).astype(np.uint8)

        print("Overlay shape:", marks.shape, "Frame shape:", frame.shape)
        output = cv.addWeighted(frame, 1, marks, 1, 0)
        # plt.imshow(output)
        # plt.show()

        # cv.imshow("output", output)
        video.write(output)
        # output_frames.append(output)

        # update last frame
        ret, frame = cap.read()
        if ret:
            # slide all frames and ditch the first one
            inputs = [inputs[j] for j in range(1, len(inputs))]
            f = cv.resize(frame, (256, 128)).astype(np.float32).transpose(2, 0, 1)
            f = (torch.from_numpy(f) / 255.).unsqueeze(0)
            # add latest frame to sliding window
            inputs.append(f)

        # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

video.release()
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()