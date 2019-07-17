import cv2
import config
import os
import sys

# get fps as script args
fps = int(sys.argv[1])
# max number of data folder to read
max_folders = int(sys.argv[2])

# skip first n folders
try:
    skip_first = int(sys.argv[3])
except:
    print("Using default value for skip_first")
    skip_first = 0
# labels = [json.loads(line) for d_label in config.ts_flabels for line in open(d_label, 'r')]
# # create map for efficient access to annotations
# labels = {label['raw_file']: label for label in labels}

# get video frames of a day of shooting
test_set_dir = config.ts_root + config.ts_subdirs[0]
frames = []
for nfolder, clipfolder in enumerate(sorted(os.listdir(test_set_dir))):
    if clipfolder.startswith('.') or nfolder < skip_first:
        continue

    subfolder = os.path.join(test_set_dir, clipfolder)
    for imgname in range(1, 21):
        frames.append(cv2.imread(os.path.join(subfolder, str(imgname) + ".jpg")))
        # print(os.path.join(subfolder, str(imgname) + ".jpg"))

    # decide when to finish
    if nfolder-skip_first == max_folders:
        break

# all frames are assumed to be of same size
height , width , layers = frames[0].shape
# todo save resized images directly

video = cv2.VideoWriter('../video2.mp4', -1, fps, (width, height))

for frame in frames:
    video.write(frame)

cv2.destroyAllWindows()
video.release()