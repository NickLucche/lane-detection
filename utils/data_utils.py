from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
import torch
import os
import json
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt


class TUSimpleBenchmarkDataLoader:

    def __init__(self, root_dir:str=None, train_dirs:list=None, batch_size=8):
        # self.root_dir = root_dir if root_dir else self.root_dir = '~/Desktop/train_set/clips'
        # self.train_dirs = train_dirs if train_dirs else self.train_dirs = ['0601', '0531', '0313-1', '0313-2']

        # Resize all images to 256x128
        t = {train_dirs[i]: transforms.Compose([
                # Data augmentation
                # randomly flip it horizontally.
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]) for i in range(len(train_dirs))}

        data_transforms = {
            *t
        }

        image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(root_dir, x),
                transform=data_transforms[x]
            )
            for x in train_dirs
        }

        # one data loader for each subfolder in 'clips' (see https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection)
        #
        dataloaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x], batch_size=batch_size,
                shuffle=False, num_workers=4
            )
            for x in train_dirs
        }

        dataset_sizes = {x: len(image_datasets[x]) for x in train_dirs}



class TUSimpleBenchmarkDataset(Dataset):
    """TUSimpleBenchmark dataset loader helper class. Filenames are kept in the
        __init__ method while actual image reading is delegated to __getitem__
        method. For more info see https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection.
    """

    def __init__(self, root_dir:str, sub_dirs:list, data_labels:list, n_frames:int=5, shuffle=True, transforms=None, shuffle_seed=7):
        """

        :param root_dir: Root dir of the dataset
        :param sub_dirs: Root dir will contain some number of sub_dirs, into which actual
                        data is kept; each folder will contain 20 frames of a short video,
                        with the last frame being the annotated target.
        :param data_labels: List of filenames containing annotations for data in a specific subfolder.
        :param n_frames: How many frames will compose an actual sample (length of single timeseries);
                        frames are counted backwards from target (20th frame).
        :param shuffle: Whether to shuffle samples.
        :param transforms: Optional transform to be applied on a sample.
        :param shuffle_seed
        """

        self.root_dir = root_dir
        self.sub_dirs = sub_dirs
        assert n_frames>=1 and n_frames<20
        self.n_frames = n_frames

        # data labels are read just once here and stored
        print("Loading data labels")
        self.labels = [json.loads(line) for d_label in data_labels for line in open(d_label, 'r')]

        # self.labels[d_label[-9:-5]] = json.load(f)

        # also pre-compute length of dataset
        # self.len = 0
        # for l in self.labels:
        #     self.len += len(l)

        # load sample 'filenames' (actually just folder containing multiple frames)
        self.sample_folders = []
        for sub_folder in sub_dirs:
            for sample_folder in os.listdir(os.path.join(root_dir, sub_folder)):
                # add special char for recognizing target-samples
                # self.sample_folders.append('~'.join([sub_folder, sample_folder]))
                self.sample_folders.append(os.path.join(sub_folder, sample_folder))

        if shuffle:
            random.seed(shuffle_seed)
            # sample folder maintains the structure in its name
            # hence we can retrieve the labels from it
            random.shuffle(self.sample_folders)
            random.shuffle(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # split sample_folders to get label and images dir
        # label_file, sample_folder = self.sample_folders[idx].split('~')[0], self.sample_folders[idx].split('~')[1]

        images_dir = os.path.join(self.root_dir, self.sample_folders[idx])

        # counting samples from target frame
        frames = []
        for img_name in os.listdir(images_dir)[-self.n_frames:]:
            frames.append(io.imread(img_name))

        # special case
        # if label_file.startsWith('0313'):
        #     label_file = '0313'

        target_d = self.labels[label_file]
        # target = get_annotated_image(target_d['lanes'] )

        if self.transform:
            for sample in frames:
                sample = self.transform(sample)

        return frames


    def get_label_dict(self, label_file:str, sample_folder_name:str):
        pass



# :param eval_seed: Evaluation set is chosen among the many samples by means of a randomly init bit-mask,
#                           so that no bias comes from data ordering.


def show_plain_image():
    pass


def show_annotated_image(gt_lanes:list, gt_hsamples:list, gt_frame_filename:str=None):
    """
    Utility method for showing an annotated frame (target).
    For information on the annotation structure please refer to
    https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection.
    :param gt_lanes: list of lists, each containing width values of i-th lane.
    :param gt_frame_filename: frame image filepath (20th frame contained in a sample folder).
    :param gt_hsamples: list of height values corresponding to the lanes.
    :return:
    """
    gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, gt_hsamples) if x >= 0] for lane in gt_lanes]
    if gt_frame_filename:
        image = io.imread(gt_frame_filename)
        c = (0, 255, 0)
    else:
        c = (255, 255, 255)
        image = np.zeros((720, 1280))

    for lane in gt_lanes_vis:
        cv2.polylines(image, np.int32([lane]), isClosed=False, color=c, thickness=5)

    plt.imshow(image, cmap='gray')
    plt.show()
    return image

if __name__ == '__main__':
    # with open('/Users/nick/Desktop/train_set/label_data_0313.json', 'r') as f:
    #     labels = json.load(f)
    labels = [json.loads(line) for line in open('/Users/nick/Desktop/train_set/label_data_0313.json', 'r')]
    lanes = labels[0]['lanes']
    height = labels[0]['h_samples']
    frame = os.path.join('/Users/nick/Desktop/train_set', labels[0]['raw_file'])
    show_annotated_image(lanes, height)