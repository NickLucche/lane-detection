from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import json
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.cuda_device import device
import utils.config as config
import torchvision.transforms.functional as F
import torch


class TUSimpleDataset(Dataset):
    """TUSimpleBenchmark dataset loader helper class. Filenames are kept in the
        __init__ method while actual image reading is delegated to __getitem__
        method. For more info on data format see
        https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection.
    """

    def __init__(self, root_dir:str, sub_dirs:list, data_labels:list, n_frames:int=5, shuffle=True, shuffle_seed=7):
        """
        :param root_dir: Root dir of the dataset.
        :param sub_dirs: Root dir will contain some number of sub_dirs, into which actual
                        data is kept; each folder will contain 20 frames of a short video,
                        with the last frame being the annotated target.
        :param data_labels: List of filenames containing annotations for data in a specific subfolder.
        :param n_frames: How many frames will compose an actual sample (length of single timeseries);
                        frames are counted backwards from target (20th frame).
        :param shuffle: Whether to shuffle samples.
        :param shuffle_seed: seed to use when shuffling samples.
        """

        self.root_dir = root_dir
        self.sub_dirs = sub_dirs
        assert n_frames>=1 and n_frames<20
        self.n_frames = n_frames
        # data labels are read just once here and stored
        labels = [json.loads(line) for d_label in data_labels for line in open(d_label, 'r')]
        # create map for efficient access to annotations, at the cost of some memory
        self.labels = {self.get_clip_id(label['raw_file']): label for label in labels}

        # load sample 'filenames' (actually just folder containing multiple frames)
        self.sample_folders = []
        for sub_folder in sub_dirs:
            for sample_folder in os.listdir(os.path.join(root_dir, sub_folder)):
                if not sample_folder.startswith('.'):
                    self.sample_folders.append(os.path.join(sub_folder, sample_folder))

        assert len(self.sample_folders) == len(self.labels)
        if shuffle:
            random.seed(shuffle_seed)
            # sample folder maintains the structure in its name
            # hence we can retrieve the labels from it
            random.shuffle(self.sample_folders)

    def __len__(self):
        return len(self.sample_folders)

    def __getitem__(self, idx):
        """
        Load N frames into memory and generate target sample
        from stored labels file.
        Both frames and target are currently rescaled to fixed
        size of 256x128.
        Tensor transformation is also executed by default.
        :param idx:
        :return: tuple containing list of rescaled and
            normalized frames and target.
        """
        images_dir = os.path.join(self.root_dir, self.sample_folders[idx])

        # counting samples from target frame
        frames = []
        for img_name in range(1, 21)[-self.n_frames:]:
            img_name = str(img_name) + ".jpg"
            # print("Loading", os.path.join(images_dir, img_name))
            frames.append(cv2.imread(os.path.join(images_dir, img_name)))

        label_key = self.sample_folders[idx]
        label = self.labels[label_key]

        assert label_key == label['raw_file'].replace('clips/', '').replace('/20.jpg', '')
        # print("Subfolder key:",label_key)
        # print("Full folder:", images_dir)
        # print("Target:", label['raw_file'])
        lanes = label['lanes']
        height = label['h_samples']
        target_ = self.get_scaled_target(lanes, height, frames[0].shape)  # all samples are assumed to be of same shape

        # apply transformations
        target_ = self.to_tensor(target_)
        frames = [self.to_tensor(self.resize(f)/255.) for f in frames] # also normalize

        return frames, target_

    def get_scaled_target(self, lanes:list, heights:list, original_shape:tuple, new_dim=(128, 256, 1)):
        """
        Generate normalized and rescaled target image;
        rescaling is performed directly on label values
        contained in file for efficiency.
        :return: numpy image target of shape 'new_dim'.
        """
        gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, heights) if x >= 0] for lane in lanes]
        # target is a probability map
        image = np.zeros(new_dim).astype(np.float32)
        c = (1., 1., 1.)   # white lane points (normalized)

        scale_factorY = new_dim[0] * (1./original_shape[0])
        scale_factorX = new_dim[1] * (1./original_shape[1])
        for lane in gt_lanes_vis:
            # rescale lane points
            lane = [(x*scale_factorX, y*scale_factorY) for (x, y) in lane]
            cv2.polylines(image, np.int32([lane]), isClosed=False, color=c, thickness=3)   # original value is 5 for full size image
        # print(image.shape, image.max(), image.astype(np.uint8).mean(), image.sum())

        return image.astype(np.float32)

    def get_clip_id(self, filename:str):
        # clip unique id is made of dataset subfolder+clip_folder
        if filename.endswith('.jpg'):
            # raw_file field was passed
            chunks = filename.split('/')
            return '/'.join([chunks[1], chunks[2]])
        else:
            # sample absolute path was passed
            pass

    def resize(self, image, new_dim=(256, 128)):
        # transform method for resizing a numpy image
        return cv2.resize(image, new_dim).astype(np.float32)


    def to_tensor(self, image):
        # swap color axis because
        # numpy image: H x W x C, while
        # torch image: C X H X W
        return torch.from_numpy(image.transpose((2, 0, 1)))


def show_plain_images(images, n_frames, save=False, fname=None):
    """
    Utility method for plotting a series of images on a single row
    using matplotlib.
    Resulting plot can be optionally saved by passing 'save' and
    'fname'.
    """
    plt.figure(num=None, figsize=(20, 4), dpi=80)
    for i in range(n_frames):
        ax = plt.subplot(1, n_frames, i + 1)
        if images[i].shape[0] > 1:
            # print(images[i].size())
            image = images[i].permute(1, 2, 0).cpu().numpy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
        else:
            ax.imshow(images[i].squeeze().cpu().numpy(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if save and fname:
        plt.savefig(os.path.join(config.data_dir, fname), dpi=200)
    else:
        plt.show()
    # plt.close()



def show_annotated_image(gt_lanes:list, gt_hsamples:list, gt_frame_filename:str=None, resize=True, save=False, filename=''):
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
    if resize:
        imageb = np.zeros((128, 256, 3))
    else:
        imageb = np.zeros((720, 1280, 3))

    if gt_frame_filename:
        image = cv2.imread(gt_frame_filename)
        if resize:
            image = cv2.resize(image, (256, 128))

    c = (255, 0, 0)


    scale_factorX = 128 * (1. / 720)
    scale_factorY = 256 * (1. / 1280)
    for lane in gt_lanes_vis:
        if resize:
            lane = [(x * scale_factorY, y * scale_factorX) for (x, y) in lane]
        cv2.polylines(imageb, np.int32([lane]), isClosed=False, color=c, thickness=5)
        # print(imageb.shape)
    if resize:
        # imageb = cv2.resize(imageb, (256, 128))
        plt.imshow(imageb.astype(np.int32))
        plt.show()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # uint8 addition uses modulo (230+30=5)
    imageb = imageb + image
    plt.imshow(imageb.astype(np.int32))
    if save:
        plt.savefig(filename, dpi=200)
    plt.show()
    # plt.imshow(image, cmap='gray')
    # plt.show()
    return imageb


# original values: 3626 train samples - 2782 test samples
if __name__ == '__main__':
    # ** Functions testing (might not be updated to reflect latest change in methods) **

    labels = [json.loads(line) for line in open('/Users/nick/Desktop/train_set/label_data_0531.json', 'r')]
    sample_no = 299
    for sample_no in range(0, 100, 10):
        lanes = labels[sample_no]['lanes']
        height = labels[sample_no]['h_samples']
        frame = os.path.join('/Users/nick/Desktop/train_set', labels[sample_no]['raw_file'])
        show_annotated_image(lanes, height, frame, resize=False, save=True, filename=f'image_{sample_no}.png')
    root = '/Users/nick/Desktop/train_set/clips/'
    subdirs = ['0601', '0531', '0313-1', '0313-2']
    flabels = ['/Users/nick/Desktop/train_set/label_data_0601.json',
               '/Users/nick/Desktop/train_set/label_data_0531.json',
               '/Users/nick/Desktop/train_set/label_data_0313.json']

    data_transform = transforms.Compose([
        transforms.Resize((128, 256)),
        # transforms.RandomHorizontalFlip(), #need to apply flip to all samples and target too
        transforms.ToTensor(),
    ])
    tu_dataset = TUSimpleDataset(root, subdirs, flabels, transforms=data_transform, shuffle_seed=9)

    # build data loader
    tu_dataloader = DataLoader(tu_dataset, batch_size=2, shuffle=True, num_workers=2)
    # ([input_tensor_1, input_tensor_2], label_tensor) -> (
    # [batched_input_tensor_1, batched_input_tensor_2], batched_label_tensor)
    tu_test_dataset = TUSimpleDataset(root.replace('train_set', 'test_set'), ['0601', '0531', '0530'], ['/Users/nick/Desktop/test_set/test_labels.json'], transforms=data_transform)

    # for i, (frames, target) in enumerate(tu_test_dataset):
    #     print("test set length:", len(tu_dataset))
    #     print("Sizes:", frames[0].size(), target.size())
    #     # print("N-samples", len(samples))
    #
    #     # resize to original image size (needed for TuSimple evaluation)
    #     target = transforms.Resize((720, 1280))(F.to_pil_image(target))
    #     target = transforms.ToTensor()(target)
    #     show_plain_images(frames + [target], len(frames) + 1)


    for i, (batched_samples, batched_target) in enumerate(tu_dataloader):
        # print(len(batched_samples))
        print("TARGET:", torch.max(batched_target), torch.sum(batched_target))
        # print(batched_samples[0].size(), batched_target.size())
        for i in range(2):
            samples = []
            for j in range(len(batched_samples)):
                a, b = torch.chunk(batched_samples[j], 2, dim=0)
                if i==0:
                    samples.append(a.squeeze())
                else:
                    samples.append(b.squeeze())
                print("samples info")
                print(a.squeeze().size())
                print(torch.max(a).item(), torch.mean(a).item(), torch.sum(a).item())

            print("Single target shape:", batched_target[i].size())
            show_plain_images(samples + [batched_target[i]], len(samples) + 1)
        # here result is a tensor, with channel first format
        # samples, target = tu_dataset[i]
        # print("Sizes:", samples[0].size(), target.size())
        # print("N-samples", len(samples))
