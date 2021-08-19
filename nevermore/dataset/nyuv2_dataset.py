"""NYUv2 Dateset Segmentation Dataloader"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

NYUv2_CLASSES = (
    'background',  # always index 0
    'bed',
    'books',
    'ceiling',
    'chair',
    'floor',
    'furniture',
    'objects',
    'painting',
    'sofa',
    'table',
    'tv',
    'wall',
    'window'
)
NUM_CLASSES = len(NYUv2_CLASSES)


class NYUv2Dateset(Dataset):
    """NYUv2Dateset Dataset"""

    def __init__(
        self,
        list_file,
        img_dir,
        mask_dir,
        depth_dir,
        normal_dir,
        transform=None
    ):
        self.images = open(list_file, "rt").read().split("\n")[:-1]
        self.transform = transform

        self.img_extension = ".jpg"
        self.mask_extension = ".png"
        self.depth_extension = ".png"
        self.normal_extension = ".png"

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir
        self.depth_root_dir = depth_dir
        self.normal_root_dir = normal_dir

        self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image_path = os.path.join(
            self.image_root_dir, name + self.img_extension
        )
        mask_path = os.path.join(
            self.mask_root_dir, name + self.mask_extension
        )
        depth_path = os.path.join(
            self.depth_root_dir, name + self.depth_extension
        )
        normal_path = os.path.join(
            self.normal_root_dir, name + self.normal_extension
        )

        image = self.load_image(path=image_path)
        gt_mask = self.load_mask(path=mask_path)
        gt_depth = self.load_depth(path=depth_path)
        gt_normal = self.load_normal(path=normal_path)
        data = {
            'image': torch.FloatTensor(image),
            'mask': torch.LongTensor(gt_mask),
            'depth': torch.FloatTensor(gt_depth),
            'normal': torch.FloatTensor(gt_normal),
            'image_name': name
        }

        return data

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))

        for name in self.images:
            mask_path = os.path.join(
                self.mask_root_dir, name + self.mask_extension
            )

            raw_image = Image.open(mask_path).resize((224, 224), Image.NEAREST)
            imx_t = np.array(raw_image).reshape(224 * 224)
            imx_t[imx_t == 255] = len(NYUv2_CLASSES)

            for i in range(NUM_CLASSES):
                counts[i] += np.sum(imx_t == i)

        return counts

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values / np.sum(values)

        return torch.Tensor(p_values)

    def load_image(self, path=None):
        raw_image = Image.open(path)
        raw_image = np.transpose(raw_image.resize((224, 224)), (2, 1, 0))
        imx_t = np.array(raw_image, dtype=np.float32) / 255.0

        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize((224, 224), Image.NEAREST)
        imx_t = np.array(raw_image)
        # border
        imx_t[imx_t == 255] = len(NYUv2_CLASSES)

        return imx_t

    def load_depth(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize((224, 224), Image.NEAREST)
        imx_t = np.array(raw_image)

        return imx_t

    def load_normal(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize((224, 224), Image.NEAREST)
        imx_t = np.array(raw_image)

        return imx_t


if __name__ == "__main__":
    data_root = '/data/dixiao.wei/NYU'
    list_file_path = os.path.join(data_root, "train.txt")
    img_dir = os.path.join(data_root, "images", "train")
    mask_dir = os.path.join(data_root, "segmentation", "train")
    # depth_dir = os.path.join(data_root, "depths","train")

    objects_dataset = NYUv2Dateset(
        list_file=list_file_path, img_dir=img_dir, mask_dir=mask_dir
    )

    print(objects_dataset.get_class_probability())

    sample = objects_dataset[0]
    image, mask = sample['image'], sample['mask']

    image.transpose_(0, 2)

    fig = plt.figure()

    a = fig.add_subplot(1, 2, 1)
    plt.imshow(image)

    a = fig.add_subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()
