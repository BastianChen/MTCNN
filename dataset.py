from torch.utils.data import Dataset
import torch
import numpy as np
import PIL.Image as Image
import os
from torchvision import transforms


class datasets(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.dataset.extend(open(os.path.join(path, "positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split(" ")
        img_path = os.path.join(self.path, strs[0])
        # img_path = os.path.join(self.path + "/image", strs[0])
        img_data = self.trans(np.array(Image.open(img_path)))
        # confidence = torch.Tensor(np.array([int(strs[1])]))
        # offset = torch.Tensor(np.array([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])]))
        confidence = torch.Tensor(np.array([int(strs[5])]))
        offset = torch.Tensor(np.array([float(strs[1]), float(strs[2]), float(strs[3]), float(strs[4])]))
        return img_data, confidence, offset


if __name__ == '__main__':
    data = datasets(r"F:\Photo_example\CelebA\sample\12")[0]
    print(data)
