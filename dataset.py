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
            # 数据增强
            # transforms.RandomRotation(5),
            # transforms.RandomVerticalFlip(0.5),  # 依据概率p对PIL图片进行垂直翻转 参数： p- 概率，默认值为0.5
            # transforms.RandomHorizontalFlip(0.5),  # 依据概率p对PIL图片进行水平翻转 参数： p- 概率，默认值为0.5
            # transforms.RandomRotation((-30, 30)),  # 依degrees随机旋转一定角度
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
        img_data = self.trans(Image.open(img_path))
        confidence = torch.Tensor([int(strs[1])])
        offset = torch.Tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])
        landmarks = torch.Tensor(
            [float(strs[6]), float(strs[7]), float(strs[8]), float(strs[9]), float(strs[10]), float(strs[11]),
             float(strs[12]), float(strs[13]), float(strs[14]), float(strs[15])])
        return img_data, confidence, offset, landmarks


if __name__ == '__main__':
    # data = datasets(r"F:\Photo_example\CelebA\sample\12")[0]
    data = datasets(r"C:\sample\12")[0]
    print(data)
