# from torch.utils.data import DataLoader
# import torch
# import torch.nn as nn
# import os
# from dataset import datasets
#
#
# class Trainer:
#     def __init__(self, net, dataset_path, save_path):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.net = net.to(self.device)
#         self.save_path = save_path
#         self.data = DataLoader(datasets(dataset_path), batch_size=100, drop_last=True, num_workers=2)
#
#         self.confidence_loss = nn.BCELoss()
#         self.offset_loss = nn.MSELoss()
#         self.optimizer = torch.optim.Adam(self.net.parameters())
#
#         if os.path.exists(self.save_path):
#             self.net.load_state_dict(torch.load(self.save_path))
#
#     def train(self):
#         while True:
#             for i, (img_data, confidence, offset) in enumerate(self.data):
#                 img_data = img_data.to(self.device)
#                 confidence = confidence.to(self.device)
#                 offset = offset.to(self.device)
#                 confidence_out, offset_out = self.net(img_data)
#                 confidence_out = confidence_out.reshape(-1, 1)
#                 offset_out = offset_out.reshape(-1, 4)
#
#                 confidence_mask = torch.lt(confidence, 2)
#                 confidence_out = torch.masked_select(confidence_out, confidence_mask)
#                 confidence = torch.masked_select(confidence, confidence_mask)
#                 loss_confidence = self.confidence_loss(confidence_out, confidence)
#
#                 offset_mask = torch.gt(confidence, 0)
#                 offset_out = torch.masked_select(offset_out, offset_mask)
#                 offset = torch.masked_select(offset, offset_mask)
#                 loss_offset = self.offset_loss(offset, offset_out)
#
#                 loss_total = loss_confidence + loss_offset
#
#                 self.optimizer.zero_grad()
#                 loss_total.backward()
#                 self.optimizer.step()

import torch
import numpy as np
import utils
import nets
from torchvision import transforms
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import os


class Detector:
    def __init__(self, pnet_path, rnet_path, onet_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pnet = nets.PNet().to(self.device)
        self.rnet = nets.RNet().to(self.device)
        self.onet = nets.ONet().to(self.device)

        self.pnet.load_state_dict(torch.load(pnet_path))
        self.rnet.load_state_dict(torch.load(rnet_path))
        self.onet.load_state_dict(torch.load(onet_path))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def detect(self):
        pass

    def pnet_detect(self):
        pass

    def rnet_detect(self):
        pass

    def onet_detect(self):
        pass

    def backToImgae(self):
        pass


if __name__ == '__main__':
    box_path = r"text/list_bbox_celeba.txt"
    landmarks_path = r"text/list_landmarks_celeba.txt"
    # img_path = r"TESTIMG/TESTIMG/000058.jpg"
    img_path = r"TESTIMG/TESTIMG"
    box_array = open(box_path, "r").readlines()
    landmarks_array = open(landmarks_path, "r").readlines()
    for i, (box, landmarks) in enumerate(zip(box_array, landmarks_array)):
        if i < 2:
            continue
        box = box.strip().split()
        landmarks = landmarks.strip().split()
        if 10 <= i <= 70:
            w_box, h_box = int(box[3]), int(box[4])
            w_landmarks, h_landmarks = int(landmarks[9]) - int(landmarks[1]), int(landmarks[10]) - int(
                landmarks[2])
            center_x, center_y = int(int(landmarks[1]) + 0.5 * w_landmarks), int(
                int(landmarks[2]) + 0.5 * h_landmarks)

            w_average, h_average = int((w_box + w_landmarks) / 2), int((h_box + h_landmarks) / 2)

            img = Image.open(os.path.join(img_path, box[0]))
            img_draw = ImageDraw.ImageDraw(img)
            img_draw.rectangle(
                (int(center_x - 0.6 * w_average), int(center_y - 0.72 * h_average), int(center_x + 0.65 * w_average),
                 int(center_y + 0.58 * h_average)), outline="green", width=3)
            img_draw.point((int(landmarks[1]), int(landmarks[2])), fill="red")
            img_draw.point((int(landmarks[3]), int(landmarks[4])), fill="red")
            img_draw.point((int(landmarks[5]), int(landmarks[6])), fill="red")
            img_draw.point((int(landmarks[7]), int(landmarks[8])), fill="red")
            img_draw.point((int(landmarks[9]), int(landmarks[10])), fill="red")
            img_draw.point((center_x, center_y), fill="blue")
            img.show()
