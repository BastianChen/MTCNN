import torch.nn as nn
import torch.nn.functional as F
import torch


class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1, 1),
            nn.BatchNorm2d(10),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(10, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.offset = nn.Conv2d(32, 4, 1, 1)
        self.confidence = nn.Conv2d(32, 1, 1, 1)
        self.landmarks = nn.Conv2d(32, 10, 1, 1)

    def forward(self, data):
        y1 = self.layer1(data)
        offset = self.offset(y1)
        confidence = F.sigmoid(self.confidence(y1))
        landmarks = self.landmarks(y1)
        return confidence, offset, landmarks


class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 28, 3, 1, 1),
            nn.BatchNorm2d(28),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(28, 48, 3, 1),
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(48, 64, 2, 1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(3 * 3 * 64, 128),
            nn.BatchNorm1d(128),
            nn.PReLU()
        )

        self.offset = nn.Linear(128, 4)
        self.confidence = nn.Linear(128, 1)
        self.landmarks = nn.Linear(128, 10)

    def forward(self, data):
        y1 = self.layer1(data)
        y1 = y1.reshape(data.size(0), -1)
        y2 = self.layer2(y1)
        offset = self.offset(y2)
        confidence = F.sigmoid(self.confidence(y2))
        landmarks = self.landmarks(y2)
        return confidence, offset, landmarks


class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),  # 23*23*32
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),  # 10*10*64
            nn.Conv2d(64, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 4*4*64
            nn.Conv2d(64, 128, 2, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(3 * 3 * 128, 256),
            nn.BatchNorm1d(256),
            nn.PReLU()
        )

        self.offset = nn.Linear(256, 4)
        self.confidence = nn.Linear(256, 1)
        self.landmarks = nn.Linear(256, 10)

    def forward(self, data):
        y1 = self.layer1(data)
        y1 = y1.reshape(data.size(0), -1)
        y2 = self.layer2(y1)
        offset = self.offset(y2)
        confidence = F.sigmoid(self.confidence(y2))
        landmarks = self.landmarks(y2)
        return confidence, offset, landmarks


# class ONet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 32, 3, 1, 1),  # 48*48*32
#             nn.BatchNorm2d(32),
#             nn.PReLU(),
#             nn.Conv2d(32, 64, 3, 1),  # 46*46*64
#             nn.BatchNorm2d(64),
#             nn.PReLU(),
#             nn.Conv2d(64, 128, 3, 1),  # 44*44*128
#             nn.BatchNorm2d(128),
#             nn.PReLU(),
#             nn.Conv2d(128, 256, 3, 1),  # 42*42*256
#             nn.BatchNorm2d(256),
#             nn.PReLU(),
#         )
# 
#         # 加入残差块
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(256, 128, 1, 1),  # 42*42*128
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.Conv2d(128, 256, 1, 1),
#             nn.BatchNorm2d(256),
#             nn.PReLU(),
#             nn.MaxPool2d(3, 2),  # 20*20*256
#             nn.Conv2d(256, 128, 1, 1),  # 20*20*128
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.Conv2d(128, 256, 1, 1),
#             nn.BatchNorm2d(256),
#             nn.PReLU(),
#             nn.MaxPool2d(3, 2),  # 9*9*256
#         )
# 
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(256, 512, 3, 1),  # 7*7*512
#             nn.BatchNorm2d(512),
#             nn.PReLU(),
#             nn.Conv2d(512, 256, 3, 1),  # 5*5*256
#             nn.BatchNorm2d(256),
#             nn.PReLU(),
#             nn.Conv2d(256, 128, 3, 1),  # 3*3*256
#             nn.BatchNorm2d(128),
#             nn.PReLU(),
#         )
# 
#         self.layer4 = nn.Sequential(
#             nn.Linear(3 * 3 * 128, 256),
#             nn.BatchNorm1d(256),
#             nn.PReLU()
#         )
# 
#         self.offset = nn.Linear(256, 4)
#         self.confidence = nn.Linear(256, 1)
#         self.landmarks = nn.Linear(256, 10)
# 
#     def forward(self, data):
#         y1 = self.layer1(data)
#         y2 = self.layer2(y1)
#         y3 = self.layer3(y2)
#         y3 = y3.reshape(data.size(0), -1)
#         y4 = self.layer4(y3)
#         offset = self.offset(y4)
#         confidence = F.sigmoid(self.confidence(y4))
#         landmarks = self.landmarks(y4)
#         return confidence, offset, landmarks


if __name__ == '__main__':
    a = torch.Tensor([1, 2, 3, 4, 5, 6]).reshape(2, 3)
    print(a)
