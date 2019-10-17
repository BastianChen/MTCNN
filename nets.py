import torch.nn as nn
import torch.nn.functional as F
import torch


# 卷积层
class ConvolutionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding=0, groups=1, bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(output_channels),
            nn.PReLU()
        )

    def forward(self, data):
        return self.layer(data)


# 下采样层
class DownSamplingLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.layer = ConvolutionLayer(input_channels, output_channels, kernel_size, stride, padding)

    def forward(self, data):
        return self.layer(data)


# 残差层
class ResidualLayer(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.layer = nn.Sequential(
            ConvolutionLayer(input_channels, input_channels // 2, 1, 1, 0, 1, False),
            ConvolutionLayer(input_channels // 2, input_channels // 2, 3, 1, 1, 1, False),
            ConvolutionLayer(input_channels // 2, input_channels, 1, 1, 0, 1, False)
        )

    def forward(self, data):
        return data + self.layer(data)


# 深度可分离卷积层
class DepthwiseLyer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super().__init__()
        self.layer = nn.Sequential(
            ConvolutionLayer(input_channels, input_channels, kernel_size, 1, 0, input_channels),
            ConvolutionLayer(input_channels, output_channels, 1, 1, 0)
        )

    def forward(self, data):
        return self.layer(data)


class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            ConvolutionLayer(3, 10, 3, 1, 1),
            ConvolutionLayer(10, 10, 3, 2),
            DepthwiseLyer(10, 16, 3),
            DepthwiseLyer(16, 32, 3)
        )
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(3, 10, 3, 1, 1),  # 12*12*10
        #     nn.BatchNorm2d(10),
        #     nn.PReLU(),
        #     # nn.MaxPool2d(3, 2),
        #     nn.Conv2d(10, 10, 3, 2),  # 5*5*10
        #     nn.BatchNorm2d(10),
        #     nn.PReLU(),
        #     # 使用深度可分离卷积
        #     nn.Conv2d(10, 10, 3, 1, groups=10),  # 3*3*10
        #     nn.BatchNorm2d(10),
        #     nn.PReLU(),
        #     nn.Conv2d(10, 16, 1, 1),  # 3*3*16
        #     nn.BatchNorm2d(16),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, 3, 1, groups=16),  # 1*1*16
        #     nn.BatchNorm2d(16),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 32, 1, 1),  # 1*1*32
        #     nn.BatchNorm2d(32),
        #     nn.PReLU()
        # )
        self.offset = nn.Conv2d(32, 4, 1, 1)
        self.confidence = nn.Conv2d(32, 1, 1, 1)
        # self.landmarks = nn.Conv2d(32, 10, 1, 1)

    def forward(self, data):
        y1 = self.layer1(data)
        offset = self.offset(y1)
        confidence = F.sigmoid(self.confidence(y1))
        # landmarks = self.landmarks(y1)
        # return confidence, offset, landmarks
        return confidence, offset


#
# class RNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 28, 3, 1, 1),
#             nn.BatchNorm2d(28),
#             nn.PReLU(),
#             nn.MaxPool2d(3, 2),
#             nn.Conv2d(28, 48, 3, 1),
#             nn.BatchNorm2d(48),
#             nn.PReLU(),
#             nn.MaxPool2d(3, 2),
#             nn.Conv2d(48, 64, 2, 1),
#             nn.BatchNorm2d(64),
#             nn.PReLU()
#         )
#
#         self.layer2 = nn.Sequential(
#             nn.Linear(3 * 3 * 64, 128),
#             nn.BatchNorm1d(128),
#             nn.PReLU()
#         )
#
#         self.offset = nn.Linear(128, 4)
#         self.confidence = nn.Linear(128, 1)
#         self.landmarks = nn.Linear(128, 10)
#
#     def forward(self, data):
#         y1 = self.layer1(data)
#         y1 = y1.reshape(data.size(0), -1)
#         y2 = self.layer2(y1)
#         offset = self.offset(y2)
#         confidence = F.sigmoid(self.confidence(y2))
#         landmarks = self.landmarks(y2)
#         return confidence, offset, landmarks
#
#
# class ONet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 32, 3, 1, 1),
#             nn.BatchNorm2d(32),
#             nn.PReLU(),
#             nn.MaxPool2d(3, 2),  # 23*23*32
#             nn.Conv2d(32, 64, 3, 1),
#             nn.BatchNorm2d(64),
#             nn.PReLU(),
#             nn.MaxPool2d(3, 2),  # 10*10*64
#             nn.Conv2d(64, 64, 3, 1),
#             nn.BatchNorm2d(64),
#             nn.PReLU(),
#             nn.MaxPool2d(2, 2),  # 4*4*64
#             nn.Conv2d(64, 128, 2, 1),
#             nn.BatchNorm2d(128),
#             nn.PReLU(),
#         )
#
#         self.layer2 = nn.Sequential(
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
#         y1 = y1.reshape(data.size(0), -1)
#         y2 = self.layer2(y1)
#         offset = self.offset(y2)
#         confidence = F.sigmoid(self.confidence(y2))
#         landmarks = self.landmarks(y2)
#         return confidence, offset, landmarks


class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            ConvolutionLayer(3, 14, 3, 1, 1),  # 24*24*14
            ResidualLayer(14),
            ResidualLayer(14),
            ResidualLayer(14),
            ResidualLayer(14),
            DownSamplingLayer(14, 28, 3, 2),  # 11*11*28
            DepthwiseLyer(28, 14, 3),  # 9*9*28
            ResidualLayer(14),
            ResidualLayer(14),
            DownSamplingLayer(14, 48, 3, 2),  # 4*4*48
            DepthwiseLyer(48, 64, 2)  # 3*3*64
        )
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(3, 28, 3, 1, 1),  # 24*24*28
        #     nn.BatchNorm2d(28),
        #     nn.PReLU(),
        #     # nn.MaxPool2d(3, 2),
        #     nn.Conv2d(28, 28, 3, 2),  # 11*11*28
        #     nn.BatchNorm2d(28),
        #     nn.PReLU(),
        #     nn.Conv2d(28, 48, 3, 1),  # 9*9*28
        #     nn.BatchNorm2d(48),
        #     nn.PReLU(),
        #     # nn.MaxPool2d(3, 2),
        #     nn.Conv2d(48, 48, 3, 2),  # 4*4*28
        #     nn.BatchNorm2d(48),
        #     nn.PReLU(),
        #     nn.Conv2d(48, 64, 2, 1),  # 3*3*28
        #     nn.BatchNorm2d(64),
        #     nn.PReLU()
        # )

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
            ConvolutionLayer(3, 16, 3, 1, 1),  # 48*48*16
            ResidualLayer(16),
            ResidualLayer(16),
            ResidualLayer(16),
            ResidualLayer(16),
            ResidualLayer(16),
            ResidualLayer(16),
            ResidualLayer(16),
            ResidualLayer(16),
            DownSamplingLayer(16, 32, 3, 2),  # 23*23*32
            ConvolutionLayer(32, 16, 3, 1),  # 21*21*16
            ResidualLayer(16),
            ResidualLayer(16),
            ResidualLayer(16),
            ResidualLayer(16),
            DownSamplingLayer(16, 64, 3, 2),  # 10*10*64
            ConvolutionLayer(64, 16, 3, 1),  # 8*8*16
            ResidualLayer(16),
            ResidualLayer(16),
            DownSamplingLayer(16, 64, 2, 2),  # 4*4*64
            ConvolutionLayer(64, 128, 2, 1)  # 3*3*128
        )
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(3, 32, 3, 1, 1),  # 48*48*32
        #     nn.BatchNorm2d(32),
        #     nn.PReLU(),
        #     # nn.MaxPool2d(3, 2),  # 23*23*32
        #     nn.Conv2d(32, 32, 3, 2, 1),  # 24*24*32
        #     nn.BatchNorm2d(32),
        #     nn.PReLU(),
        #     nn.Conv2d(32, 64, 3, 1),  # 22*22*64
        #     nn.BatchNorm2d(64),
        #     nn.PReLU(),
        #     # nn.MaxPool2d(3, 2),  # 10*10*64
        #     nn.Conv2d(64, 64, 3, 2),  # 10*10*64
        #     nn.BatchNorm2d(64),
        #     nn.PReLU(),
        #     nn.Conv2d(64, 64, 3, 1),  # 8*8*64
        #     nn.BatchNorm2d(64),
        #     nn.PReLU(),
        #     # nn.MaxPool2d(2, 2),  # 4*4*64
        #     nn.Conv2d(64, 64, 3, 2, 1),  # 4*4*64
        #     nn.BatchNorm2d(64),
        #     nn.PReLU(),
        #     nn.Conv2d(64, 128, 2, 1),
        #     nn.BatchNorm2d(128),
        #     nn.PReLU(),
        # )

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


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    print(get_parameter_number(pnet))
    print(get_parameter_number(rnet))
    print(get_parameter_number(onet))
    p_input = torch.Tensor(100, 3, 12, 12)
    r_input = torch.Tensor(100, 3, 24, 24)
    o_input = torch.Tensor(100, 3, 48, 48)
    p_confidence, p_offset = pnet(p_input)
    r_confidence, r_offset, r_landmarks = rnet(r_input)
    o_confidence, o_offset, o_landmarks = onet(o_input)
    print(p_confidence.shape)
    print(r_confidence.shape)
    print(o_confidence.shape)
