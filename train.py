from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from dataset import datasets
import os


class Trainer:
    def __init__(self, net, dataset_path, save_path, isCuda=True):
        self.net = net
        self.dataset = DataLoader(datasets(dataset_path), batch_size=100, shuffle=True, drop_last=True, num_workers=2)
        self.save_path = save_path
        self.isCuda = isCuda

        if isCuda:
            self.net.cuda()

        self.confidence_loss_function = nn.BCELoss()
        self.offset_loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())

        if os.path.exists(save_path):
            self.net.load_state_dict(torch.load(save_path))

    def train(self):
        while True:
            previous_loss = 0
            for i, (image, confidence, offset) in enumerate(self.dataset):
                if self.isCuda:
                    image = image.cuda()
                    confidence = confidence.cuda()
                    offset = offset.cuda()
                out_confidence, out_offset = self.net(image)
                # 将输出的置信度的格式转换成与标签一样
                out_confidence = out_confidence.reshape(-1, 1)
                out_offset = out_offset.reshape(-1, 4)
                # 需要根据训练的目标删除个别样本
                # 使用正负样本训练置信度
                confidence_mask = torch.lt(confidence, 2)
                confidence_select = torch.masked_select(confidence, confidence_mask)
                out_confidence = torch.masked_select(out_confidence, confidence_mask)
                confidence_loss = self.confidence_loss_function(out_confidence, confidence_select)
                # 使用正、部分样本训练偏移量
                offset_mask = torch.gt(confidence, 0)
                offset_select = torch.masked_select(offset, offset_mask)
                out_offset = torch.masked_select(out_offset, offset_mask)
                offset_loss = self.offset_loss_function(out_offset, offset_select)
                total_loss = confidence_loss + offset_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # if i % 10 == 0:
                #     print("total_loss:{0},confidence_loss:{1},offset_loss:{2}".format(total_loss.item(),
                #                                                                       confidence_loss.item(),
                #                                                                       offset_loss.item()))
                #     torch.save(self.net.state_dict(), self.save_path)
                print("total_loss:{0},confidence_loss:{1},offset_loss:{2}".format(total_loss.item(),
                                                                                  confidence_loss.item(),
                                                                                  offset_loss.item()))
                if total_loss.item() < previous_loss:
                    previous_loss = total_loss.item()
                    torch.save(self.net.state_dict(), self.save_path)
                # print("success!")
