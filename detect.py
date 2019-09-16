import nets
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import utils
import numpy as np
from torchvision import transforms
import torch
import time
import os


class Detector:
    def __init__(self, pnet_path, rnet_path, onet_path, isCuda=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.isCuda = isCuda
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

    def detect(self, image):
        start_time = time.time()
        pnet_boxes = self.pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_pnet = end_time - start_time
        start_time = time.time()
        rnet_boxes = self.rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time

        start_time = time.time()
        onet_boxes = self.onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_onet = end_time - start_time

        t_sum = t_pnet + t_rnet + t_onet

        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return onet_boxes

    def pnet_detect(self, image):

        # 用于存放所有经过NMS删选的真实框
        boxes_nms_all = []
        w, h = image.size
        min_length = np.minimum(w, h)
        scale = scale_new = 1

        while min_length > 12:
            img_data = self.trans(image).to(self.device)
            # 升维，因为存在批次这一维度
            img_data.unsqueeze_(0)
            with torch.no_grad():
                confidence, offset, landmarks = self.pnet(img_data)
            confidence = confidence[0][0].cpu().detach()
            offset = offset[0].cpu().detach()
            # 根据阈值先删除掉一些置信度低的候选框,并返回符合要求的索引
            # indexs = torch.nonzero(torch.gt(confidence, 0.6))
            indexs = torch.nonzero(torch.gt(confidence, 0.6))
            if indexs.shape[0] == 0:
                nms = np.array([])
            else:
                boxes = self.backToImage(np.array(indexs, dtype=np.float), offset, scale_new, confidence)
                nms = utils.NMS(boxes, 0.5)
            boxes_nms_all.extend(nms)
            scale *= 0.7
            w_ = int(w * scale)
            h_ = int(h * scale)
            min_length = np.minimum(w_, h_)
            scale_new = min_length / np.minimum(w, h)
            image = image.resize((w_, h_))
        if len(boxes_nms_all) == 0:
            return np.array([])
        boxes_nms_all = np.stack(boxes_nms_all)
        return boxes_nms_all

    def rnet_detect(self, image, pnet_boxes):
        img_dataset = []
        pnet_boxes = utils.convertToRectangle(pnet_boxes)
        for pnet_box in pnet_boxes:
            x1 = int(pnet_box[0])
            y1 = int(pnet_box[1])
            x2 = int(pnet_box[2])
            y2 = int(pnet_box[3])

            img = image.crop((x1, y1, x2, y2))
            img = img.resize((24, 24))
            img_data = self.trans(img)
            img_dataset.append(img_data)

        img_dataset = torch.stack(img_dataset).to(self.device)

        with torch.no_grad():
            confidence, offset, landmarks = self.rnet(img_dataset)

        confidence = confidence.cpu().detach().numpy()
        offset = offset.cpu().detach().numpy()

        indexs, _ = np.where(confidence > 0.6)
        if indexs.shape[0] == 0:
            return np.array([])
        else:
            boxes = pnet_boxes[indexs]
            # 直接返回到P网络传入的真实框
            x1_array = boxes[:, 0]
            y1_array = boxes[:, 1]
            x2_array = boxes[:, 2]
            y2_array = boxes[:, 3]

            w_array = x2_array - x1_array
            h_array = y2_array - y1_array

            offset = offset[indexs]
            confidence = confidence[indexs]

            x1_real = x1_array + w_array * offset[:, 0]
            y1_real = y1_array + h_array * offset[:, 1]
            x2_real = x2_array + w_array * offset[:, 2]
            y2_real = y2_array + h_array * offset[:, 3]

            box = np.array([x1_real, y1_real, x2_real, y2_real, confidence[:, 0]]).T
        return utils.NMS(box, 0.5)

    def onet_detect(self, image, rnet_boxes):
        img_dataset = []
        rnet_boxes = utils.convertToRectangle(rnet_boxes)
        for rnet_box in rnet_boxes:
            x1 = int(rnet_box[0])
            y1 = int(rnet_box[1])
            x2 = int(rnet_box[2])
            y2 = int(rnet_box[3])

            img = image.crop((x1, y1, x2, y2))
            img = img.resize((48, 48))
            img_data = self.trans(img)
            img_dataset.append(img_data)

        img_dataset = torch.stack(img_dataset).to(self.device)

        with torch.no_grad():
            confidence, offset, landmarks = self.onet(img_dataset)
        confidence = confidence.cpu().detach().numpy()
        offset = offset.cpu().detach().numpy()
        landmarks = landmarks.cpu().detach().numpy()

        indexs, _ = np.where(confidence > 0.97)
        if indexs.shape[0] == 0:
            return np.array([])
        else:
            boxes = []
            # 判断关键点是否在建议框中
            for index in indexs:
                box = rnet_boxes[index]
                offset_new = offset[index]
                confidence_new = confidence[index]
                landmark = landmarks[index]

                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]

                w = x2 - x1
                h = y2 - y1

                x1_real = x1 + w * offset_new[0]
                y1_real = y1 + h * offset_new[1]
                x2_real = x2 + w * offset_new[2]
                y2_real = y2 + h * offset_new[3]

                landmark_x1, landmark_y1 = landmark[0], landmark[1]
                landmark_x2, landmark_y2 = landmark[2], landmark[3]
                landmark_x3, landmark_y3 = landmark[4], landmark[5]
                landmark_x4, landmark_y4 = landmark[6], landmark[7]
                landmark_x5, landmark_y5 = landmark[8], landmark[9]

                landmark_x1_real, landmark_y1_real = x1 + w * landmark_x1, y1 + h * landmark_y1
                landmark_x2_real, landmark_y2_real = x1 + w * landmark_x2, y1 + h * landmark_y2
                landmark_x3_real, landmark_y3_real = x1 + w * landmark_x3, y1 + h * landmark_y3
                landmark_x4_real, landmark_y4_real = x1 + w * landmark_x4, y1 + h * landmark_y4
                landmark_x5_real, landmark_y5_real = x1 + w * landmark_x5, y1 + h * landmark_y5

                if (
                        landmark_x1_real > x1_real and landmark_y1_real > y1_real and landmark_x2_real < x2_real and landmark_y2_real < y2_real) and (
                        landmark_x3_real > x1_real and landmark_y3_real > y1_real and landmark_x3_real < x2_real and landmark_y3_real < y2_real) and (
                        landmark_x4_real > x1_real and landmark_y4_real > y1_real and landmark_x5_real < x2_real and landmark_y5_real < y2_real):
                    boxes.append(
                        [x1_real, y1_real, x2_real, y2_real, confidence_new[0], landmark_x1_real, landmark_y1_real,
                         landmark_x2_real, landmark_y2_real, landmark_x3_real, landmark_y3_real, landmark_x4_real,
                         landmark_y4_real, landmark_x5_real, landmark_y5_real])
            # 不判断关键点是否在建议框中
            # boxes = rnet_boxes[indexs]
            # x1_array = boxes[:, 0]
            # y1_array = boxes[:, 1]
            # x2_array = boxes[:, 2]
            # y2_array = boxes[:, 3]
            #
            # w_array = x2_array - x1_array
            # h_array = y2_array - y1_array
            #
            # offset = offset[indexs]
            # confidence = confidence[indexs]
            # landmarks = landmarks[indexs]
            #
            # x1_real = x1_array + w_array * offset[:, 0]
            # y1_real = y1_array + h_array * offset[:, 1]
            # x2_real = x2_array + w_array * offset[:, 2]
            # y2_real = y2_array + h_array * offset[:, 3]
            #
            # landmarks_x1, landmarks_y1 = x1_array + w_array * landmarks[:, 0], y1_array + h_array * landmarks[:, 1]
            # landmarks_x2, landmarks_y2 = x1_array + w_array * landmarks[:, 2], y1_array + h_array * landmarks[:, 3]
            # landmarks_x3, landmarks_y3 = x1_array + w_array * landmarks[:, 4], y1_array + h_array * landmarks[:, 5]
            # landmarks_x4, landmarks_y4 = x1_array + w_array * landmarks[:, 6], y1_array + h_array * landmarks[:, 7]
            # landmarks_x5, landmarks_y5 = x1_array + w_array * landmarks[:, 8], y1_array + h_array * landmarks[:, 9]
            #
            # box = np.array([x1_real, y1_real, x2_real, y2_real, confidence[:, 0], landmarks_x1, landmarks_y1,
            #                 landmarks_x2, landmarks_y2, landmarks_x3, landmarks_y3, landmarks_x4, landmarks_y4,
            #                 landmarks_x5, landmarks_y5]).T
        box = np.stack(boxes)
        return utils.NMS(box, 0.7, isMin=True)

    # 用于根据偏移量还原真实框到原图
    def backToImage(self, index, offset, scale, confidence, stride=2, side_len=12):
        x1_array = (index[:, 1] * stride) / scale
        y1_array = (index[:, 0] * stride) / scale
        x2_array = (index[:, 1] * stride + side_len) / scale
        y2_array = (index[:, 0] * stride + side_len) / scale

        # 算出建议框的w和h，用于后面根据偏移量算出真实框的坐标
        w_array = x2_array - x1_array
        h_array = y2_array - y1_array

        offset = np.array(offset[:, index[:, 0], index[:, 1]])
        confidence = np.array(confidence[index[:, 0], index[:, 1]])

        x1_real = x1_array + w_array * offset[0]
        y1_real = y1_array + h_array * offset[1]
        x2_real = x2_array + w_array * offset[2]
        y2_real = y2_array + h_array * offset[3]

        # 返回真实框
        return np.array([x1_real, y1_real, x2_real, y2_real, confidence]).T


if __name__ == '__main__':
    detector = Detector(r"models/pnet.pth", r"models/rnet.pth", r"models/onet.pth")  # 加了五个坐标点
    # detector = Detector(r"models_old/pnet.pth", r"models_old/rnet.pth", r"models_old/onet.pth", isCuda)# 没加五个坐标点
    # image_path = r"F:\Photo_example\CelebA\test_image"
    image_path = r"C:\Users\Administrator\Desktop\test"
    image_list = os.listdir(image_path)

    for path in image_list:
        with Image.open(os.path.join(image_path, path)) as img:
            boxes = detector.detect(img)
            imDraw = ImageDraw.ImageDraw(img)
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                w, h = x2 - x1, y2 - y1
                landmarks_x1, landmarks_y1 = int(box[5]), int(box[6])
                landmarks_x2, landmarks_y2 = int(box[7]), int(box[8])
                landmarks_x3, landmarks_y3 = int(box[9]), int(box[10])
                landmarks_x4, landmarks_y4 = int(box[11]), int(box[12])
                landmarks_x5, landmarks_y5 = int(box[13]), int(box[14])
                landmarks_w1 = landmarks_x2 - landmarks_x1
                landmarks_w2 = landmarks_x5 - landmarks_x4
                landmarks_h1 = landmarks_y2 - landmarks_y1
                landmarks_h2 = landmarks_y5 - landmarks_y4
                landmarks_w_average = (landmarks_w1 + landmarks_w2) / 2
                landmarks_h_average = (landmarks_h1 + landmarks_h2) / 2
                w_avergae = (landmarks_w_average + w) / 2
                h_avergae = (landmarks_h_average + h) / 2
                x1 = landmarks_x3 - 0.6 * w_avergae
                y1 = landmarks_y3 - h_avergae
                x2 = x1 + 1.2 * w_avergae
                y2 = y1 + 1.8 * h_avergae
                imDraw.rectangle((x1, y1, x2, y2), outline='red', width=3)
            img.show()
