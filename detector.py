import nets
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import utils
import numpy as np
from torchvision import transforms
import torch
import time
import os
import cv2


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
        # print("pnet:{0}".format(t_pnet))
        return onet_boxes

    def pnet_detect(self, image):
        # 用于存放所有经过NMS删选的真实框
        boxes_nms_all = []
        w, h = image.size
        # 侦测图片中各种大小的人脸
        # min_length = np.minimum(w, h)
        # scale = scale_new = 1
        # 用于侦测图片，且图片里的人脸比较大
        scale = 0.7
        # 用于侦测视频，且视频里的人脸比较大
        # scale = 0.7**10
        w_ = int(w * scale)
        h_ = int(h * scale)
        min_length = np.minimum(w_, h_)
        scale_new = min_length / np.minimum(w, h)
        image = image.resize((w_, h_))

        while min_length > 12:
            img_data = self.trans(image).to(self.device)
            # 升维，因为存在批次这一维度
            img_data.unsqueeze_(0)
            with torch.no_grad():
                confidence, offset, _ = self.pnet(img_data)
            confidence = confidence[0][0].cpu().detach()
            offset = offset[0].cpu().detach()
            # 根据阈值先删除掉一些置信度低的候选框,并返回符合要求的索引
            indexs = torch.nonzero(torch.gt(confidence, 0.8))
            if indexs.shape[0] == 0:
                nms = np.array([])
            else:
                boxes = self.backToImage(np.array(indexs, dtype=np.float), offset, scale_new, confidence)
                nms = utils.NMS(boxes, 0.3)
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
            confidence, offset, _ = self.rnet(img_dataset)

        confidence = confidence.cpu().detach().numpy()
        offset = offset.cpu().detach().numpy()

        indexs, _ = np.where(confidence > 0.93)
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
            box = np.stack([x1_real, y1_real, x2_real, y2_real, confidence[:, 0]], axis=1)
        return utils.NMS(box, 0.3)

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

        indexs, _ = np.where(confidence > 0.99)
        if indexs.shape[0] == 0:
            return np.array([])
        else:
            boxes = rnet_boxes[indexs]
            x1_array = boxes[:, 0]
            y1_array = boxes[:, 1]
            x2_array = boxes[:, 2]
            y2_array = boxes[:, 3]

            w_array = x2_array - x1_array
            h_array = y2_array - y1_array

            offset = offset[indexs]
            confidence = confidence[indexs]
            landmarks = landmarks[indexs]

            x1_real = x1_array + w_array * offset[:, 0]
            y1_real = y1_array + h_array * offset[:, 1]
            x2_real = x2_array + w_array * offset[:, 2]
            y2_real = y2_array + h_array * offset[:, 3]

            landmarks_x1, landmarks_y1 = x1_array + w_array * landmarks[:, 0], y1_array + h_array * landmarks[:, 1]
            landmarks_x2, landmarks_y2 = x1_array + w_array * landmarks[:, 2], y1_array + h_array * landmarks[:, 3]
            landmarks_x3, landmarks_y3 = x1_array + w_array * landmarks[:, 4], y1_array + h_array * landmarks[:, 5]
            landmarks_x4, landmarks_y4 = x1_array + w_array * landmarks[:, 6], y1_array + h_array * landmarks[:, 7]
            landmarks_x5, landmarks_y5 = x1_array + w_array * landmarks[:, 8], y1_array + h_array * landmarks[:, 9]

            boxes = np.stack([x1_real, y1_real, x2_real, y2_real, confidence[:, 0], landmarks_x1, landmarks_y1,
                              landmarks_x2, landmarks_y2, landmarks_x3, landmarks_y3, landmarks_x4, landmarks_y4,
                              landmarks_x5, landmarks_y5], axis=1)
            # 判断关键点是否在真实框中
            empty_box = []
            for box in boxes:
                if (box[5] > box[0] and box[6] > box[1] and box[7] < box[2] and box[8] > box[1]) and (
                        box[9] > box[0] and box[10] > box[1] and box[9] < box[2] and box[10] < box[3]) and (
                        box[11] > box[0] and box[12] < box[3] and box[13] < box[2] and box[14] < box[3]):
                    empty_box.append(box)
            boxes = np.stack(empty_box)
        # box = np.stack(boxes)
        return utils.NMS(boxes, 0.3, isMin=True)

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
        return np.stack([x1_real, y1_real, x2_real, y2_real, confidence], axis=1)


if __name__ == '__main__':
    detector = Detector(r"models/pnet_depthwiseconv.pth", r"models/rnet_depthwiseconv.pth",
                        r"models/onet_residualconv.pth")  # 加了五个关键点
    # detector = Detector(r"models/pnet_normal_data_enhancement.pth", r"models/rnet_depthwiseconv.pth",
    #                     r"models/onet_residualconv.pth")  # P网络加入了数据增强，效果不好
    # detector = Detector(r"models_old/pnet.pth", r"models_old/rnet.pth", r"models_old/onet.pth")# 没加五个坐标点

    # 用opencv侦测图片
    # image_path = r"F:\Photo_example\CelebA\test_image"
    image_path = r"C:\Users\Administrator\Desktop\test"
    image_list = os.listdir(image_path)
    for path in image_list:
        img = cv2.imread(os.path.join(image_path, path))
        # img = cv2.medianBlur(img, 5)
        img_revert = img[:, :, ::-1]
        image_data = Image.fromarray(img_revert, "RGB")
        boxes = detector.detect(image_data)
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
            x1 = int(landmarks_x3 - 0.6 * w_avergae)
            y1 = int(landmarks_y3 - 1.2 * h_avergae)
            x2 = int(x1 + 1.4 * w_avergae)
            # y2 = int(y1 + 2.0 * h_avergae)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.rectangle(img, (landmarks_x1 - 1, landmarks_y1 - 1), (landmarks_x1 + 1, landmarks_y1 + 1),
                          (255, 0, 0))
            cv2.rectangle(img, (landmarks_x2 - 1, landmarks_y2 - 1), (landmarks_x2 + 1, landmarks_y2 + 1),
                          (255, 0, 0))
            cv2.rectangle(img, (landmarks_x3 - 1, landmarks_y3 - 1), (landmarks_x3 + 1, landmarks_y3 + 1),
                          (255, 0, 0))
            cv2.rectangle(img, (landmarks_x4 - 1, landmarks_y4 - 1), (landmarks_x4 + 1, landmarks_y4 + 1),
                          (255, 0, 0))
            cv2.rectangle(img, (landmarks_x5 - 1, landmarks_y5 - 1), (landmarks_x5 + 1, landmarks_y5 + 1),
                          (255, 0, 0))
        cv2.namedWindow("MTCNN", 0)
        cv2.imshow("MTCNN", img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # 用PIL侦测图片
    # # image_path = r"F:\Photo_example\CelebA\test_image"
    # image_path = r"C:\Users\Administrator\Desktop\test"
    # image_list = os.listdir(image_path)
    # for path in image_list:
    #     with Image.open(os.path.join(image_path, path)) as img:
    #         boxes = detector.detect(img)
    #         imDraw = ImageDraw.ImageDraw(img)
    #         for box in boxes:
    #             x1 = int(box[0])
    #             y1 = int(box[1])
    #             x2 = int(box[2])
    #             y2 = int(box[3])
    #             w, h = x2 - x1, y2 - y1
    #             landmarks_x1, landmarks_y1 = int(box[5]), int(box[6])
    #             landmarks_x2, landmarks_y2 = int(box[7]), int(box[8])
    #             landmarks_x3, landmarks_y3 = int(box[9]), int(box[10])
    #             landmarks_x4, landmarks_y4 = int(box[11]), int(box[12])
    #             landmarks_x5, landmarks_y5 = int(box[13]), int(box[14])
    #             landmarks_w1 = landmarks_x2 - landmarks_x1
    #             landmarks_w2 = landmarks_x5 - landmarks_x4
    #             landmarks_h1 = landmarks_y2 - landmarks_y1
    #             landmarks_h2 = landmarks_y5 - landmarks_y4
    #             landmarks_w_average = (landmarks_w1 + landmarks_w2) / 2
    #             landmarks_h_average = (landmarks_h1 + landmarks_h2) / 2
    #             w_avergae = (landmarks_w_average + w) / 2
    #             h_avergae = (landmarks_h_average + h) / 2
    #             x1 = landmarks_x3 - 0.6 * w_avergae
    #             y1 = landmarks_y3 - 1.2 * h_avergae
    #             x2 = x1 + 1.4 * w_avergae
    #             y2 = y1 + 2.0 * h_avergae
    #             imDraw.rectangle((x1, y1, x2, y2), outline='red', width=5)
    #             imDraw.rectangle((landmarks_x1 - 1, landmarks_y1 - 1, landmarks_x1 + 1, landmarks_y1 + 1), fill="blue")
    #             imDraw.rectangle((landmarks_x2 - 1, landmarks_y2 - 1, landmarks_x2 + 1, landmarks_y2 + 1), fill="blue")
    #             imDraw.rectangle((landmarks_x3 - 1, landmarks_y3 - 1, landmarks_x3 + 1, landmarks_y3 + 1), fill="blue")
    #             imDraw.rectangle((landmarks_x4 - 1, landmarks_y4 - 1, landmarks_x4 + 1, landmarks_y4 + 1), fill="blue")
    #             imDraw.rectangle((landmarks_x5 - 1, landmarks_y5 - 1, landmarks_x5 + 1, landmarks_y5 + 1), fill="blue")
    #         img.show()

    # 侦测视频
    # # cap = cv2.VideoCapture("video/jj.mp4")
    # cap = cv2.VideoCapture("video/jay.mp4")
    # # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
    # # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # # out = cv2.VideoWriter('output.avi', fourcc, 20.0, size)
    # i = 0
    # boxes = []
    # x1 = 0
    # y1 = 0
    # x2 = 0
    # y2 = 0
    # landmarks_x1, landmarks_y1 = 0, 0
    # landmarks_x2, landmarks_y2 = 0, 0
    # landmarks_x3, landmarks_y3 = 0, 0
    # landmarks_x4, landmarks_y4 = 0, 0
    # landmarks_x5, landmarks_y5 = 0, 0
    # while True:
    #     ret, frame = cap.read()
    #     if ret:
    #         # 对每一帧做中值滤波，保存边缘信息降低噪声
    #         frame = cv2.medianBlur(frame, 5)
    #         frame_revert = frame[:, :, ::-1]
    #         image_data = Image.fromarray(frame_revert, "RGB")
    #         if i % 4 == 0:
    #             detector.detect(image_data)
    #             boxes = detector.detect(image_data)
    #             if boxes.shape[0] != 0:
    #                 for box in boxes:
    #                     x1 = int(box[0])
    #                     y1 = int(box[1])
    #                     x2 = int(box[2])
    #                     y2 = int(box[3])
    #                     w, h = x2 - x1, y2 - y1
    #                     landmarks_x1, landmarks_y1 = int(box[5]), int(box[6])
    #                     landmarks_x2, landmarks_y2 = int(box[7]), int(box[8])
    #                     landmarks_x3, landmarks_y3 = int(box[9]), int(box[10])
    #                     landmarks_x4, landmarks_y4 = int(box[11]), int(box[12])
    #                     landmarks_x5, landmarks_y5 = int(box[13]), int(box[14])
    #                     landmarks_w1 = landmarks_x2 - landmarks_x1
    #                     landmarks_w2 = landmarks_x5 - landmarks_x4
    #                     landmarks_h1 = landmarks_y2 - landmarks_y1
    #                     landmarks_h2 = landmarks_y5 - landmarks_y4
    #                     landmarks_w_average = (landmarks_w1 + landmarks_w2) / 2
    #                     landmarks_h_average = (landmarks_h1 + landmarks_h2) / 2
    #                     w_avergae = (landmarks_w_average + w) / 2
    #                     h_avergae = (landmarks_h_average + h) / 2
    #                     x1 = int(landmarks_x3 - 0.6 * w_avergae)
    #                     y1 = int(landmarks_y3 - 1.2 * h_avergae)
    #                     x2 = int(x1 + 1.4 * w_avergae)
    #                     # y2 = int(y1 + 2.0 * h_avergae)
    #         if boxes.shape[0] != 0:
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    #             cv2.rectangle(frame, (landmarks_x1 - 1, landmarks_y1 - 1), (landmarks_x1 + 1, landmarks_y1 + 1),
    #                           (255, 0, 0))
    #             cv2.rectangle(frame, (landmarks_x2 - 1, landmarks_y2 - 1), (landmarks_x2 + 1, landmarks_y2 + 1),
    #                           (255, 0, 0))
    #             cv2.rectangle(frame, (landmarks_x3 - 1, landmarks_y3 - 1), (landmarks_x3 + 1, landmarks_y3 + 1),
    #                           (255, 0, 0))
    #             cv2.rectangle(frame, (landmarks_x4 - 1, landmarks_y4 - 1), (landmarks_x4 + 1, landmarks_y4 + 1),
    #                           (255, 0, 0))
    #             cv2.rectangle(frame, (landmarks_x5 - 1, landmarks_y5 - 1), (landmarks_x5 + 1, landmarks_y5 + 1),
    #                           (255, 0, 0))
    #         cv2.namedWindow("video", 0)
    #         cv2.imshow("video", frame)
    #         cv2.waitKey(1)
    #         # out.write(frame)
    #     else:
    #         cap.release()
    #         # out.release()
    #         cv2.destroyAllWindows()
    #         break
    #     i += 1
