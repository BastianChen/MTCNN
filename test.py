import torch
import numpy as np
import utils
import nets
from torchvision import transforms
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import os
import time

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
        total_time = t_pnet + t_rnet + t_onet
        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(total_time, t_pnet, t_rnet, t_onet))
        return onet_boxes

    def pnet_detect(self, image):
        w, h = image.size
        min_length = np.minimum(w, h)
        boxex_all_nms = []
        scale = scale_new = 1

        while min_length > 12:
            image_data = self.trans(image).to(self.device)
            image_data.unsqueeze_(0)
            with torch.no_grad():
                confidence, offset = self.pnet(image_data)
                confidence = confidence[0][0].cpu().detach()
                offset = offset[0].cpu().detach()
                indexs = torch.nonzero(torch.gt(confidence, 0.6))
                if indexs.shape[0] == 0:
                    nms = np.array([])
                else:
                    boxes = self.backToImgae(np.array(indexs, dtype=np.float), offset, scale_new, confidence)
                    nms = utils.NMS(boxes, 0.5)
                boxex_all_nms.extend(nms)
                scale *= 0.7
                w_, h_ = int(w * scale), int(h * scale)
                min_length = np.minimum(w_, h_)
                scale_new = min_length / np.minimum(w, h)
                image = image.resize((w_, h_))
        return np.stack(boxex_all_nms)

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
            image_data = self.trans(img)
            img_dataset.append(image_data)

        img_dataset = torch.stack(img_dataset).to(self.device)

        with torch.no_grad():
            confidence, offset = self.rnet(img_dataset)

        confidence = confidence.cpu().detach().numpy()
        offset = offset.cpu().detach().numpy()

        indexs, _ = np.where(confidence > 0.6)
        if indexs.shape[0] == 0:
            return np.array([])
        else:
            boxes = pnet_boxes[indexs]

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
            image_data = self.trans(img)
            img_dataset.append(image_data)

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
            boxes = rnet_boxes[indexs]

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
        return utils.NMS(box, 0.7, isMin=True)

    def backToImgae(self, index, offset, scale, confidence, stride=2, length=12):
        x1_array = (index[:, 1] * stride) / scale
        y1_array = (index[:, 0] * stride) / scale
        x2_array = (index[:, 1] * stride + length) / scale
        y2_array = (index[:, 0] * stride + length) / scale

        w_array = x2_array - x1_array
        h_array = y2_array - y1_array

        offset = np.array(offset[:, index[:, 0], index[:, 1]])
        confidence = np.array(confidence[index[:, 0], index[:, 1]])

        x1_real = x1_array + w_array * offset[0]
        y1_real = y1_array + h_array * offset[1]
        x2_real = x2_array + w_array * offset[2]
        y2_real = y2_array + h_array * offset[3]

        return np.array([x1_real, y1_real, x2_real, y2_real, confidence]).T


if __name__ == '__main__':
    detector = Detector(r"models/pnet.pth", r"models/rnet.pth", r"models/onet.pth")
    image_path = r"F:\Photo_example\CelebA\test_image\0020bedfd030652e34fc101067625e43.jpg"
    with Image.open(image_path) as img_data:
        boxes = detector.detect(img_data)
        imgDraw = ImageDraw.ImageDraw(img_data)
        for box in boxes:
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]

            imgDraw.rectangle((x1, y1, x2, y2), outline="green", width=3)
        img_data.show()

        # img = Image.open(r"C:\Users\Administrator\Desktop\test\微信图片_20190909124341.jpg")
        # w, h = img.size
        # img = img.resize((int(0.33 * w), int(0.33 * h)))
        # img.save("{}.jpg".format(r"C:\Users\Administrator\Desktop\test\1.jpg"))

# if __name__ == '__main__':
#     box_path = r"text/list_bbox_celeba.txt"
#     landmarks_path = r"text/list_landmarks_celeba.txt"
#     # img_path = r"TESTIMG/TESTIMG/000058.jpg"
#     img_path = r"TESTIMG/TESTIMG"
#     box_array = open(box_path, "r").readlines()
#     landmarks_array = open(landmarks_path, "r").readlines()
#     for i, (box, landmarks) in enumerate(zip(box_array, landmarks_array)):
#         if i < 2:
#             continue
#         box = box.strip().split()
#         landmarks = landmarks.strip().split()
#         if 10 <= i <= 70:
#             w_box, h_box = int(box[3]), int(box[4])
#             w_landmarks, h_landmarks = int(landmarks[9]) - int(landmarks[1]), int(landmarks[10]) - int(
#                 landmarks[2])
#             center_x, center_y = int(int(landmarks[1]) + 0.5 * w_landmarks), int(
#                 int(landmarks[2]) + 0.5 * h_landmarks)
#
#             w_average, h_average = int((w_box + w_landmarks) / 2), int((h_box + h_landmarks) / 2)
#
#             img = Image.open(os.path.join(img_path, box[0]))
#             img_draw = ImageDraw.ImageDraw(img)
#             img_draw.rectangle(
#                 (float(center_x - 0.6 * w_average), float(center_y - 0.72 * h_average), float(center_x + 0.65 * w_average),
#                  float(center_y + 0.58 * h_average)), outline="green", width=3)
#             img_draw.point((int(landmarks[1]), int(landmarks[2])), fill="red")
#             img_draw.point((int(landmarks[3]), int(landmarks[4])), fill="red")
#             img_draw.point((int(landmarks[5]), int(landmarks[6])), fill="red")
#             img_draw.point((int(landmarks[7]), int(landmarks[8])), fill="red")
#             img_draw.point((int(landmarks[9]), int(landmarks[10])), fill="red")
#             img_draw.point((center_x, center_y), fill="blue")
#             img.show()
