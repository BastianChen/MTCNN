import numpy as np
import torch

'''该类用于编写IOU、NMS以及真实框扩充成矩形的工具函数'''


# (x1,y1,x2,y2,c)
def IOU(box, boxes, isMin=False):
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    xx1, yy1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    xx2, yy2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    intersection = w * h
    if isMin:
        rate = np.true_divide(intersection, np.minimum(area, areas))
    else:
        rate = np.true_divide(intersection, area + areas - intersection)
    return rate


def NMS(boxes, threshold=0.3, isMin=False, method=1, sigma=0.5):
    if boxes.shape[0] == 0:
        return np.array([])
    boxes = boxes[(-boxes[:, 4]).argsort()]
    empty_boxes = []
    while boxes.shape[0] > 1:
        first_box = boxes[0]
        other_box = boxes[1:]
        empty_boxes.append(first_box)
        ious = IOU(first_box, other_box, isMin)
        if method == 1:  # nms
            index = np.where(IOU(first_box, other_box, isMin) < threshold)
            boxes = other_box[index]
        else:  # softnms
            weight_array = np.exp(-(ious ** 2) / sigma)
            # 更新置信度
            other_box[:, 4] = other_box[:, 4] * weight_array
            index = np.where(other_box[:, 4] > threshold)
            boxes = other_box[index]
            boxes = boxes[(-boxes[:, 4]).argsort()]
    if boxes.shape[0] > 0:
        empty_boxes.append(boxes[0])
    return np.stack(empty_boxes)


def convertToRectangle(crop_boxes):
    if crop_boxes.shape[0] == 0:
        return np.array([])
    new_boxes = crop_boxes.copy()
    w = crop_boxes[:, 2] - crop_boxes[:, 0]
    h = crop_boxes[:, 3] - crop_boxes[:, 1]
    face_length = np.maximum(w, h)
    # 通过将左上角坐标点加上二分之一的边长再减去二分之一的最大边长，从而获得新的左上角坐标
    new_boxes[:, 0] = crop_boxes[:, 0] + w * 0.5 - face_length * 0.5
    new_boxes[:, 1] = crop_boxes[:, 1] + h * 0.5 - face_length * 0.5
    new_boxes[:, 2], new_boxes[:, 3] = new_boxes[:, 0] + face_length, new_boxes[:, 1] + face_length
    return new_boxes


if __name__ == '__main__':
    # data = np.array([
    #     [10, 30, 50, 70, 0.2],
    #     [20, 15, 60, 60, 0.6],
    #     [5, 20, 30, 40, 0.98],
    #     [100, 80, 150, 130, 0.96],
    #     [125, 65, 165, 125, 0.57],
    #     [120, 60, 160, 100, 0.86],
    #     [145, 50, 170, 90, 0.83],
    # ])
    data = np.array([
        [1, 1, 100, 101, 0.9], [5, 6, 90, 110, 0.7], [17, 19, 80, 120, 0.8], [10, 8, 115, 105, 0.98]
    ])
    result = NMS(data)
    print(result)
