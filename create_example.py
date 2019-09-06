import os
import numpy as np
import utils
import PIL.Image as image
import traceback

'''该类用于生成正、负以及部分样本'''

image_path = r"F:\Photo_example\CelebA\photos\img_celeba"
# image_path = "TESTIMG/TESTIMG"
box_path = r"text/list_bbox_celeba.txt"
landmarks_path = r"text/list_landmarks_celeba.txt"
# save_path = r"F:\Photo_example\CelebA\sample"
# save_path = r"C:\new_sample"
save_path = r"C:\sample"
# save_path = "celebA"
for size in [12, 24, 48]:
    print("create {} start".format(size))
    positive_image_path = os.path.join(save_path, str(size), "positive")
    part_image_path = os.path.join(save_path, str(size), "part")
    negative_image_path = os.path.join(save_path, str(size), "negative")

    for path in [positive_image_path, part_image_path, negative_image_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    positive_text_path = os.path.join(save_path, str(size), "positive.txt")
    part_text_path = os.path.join(save_path, str(size), "part.txt")
    negative_text_path = os.path.join(save_path, str(size), "negative.txt")

    positive_text = open(positive_text_path, "w")
    part_text = open(part_text_path, "w")
    negative_text = open(negative_text_path, "w")

    positive_count = 0
    part_count = 0
    negative_count = 0

    for i, (box, landmarks) in enumerate(zip(open(box_path), open(landmarks_path))):
        if i < 2:
            continue
        # if i > 89:
        #     break
        if i > 2000:
            break
        try:
            box = box.strip().split()
            landmarks = landmarks.strip().split()
            # 算出样本标签的宽高
            w_box, h_box = int(box[3]), int(box[4])
            # 算出样本五官的宽距与高距
            w_landmarks, h_landmarks = int(landmarks[9]) - int(landmarks[1]), int(landmarks[10]) - int(
                landmarks[2])
            # 算出样本五官的中心点
            center_x, center_y = int(int(landmarks[1]) + 0.5 * w_landmarks), int(
                int(landmarks[2]) + 0.5 * h_landmarks)
            # 根据样本标签的宽高和五官的宽距与高距算出平均样本宽高
            w_average, h_average = int((w_box + w_landmarks) / 2), int((h_box + h_landmarks) / 2)
            img_path = os.path.join(image_path, box[0].strip())
            with image.open(img_path) as img:
                img_w, img_h = img.size
                x1 = float(box[1].strip())
                y1 = float(box[2].strip())
                w = float(box[3].strip())
                h = float(box[4].strip())
                x2 = x1 + w
                y2 = y1 + h

                if max(w, h) < 40 or w < 0 or h < 0 or x1 < 0 or y1 < 0:
                    continue

                box = [x1, y1, x2, y2]

                # 算出中心点坐标
                center_x = x1 + w / 2
                center_y = y1 + h / 2

                # for _ in range(9):
                for _ in range(16):
                    # 让中心点有少许偏移
                    # w_randint = np.random.randint(-w * 0.2, w * 0.2)
                    # h_randint = np.random.randint(-h * 0.2, h * 0.2)
                    w_randint = np.random.randint(-w * 0.4, w * 0.4)
                    h_randint = np.random.randint(-h * 0.4, h * 0.4)
                    center_x_new = center_x + w_randint
                    center_y_new = center_y + h_randint

                    # rectangle_length = np.random.randint(int(np.minimum(w, h) * 0.3), np.ceil(np.maximum(w, h) * 1))
                    rectangle_length = np.random.randint(int(np.minimum(w, h) * 0.8), np.ceil(np.maximum(w, h) * 1.25))
                    x1_new = np.maximum(center_x_new - rectangle_length * 0.5, 0)
                    y1_new = np.maximum(center_y_new - rectangle_length * 0.5, 0)
                    x2_new = x1_new + rectangle_length
                    y2_new = y1_new + rectangle_length

                    # 计算偏移量
                    offset_x1 = (x1 - x1_new) / rectangle_length
                    offset_y1 = (y1 - y1_new) / rectangle_length
                    offset_x2 = (x2 - x2_new) / rectangle_length
                    offset_y2 = (y2 - y2_new) / rectangle_length
                    # 将制作的样本用list保存
                    crop_box = [x1_new, y1_new, x2_new, y2_new]

                    crop_img = img.crop(crop_box)
                    crop_img_resize = crop_img.resize((size, size))
                    crop_box = np.array([crop_box])

                    # 计算IOU来判断将样本保存为哪种类型
                    rate = utils.IOU(box, crop_box)[0]
                    # if rate > 0.68:
                    if rate > 0.64:
                        positive_text.write(
                            "positive/{0}.jpg {1} {2} {3} {4} {5}\n".format(positive_count, offset_x1, offset_y1,
                                                                            offset_x1,
                                                                            offset_y2, 1))
                        # 刷新缓存区，防止进程意外退出或正常退出时而未执行文件的close方法，缓冲区中的内容将会丢失。
                        positive_text.flush()
                        crop_img_resize.save(os.path.join(positive_image_path, "{}.jpg".format(positive_count)))
                        positive_count += 1
                    # elif rate > 0.45 and rate < 0.53:
                    elif 0.23 < rate < 0.31:
                        part_text.write(
                            "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(part_count, offset_x1, offset_y1, offset_x1,
                                                                        offset_y2, 2))
                        part_text.flush()
                        crop_img_resize.save(os.path.join(part_image_path, "{}.jpg".format(part_count)))
                        part_count += 1
                    elif rate < 0.1:
                        negative_text.write("negative/{0}.jpg 0 0 0 0 0\n".format(negative_count))
                        negative_text.flush()
                        crop_img_resize.save(os.path.join(negative_image_path, "{}.jpg".format(negative_count)))
                        negative_count += 1
                # 生成负样本
                # for _ in range(4):
                for _ in range(5):
                    rectangle_length_negative = np.random.randint(size, np.minimum(img_w, img_h) / 2)
                    x1_negative = np.random.randint(0, img_w - rectangle_length_negative)
                    y1_negative = np.random.randint(0, img_h - rectangle_length_negative)
                    crop_box_negative = [x1_negative, y1_negative, x1_negative + rectangle_length_negative,
                                         y1_negative + rectangle_length_negative]
                    rate_negative = utils.IOU(box, np.array([crop_box_negative]))[0]
                    # print(rate_negative)

                    if rate_negative < 0.1:
                        crop_img_negative = img.crop(crop_box_negative)
                        crop_img_resize_negative = crop_img_negative.resize((size, size))
                        negative_text.write("negative/{0}.jpg 0 0 0 0 0\n".format(negative_count))
                        negative_text.flush()
                        crop_img_resize_negative.save(
                            os.path.join(negative_image_path, "{}.jpg".format(negative_count)))
                        negative_count += 1
        except Exception as e:
            traceback.print_exc()
