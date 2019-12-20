# -*- coding:utf-8 -*-


def xyxy2xcycwh(box):
    x_center = (box[0] + box[2]) / 2.0 - 1
    y_center = (box[1] + box[3]) / 2.0 - 1
    width = box[1] - box[0]
    height = box[3] - box[2]
    return [x_center, y_center, width, height]


def bbox_absolute2relative(bbox, width, height):
    b1 = bbox[0] / width
    b2 = bbox[1] / height
    b3 = bbox[2] / width
    b4 = bbox[3] / height
    return [b1, b2, b3, b4]