#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import cv2


def bb_intersection_over_union(boxA, boxB):
    """calculate intersection over union between two boundboxes.

    Args:
        boxA: list of xmin, ymin, xmax, ymax;
        boxB: list of xmin, ymin, xmax, ymax;
    Returns:
        a float number of iou between two inputs.
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def average_precision_11_point():
    return


def average_precision_matric():
    return


def resize_image_by_size(image, width, height):
    assert len(image.shape) == 3, 'Can only use RGB image.'
    original_height, original_width, channel= image.shape
    mask_image = np.zeros((height, width, channel), dtype=np.uint8)

    rate = min(float(width) / original_width, float(height) / original_height)
    new_width = int(original_width * rate)
    new_height = int(original_height * rate)

    horizion_bias = int((width - new_width) / 2)
    vertical_bias = int((height - new_height) / 2)

    resized_image = cv2.resize(image, (new_width, new_height))
    mask_image[vertical_bias:vertical_bias+new_height, horizion_bias:horizion_bias+new_width] = resized_image

    return mask_image, rate, (horizion_bias, vertical_bias)
