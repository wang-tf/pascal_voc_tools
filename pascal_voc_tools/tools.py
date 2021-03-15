# -*- coding:utf-8 -*-
"""Some useful tools.
"""

import logging

logger = logging.getLogger(__name__)


def bb_intersection_over_union(box_a, box_b):
  """calculate intersection over union between two boundboxes.

  Args:
    box_a: list of xmin, ymin, xmax, ymax;
    box_b: list of xmin, ymin, xmax, ymax;
  Returns:
    a float number of iou between two inputs.
  """
  # determine the (x, y)-coordinates of the intersection rectangle
  x_a = max(box_a[0], box_b[0])
  y_a = max(box_a[1], box_b[1])
  x_b = min(box_a[2], box_b[2])
  y_b = min(box_a[3], box_b[3])

  # compute the area of intersection rectangle
  inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

  # compute the area of both the prediction and ground-truth
  # rectangles
  box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
  box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  try:
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
  except ZeroDivisionError as err:
    logger.exception(err)
    print(box_a, box_b)
  # return the intersection over union value
  return iou


def average_precision_11_point():
  return


def average_precision_matric():
  return
