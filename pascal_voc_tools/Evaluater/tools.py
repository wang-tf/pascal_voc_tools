#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File : tools.py
@Time : 2019/03/04 08:35:46
@Author : wangtf
@Version : 1.0
@Desc : None
'''

# here put the import lib
import numpy as np


def voc_ap(recall, precision, use_07_metric=False):
    """ 
    ap = voc_ap(recall, precision, [use_07_metric])

    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses  the
    VOC 07 11 point method (default: False).
    Please make shure that recall and precison are sorted by scores.

    Arguments:
    ==========
        recall: the shape of (n,) ndarray;
        precision: the shape of (n,) ndarray;
        use_07_metric: if true, the 11 points method will be used.
    Return:
    =======
        the float number result of average precision.
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
    return ap


def compute_overlaps(boxes, one_box):
    """
    iou = compute_overlaps(boxes, one_box)

    compute intersection over union of ndarray.
    The format of one_box is [xmin, ymin, xmax, ymax].

    Arguments:
    ==========
        boxes: the (n, 4) shape ndarray, ground truth boundboxes;
        bb: the (4,) shape ndarray, detected boundboxes;
    Return:
    =======
        a (n, ) shape ndarray.
    """
    # compute overlaps
    # intersection
    ixmin = np.maximum(boxes[:, 0], one_box[0])
    iymin = np.maximum(boxes[:, 1], one_box[1])
    ixmax = np.minimum(boxes[:, 2], one_box[2])
    iymax = np.minimum(boxes[:, 3], one_box[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    boxes_area = (boxes[:, 2] - boxes[:, 0] + 1.) * (boxes[:, 3] - boxes[:, 1] + 1.)
    one_box_area = (one_box[2] - one_box[0] + 1.) * (one_box[3] - one_box[1] + 1.)
    iou = inters / (one_box_area + boxes_area - inters)

    return iou


def voc_eval(class_recs, detect, iou_thresh=0.5, use_07_metric=False):
    """
    recall, precision, ap = voc_eval(class_recs, detection,
                                [iou_thresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.
    Please make sure that the class_recs only have one class annotations.

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    Arguments:
    ==========
        class_recalls: recalls dict of a class
            class_recs[image_name]={'bbox': [], 'difficult': []}.
        detection: Path to annotations
            detection={'image_ids':[], bbox': [], 'confidence':[]}.
        [iou_thresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use VOC07's 11 point AP computation
            (default False)
    Return:
    =======
        a dict of result including true_positive_number, false_positive_number,
        recall, precision and average_precision.
    """
    # format data
    # class_rec data load
    npos = 0
    for imagename in class_recs.keys():
        assert isinstance(class_recs[imagename]['bbox'], np.ndarray)

        npos += class_recs[imagename]['bbox'].shape[0]
        class_recs[imagename]['det'] = [False] * class_recs[imagename]['bbox'].shape[0]
    # detections data load
    image_ids = detect['image_ids']
    confidence = detect['confidence']
    BB = detect['bbox']
    assert isinstance(confidence, np.ndarray)
    assert isinstance(BB, np.ndarray)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        iou_max = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            overlaps = compute_overlaps(BBGT, bb)
            iou_max = np.max(overlaps)
            iou_max_index = np.argmax(overlaps)

        if iou_max > iou_thresh:
            if not R['det'][iou_max_index]:
                tp[d] = 1.
                R['det'][iou_max_index] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    true_positive_number = tp[-1]
    false_positive_number = fp[-1]

    recall = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult ground truth
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    average_precision = voc_ap(recall, precision, use_07_metric)

    result = {}
    result['true_positive_number'] = true_positive_number
    result['false_positive_number'] = false_positive_number
    result['recall'] = recall
    result['precision'] = precision
    result['average_precision'] = average_precision
    return result


# class EvaluatTools():
#     def __init__(self):

#     def get_recs():
#         recs = {}
#         for i, imagename in enumerate(imagenames):
#             recs[imagename] = parse_rec(annopath.format(imagename))

#     # extract gt objects for this class
#     class_recs = {}
#     for imagename in recs.keys():
#         R = [obj for obj in recs[imagename] if obj['name'] == classname]
#         bbox = np.array([x['bbox'] for x in R])
#         difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
#         class_recs[imagename] = {'bbox': bbox, 'difficult': difficult,}

#     # read dets
#     with open(detpath.format(classname), 'r') as f:
#         lines = f.readlines()
#         splitlines = [x.strip().split(' ') for x in lines]

#     detect = {}
#     detect['image_ids'] = [x[0] for x in splitlines]
#     detect['confidence'] = np.array([float(x[1]) for x in splitlines])
#     detect['BB'] = np.array([[float(z) for z in x[2:]] for x in splitlines])