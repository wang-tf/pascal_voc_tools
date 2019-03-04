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


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses  the
    VOC 07 11 point method (default: False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
    return ap


def compute_overlaps(BBGT, bb):
    # compute overlaps
    # intersection
    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
        (BBGT[:, 2] - BBGT[:, 0] + 1.) *
        (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

    return inters / uni


def voc_eval(class_recs, detect, ovthresh=0.5, use_07_metric=False, use_difficult=True):
    """rec, prec, ap = voc_eval(class_recs, detect,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    class_recs: recs dict of a class
        class_recs[image_name]={'bbox': [], 'difficult': []}.
    detect: Path to annotations
        detect={'image_ids':[], bbox': [], 'confidence':[]}.
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # format data
    npos = 0
    for imagename in class_recs.keys():
        assert isinstance(class_recs[imagename]['bbox'], np.ndarray)
        assert isinstance(class_recs[imagename]['difficult'], np.ndarray)

        if use_difficult:
            npos += np.sum(np.logical_not(class_recs[imagename]['difficult']))
        else:
            npos += class_recs[imagename]['difficult'].shape[0]
        det = [False] * class_recs[imagename]['difficult'].shape[0]
        class_recs[imagename]['det'] = det

    image_ids = detect['image_ids']
    confidence = detect['confidence']
    BB = detect['BB']
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
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            overlaps = compute_overlaps(BBGT, bb)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if (use_difficult and not R['difficult'][jmax]) or (not use_difficult):
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


# def parse_rec(filename):
#     """Parse a PASCAL VOC xml file."""
#     tree = ET.parse(filename)
#     objects = []
#     for obj in tree.findall('object'):
#         obj_struct = {}
#         obj_struct['name'] = obj.find('name').text
#         obj_struct['pose'] = obj.find('pose').text
#         obj_struct['truncated'] = int(obj.find('truncated').text)
#         obj_struct['difficult'] = int(obj.find('difficult').text)
#         bbox = obj.find('bndbox')
#         obj_struct['bbox'] = [int(bbox.find('xmin').text),
#                               int(bbox.find('ymin').text),
#                               int(bbox.find('xmax').text),
#                               int(bbox.find('ymax').text)]
#         objects.append(obj_struct)
#     return objects


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