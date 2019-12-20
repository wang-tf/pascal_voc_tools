#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File : nms.py
@Time : 2019/03/05 10:35:46
@Author : wangtf
@Version : 1.0
@Desc : None
'''

# here put the import lib
import numpy as np
# import .cython_nms


# def nms(dets, thresh):
#     """Apply classic DPM-style greedy NMS."""
#     if dets.shape[0] == 0:
#         return []
#     return cython_nms.nms(dets, thresh)


# def soft_nms(
#     dets, sigma=0.5, overlap_thresh=0.3, score_thresh=0.001, method='linear'
# ):
#     """Apply the soft NMS algorithm from https://arxiv.org/abs/1704.04503."""
#     if dets.shape[0] == 0:
#         return dets, []

#     methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
#     assert method in methods, 'Unknown soft_nms method: {}'.format(method)

#     dets, keep = cython_nms.soft_nms(
#         np.ascontiguousarray(dets, dtype=np.float32),
#         np.float32(sigma),
#         np.float32(overlap_thresh),
#         np.float32(score_thresh),
#         np.uint8(methods[method])
#     )
#     return dets, keep


def nms(dets, thresh):
    """Apply classic DPM-style greedy NMS."""
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = np.maximum(ix1, x1[j])
            yy1 = np.maximum(iy1, y1[j])
            xx2 = np.minimum(ix2, x2[j])
            yy2 = np.minimum(iy2, y2[j])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return np.where(suppressed == 0)[0]


def soft_nms(boxes_in, sigma=0.5, Nt=0.3, threshold=0.001, method=0):
    """methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
    """
    if boxes_in.shape[0] == 0:
        return boxes_in, []

    boxes = boxes_in.copy()
    N = boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0
    inds = np.arange(N)

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]
        ti = inds[i]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        boxes[i, 4] = boxes[maxpos, 4]
        inds[i] = inds[maxpos]

        # swap ith box with position of max box
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        boxes[maxpos, 4] = ts
        inds[maxpos] = ti

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below
        # threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (np.minimum(tx2, x2) - np.maximum(tx1, x1) + 1)
            if iw > 0:
                ih = (np.minimum(ty2, y2) - np.maximum(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) *
                               (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box

                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight*boxes[pos, 4]

                    # if box score falls below threshold, discard the box by
                    # swapping with last box update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos, 0] = boxes[N-1, 0]
                        boxes[pos, 1] = boxes[N-1, 1]
                        boxes[pos, 2] = boxes[N-1, 2]
                        boxes[pos, 3] = boxes[N-1, 3]
                        boxes[pos, 4] = boxes[N-1, 4]
                        inds[pos] = inds[N-1]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    return boxes[:N], inds[:N]

