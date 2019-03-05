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
import .cython_nms 


def nms(dets, thresh):
    """Apply classic DPM-style greedy NMS."""
    if dets.shape[0] == 0:
        return []
    return cython_nms.nms(dets, thresh)


def soft_nms(
    dets, sigma=0.5, overlap_thresh=0.3, score_thresh=0.001, method='linear'
):
    """Apply the soft NMS algorithm from https://arxiv.org/abs/1704.04503."""
    if dets.shape[0] == 0:
        return dets, []

    methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
    assert method in methods, 'Unknown soft_nms method: {}'.format(method)

    dets, keep = cython_nms.soft_nms(
        np.ascontiguousarray(dets, dtype=np.float32),
        np.float32(sigma),
        np.float32(overlap_thresh),
        np.float32(score_thresh),
        np.uint8(methods[method])
    )
    return dets, keep
