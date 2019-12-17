#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os
import sys
import glob
import argparse
import cv2
from pascal_voc_tools import XmlParser


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('comp')
    parser.add_argument('--jpeg_dir', default='/diskb/GlodonDataset/SteelPipe/SteelPipe-20191025/VOCdevkit-testonly-1000/VOC2007/JPEGImages')
    parser.add_argument('--label_path', default='/diskb/GlodonDataset/SteelPipe/SteelPipe-20191025/VOCdevkit-testonly-1000/labels.list')
    parser.add_argument('--confthresh', default=0.5, type=float)
    args = parser.parse_args()
    return args

    
def get_labels(label_path):
    assert os.path.exists(label_path), label_path

    with open(label_path) as f:
        names = f.read().strip().split('\n')
    id_name_map = {k:v for k,v in enumerate(names)}
    return id_name_map


def int_fstr(x):
     return int(float(x))


def decode_comp_result(comp_result_path):
    assert os.path.exists(comp_result_path), comp_result_path

    with open(comp_result_path) as f:
        data = f.read().strip().split('\n')
    name_bbox_map = {}
    for line in data:
        # pass no obj image
        if len(line.split(' ')) == 1:
            name = line.split(' ')[0]
            name_bbox_map[name] = []
            continue
        name, score, xmin, ymin, xmax, ymax = line.split(' ')
        bbox = list(map(int_fstr, [xmin, ymin, xmax, ymax]))
        bbox += [float(score)]
        if name not in name_bbox_map:
            name_bbox_map[name] = []
        name_bbox_map[name].append(bbox)
    return name_bbox_map


def save2xml(name, bbox, score_thresh, save_dir, width, height):
    parser = XmlParser()
    parser.set_head(path=name, width=width, height=height)
    for b in bbox:
        if b[4] < score_thresh:
            continue
        parser.add_object(name='SteelPipe', xmin=b[0], ymin=b[1], xmax=b[2], ymax=b[3])
    save_path = os.path.join(save_dir, name+'.xml')
    parser.save(save_path)


def save2xmlandshow(comp_result_path, jpeg_dir, label_path, score_thresh, save=False):
    save_dir = os.path.dirname(comp_result_path)

    id_name_map = get_labels(label_path)
    name_bbox_map = decode_comp_result(comp_result_path)

    for name, bbox in name_bbox_map.items():
        image_path = os.path.join(jpeg_dir, name+'.jpg')
        assert os.path.exists(image_path), image_path

        image = cv2.imread(image_path)
        # save detect object to xml
        save2xml(name, bbox, score_thresh, save_dir, image.shape[1], image.shape[0])
        
        if not save:
            continue

        for b in bbox:
            if b[4] < score_thresh:
                continue
            color = (int(255*(1-b[4])), 255, int(255*(1-b[4])))
            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, 1)
            cv2.putText(image, '{:.3}'.format(b[4]), (b[0], b[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
        #cv2.imshow('pipe', image)
        #save_path = os.path.join(save_dir, name+'.jpg')
        #cv2.imwrite(save_path, image)

        #if cv2.waitKey(1) == ord('q'):
        #    sys.exit()

def to_class_bbox(bbox):
    for b in bbox:
        if b[4] < score_thresh:
            continue
        xmin = b[0]
        ymin = b[1]
        xmax = b[2]
        ymax = b[3]


if __name__ == '__main__':
    args = get_args()
    comp_result_path = args.comp
    jpeg_dir = args.jpeg_dir
    label_path = args.label_path
    score_thresh = args.confthresh

    save2xmlandshow(comp_result_path, jpeg_dir, label_path, score_thresh)

