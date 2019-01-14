#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@File: image_annotation_split.py
@Author: ternencewang
@Time: 2019-01-14
@Desc: split image and annotation.
"""

import os
import glob
import numpy as np
import cv2

from .xmlreader import XmlReader
from .xmlwriter import XmlWriter


class SplitImageAnnotation():
    """Crop subimage and subannotation in data and generate an new dataset."""
    def __init__(self, jpg_dir, xml_dir, save_jpg_dir, save_xml_dir):
        self.jpg_dir = jpg_dir
        self.xml_dir = xml_dir
        self.save_jpg_dir = save_jpg_dir
        self.save_xml_dir = save_xml_dir

        self.image_min_side = 1000
        self.image_max_side = 1333

    def load_images_path(self):
        images_path = glob.glob(os.path.join(self.jpg_dir, '*.jpg'))
        return images_path

    def load_xmls_path(self):
        xmls_path = glob.glob(os.path.join(self.xml_dir, '*.xml'))
        return xmls_path

    def match_jpg_xml(self):
        images_path = self.load_images_path()
        xmls_path = self.load_xmls_path()
        images_name = [os.path.basename(path).split('.')[0] for path in images_path]
        xmls_name = [os.path.basename(path).split('.')[0] for path in xmls_path]

        matched_name = [name for name in xmls_name if name in images_name]

        matched_images_path = [os.path.join(self.jpg_dir, name+'.jpg') for name in matched_name]
        matched_xmls_path = [os.path.join(self.xml_dir, name+'.xml') for name in matched_name]
        return matched_images_path, matched_xmls_path

    def split_image(self, image, cover_thresh=0.2):
        """Split an image to some subimages.
        Arguments:
            image: ndarray, image data.
            cover_thresh: float, keep cover thresh for every subimages.
        Returns:
            images: list, all subimages.
            left_top_strides: list, (left, top) point strides.
            right_down_strides: list, (right, down) point strides.
        """
        images = []
        left_top_strides = []
        right_down_strides = []
        height, width = image.shape[0:2]
        if width >= height:
            x_stride = self.image_max_side
            y_stride = self.image_min_side
        else:
            x_stride = self.image_min_side
            y_stride = self.image_max_side

        def add_row_crop(start_y):
            start_x = 0
            while start_x + x_stride < width:
                images.append(image[start_y:start_y+y_stride, start_x:start_x+x_stride])
                left_top_strides.append([start_x, start_y])
                right_down_strides.append([start_x+x_stride, start_y+y_stride])
                start_x += int(x_stride * (1-cover_thresh))
            images.append(image[start_y:start_y+y_stride, width-x_stride:])
            left_top_strides.append([width-x_stride, start_y])
            right_down_strides.append([width, start_y+y_stride])

        start_y = 0
        while start_y + y_stride < height:
            add_row_crop(start_y)
            start_y += int(y_stride * (1-cover_thresh))
        add_row_crop(height-y_stride)
        
        return images, left_top_strides, right_down_strides

    def get_annotations(self, xml_path): 
        xml_reader = XmlReader(xml_path)
        annotations = xml_reader.get_all_object()
        return annotations

    def split_dir(self):
        images_path, xmls_path = self.match_jpg_xml()
        print('images: {}, xmls: {}'.format(len(images_path), len(xmls_path)))

        for index, (jpg_path, xml_path) in enumerate(zip(images_path, xmls_path)):
            print('{}: split image: {}\t\t\t'.format(index, os.path.basename(jpg_path)), end='\r')
            self.split_image_annotations(jpg_path, xml_path)
    
    def split_image_annotations(self, jpg_path, xml_path, iou_thresh=1.0, database='Unknown', segmented=0):
        image = cv2.imread(jpg_path)
        annotations = self.get_annotations(xml_path)
        images, left_top_strides, right_down_strides = self.split_image(image)
        
        names = [i['name'] for i in annotations]
        bboxes = np.array([i['bbox'] for i in annotations])
        pose = [i['pose'] for i in annotations]
        truncated = [i['truncated'] for i in annotations]
        difficult = [i['difficult'] for i in annotations]

        # count iou
        for index in range(len(images)):
            if annotations:
                inside_bboxes = bboxes.copy()

                area = (inside_bboxes[:, 2] - inside_bboxes[:, 0]) * (inside_bboxes[:, 3] - inside_bboxes[:, 1])

                inside_bboxes[inside_bboxes[:, 0] < left_top_strides[index][0], 0] = left_top_strides[index][0] + 1
                inside_bboxes[inside_bboxes[:, 1] < left_top_strides[index][1], 1] = left_top_strides[index][1] + 1
                inside_bboxes[inside_bboxes[:, 2] > right_down_strides[index][0], 2] = right_down_strides[index][0] - 1
                inside_bboxes[inside_bboxes[:, 3] > right_down_strides[index][1], 3] = right_down_strides[index][1] - 1

                useful_index_x = inside_bboxes[:, 2] > inside_bboxes[:, 0]
                useful_index_y = inside_bboxes[:, 3] > inside_bboxes[:, 1]
                useful_index = np.logical_and(useful_index_x, useful_index_y)
                
                intersection = (inside_bboxes[:, 2] - inside_bboxes[:, 0]) * (inside_bboxes[:, 3] - inside_bboxes[:, 1])

                coinside = intersection / area

                useful_index_area = coinside >= iou_thresh

                useful_index = np.logical_and(useful_index, useful_index_area)
            else:
                useful_index = []

            save_xml_path = os.path.join(self.save_xml_dir, os.path.basename(xml_path).replace('.xml', '_{:0>2d}.xml'.format(index)))
            save_jpg_path = os.path.join(self.save_jpg_dir, os.path.basename(jpg_path).replace('.jpg', '_{:0>2d}.jpg'.format(index)))
            cv2.imwrite(save_jpg_path, images[index])
            xml_writer = XmlWriter(save_jpg_path, images[index].shape[1], images[index].shape[0], depth=3, database=database, segmented=segmented)
            for i, use in enumerate(useful_index):
                if use:
                    xmin, ymin, xmax, ymax = inside_bboxes[i]
                    xmin -= left_top_strides[index][0]
                    ymin -= left_top_strides[index][1]
                    xmax -= left_top_strides[index][0]
                    ymax -= left_top_strides[index][1]
                    xml_writer.add_object(name=names[i], xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, pose=pose[i], truncated=truncated[i], difficult=difficult[i])
            xml_writer.save(save_xml_path)

    def copy_split_data_name(self):
        root_dir = os.path.join(self.xml_dir, '../ImageSets/Main')
        train_path = os.path.join(root_dir, 'train.txt')
        val_path = os.path.join(root_dir, 'val.txt')
        test_path = os.path.join(root_dir, 'test.txt')

        assert os.path.exists(train_path), train_path
        assert os.path.exists(val_path), val_path
        assert os.path.exists(test_path), test_path

        save_path = os.path.join(self.save_xml_dir, '../ImageSets/Main')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for txt_path in [train_path, val_path, test_path]:
            with open(txt_path) as file:
                name_list = file.readlines()
                new_path_list = []
                for name in name_list:
                    new_path_list += glob.glob(os.path.join(self.save_xml_dir, name.strip()+'*.xml'))
                new_name_list = [os.path.basename(name).split('.')[0] for name in new_path_list]
                with open(os.path.join(save_path, os.path.basename(txt_path)), 'w') as f:
                    f.write('\n'.join(new_name_list))


def test():
    root_dir = '/home/wangtf/ShareDataset/dataset/RebarDataset/VOCdevkit_rebar_v9_0-12-14-16-18-20-22-25-32'
    jpg_dir = os.path.join(root_dir, 'VOC2007/JPEGImages')
    xml_dir = os.path.join(root_dir, 'VOC2007/Annotations')
    assert os.path.exists(xml_dir), xml_dir

    save_root_dir = '{}_split_test/VOC2007'.format(root_dir)
    save_jpg_dir = os.path.join(save_root_dir, 'JPEGImages')
    save_xml_dir = os.path.join(save_root_dir, 'Annotations')
    if not os.path.exists(save_jpg_dir):
        os.makedirs(save_jpg_dir)
    if not os.path.exists(save_xml_dir):
        os.makedirs(save_xml_dir)
    
    spliter = SplitImageAnnotation(jpg_dir, xml_dir, save_jpg_dir, save_xml_dir)
    spliter.split_dir()
    spliter.copy_split_data_name()


if __name__ == '__main__':
    test()
