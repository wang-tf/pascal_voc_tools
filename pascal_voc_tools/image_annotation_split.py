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
from .iou import bb_intersection_over_union as iou


class SplitImageAnnotation():
    """Crop subimage and subannotation in data and generate an new dataset."""
    def __init__(self, jpg_dir, xml_dir, save_jpg_dir, save_xml_dir, image_min_side=1000, image_max_side=1333):
        self.jpg_dir = jpg_dir
        self.xml_dir = xml_dir
        self.save_jpg_dir = save_jpg_dir
        self.save_xml_dir = save_xml_dir

        self.image_min_side = image_min_side
        self.image_max_side = image_max_side

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

    def get_split_bboxes(self, width, height, cover_thresh=0.2):
        bboxes = []
        if width >= height:
            x_stride = self.image_max_side
            y_stride = self.image_min_side
        else:
            x_stride = self.image_min_side
            y_stride = self.image_max_side

        def add_row_crop(start_y):
            start_x = 0
            while start_x + x_stride < width:
                bboxes.append([start_x, start_y, start_x+x_stride, start_y+y_stride])
                start_x += int(x_stride * (1-cover_thresh))
            bboxes.append([width-x_stride, start_y, width, start_y+y_stride])

        start_y = 0
        while start_y + y_stride < height:
            add_row_crop(start_y)
            start_y += int(y_stride * (1-cover_thresh))
        add_row_crop(height-y_stride)
        return bboxes

    def split_image(self, image, split_bboxes):
        """Split an image to some subimages.
        Arguments:
            image: ndarray, image data.
            split_bboxes: list, like[[xmin, ymin, xmax, ymax], ]
        Returns:
            images: list, all subimages.
            left_top_strides: list, (left, top) point strides.
            right_down_strides: list, (right, down) point strides.
        """
        subimages = []
        for bbox in split_bboxes:
            subimages.append(image[bbox[1]:bbox[3], bbox[0]:bbox[2]])
        return subimages

    def split_annotations(self, xml_info, split_bboxes, iou_thresh=0.7):
        """Using split_bboxes to split an xml file.
        Arguments:
            xml_info: dict, all info about a xml.
            split_bboxes: list, like [[xmin, ymin, xmax, ymax], ]
        Returns:
            subannotations: list, like [xml_info, ]
        """
        subannotations = []
        for bbox in split_bboxes:
            xmin, ymin, xmax, ymax = bbox

            # init sub xml info
            sub_xml_info = {
                'path': xml_info['path'],
                'filename': xml_info['filename'],
                'folder': xml_info['folder'],
                'width': int(xmax)-int(xmin),
                'height': int(ymax)-int(ymin),
                'depth': xml_info['depth'],
                'database': xml_info['database'],
                'segmented': xml_info['segmented'],
                'objects': []
            }

            for bbox_info in xml_info['objects']:
                ob_xmin, ob_ymin, ob_xmax, ob_ymax = bbox_info['xmin'], bbox_info['ymin'], bbox_info['xmax'], bbox_info['ymax'], 
                if iou([ob_xmin, ob_ymin, ob_xmax, ob_ymax], [xmin, ymin, xmax, ymax]) > iou_thresh:
                    sub_xml_info['objects'].append({
                        'name': bbox_info['name'],
                        'xmin': max(ob_xmin - xmin, 1),
                        'ymin': max(ob_ymin - ymin, 1),
                        'xmax': min(ob_xmax - xmin, xmax-xmin-1),
                        'ymax': min(ob_ymax - ymin, ymax-ymin-1),
                        'pose': bbox_info['pose'],
                        'truncated': bbox_info['truncated'],
                        'difficult': bbox_info['difficult'],
                        })
            subannotations.append(sub_xml_info)

        return subannotations

    def get_annotations(self, xml_path):
        xml_reader = XmlReader(xml_path)
        annotations = xml_reader.get_all_object()
        return annotations

    def split_dir(self):
        images_path, xmls_path = self.match_jpg_xml()
        print('images: {}, xmls: {}'.format(len(images_path), len(xmls_path)))

        for index, (jpg_path, xml_path) in enumerate(zip(images_path, xmls_path)):
            print('{}: split image: {}\t\t\t'.format(index, os.path.basename(jpg_path)), end='\r')
            images, xml_writers = self.split_image_annotations(jpg_path, xml_path)
            for i, (image, xml_writer) in enumerate(zip(images, xml_writers)):
                save_xml_path = os.path.join(self.save_xml_dir, os.path.basename(xml_path).replace('.xml', '_{:0>2d}.xml'.format(i)))
                save_jpg_path = os.path.join(self.save_jpg_dir, os.path.basename(jpg_path).replace('.jpg', '_{:0>2d}.jpg'.format(i)))
                cv2.imwrite(save_jpg_path, image)
                xml_writer.save(save_xml_path, image_parameters={'path': save_jpg_path})

    def split_image_annotations(self, jpg_path, xml_path, cover_thresh=0.2, iou_thresh=1.0, database='Unknown', segmented=0):
        """Split an image and it's annotation file.
        Arguments:
            jpg_path: str, image path.
            xml_path: str, xml path.
            cover_thresh: float, cover rate about each subimage, default=0.2.
            iou_thresh: float, filter annotations which one's iou is smaller than thresh, default=1.0.
            database: str, save xml database name.
            segmented: bool, save xml segmented name.
        """
        image = cv2.imread(jpg_path)
        split_bboxes = self.get_split_bboxes(width=image.shape[1], height=image.shape[0], cover_thresh=cover_thresh)
        subimages = self.split_image(image, split_bboxes)

        annotations = self.get_annotations(xml_path)
        subannotations = self.split_annotations(annotations, split_bboxes)
        
        return subimages, subannotations

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
