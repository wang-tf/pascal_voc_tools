#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@File: datasetresize.py
@Time: 2019-01-10 19:05:00
@Author: wangtf
@Version: 1.0
@Dest: resize the whole dataset
"""


import os
import glob
import cv2
import shutil
import tqdm

from .xmlreader import XmlReader


class DatasetResize():
    def __init__(self, root_voc_dir, save_root_dir=None):
        """
        Arguments:
            root_voc_dir: str, the path of pascal voc dir include VOC2007.
            save_root_dir: str, the path of save path, default path is input path.
        """
        self.root_dir = root_voc_dir
        self.save_root_dir = self.root_dir if save_root_dir is None else save_root_dir

        self.annotations_dir = os.path.join(self.root_dir, 'Annotations')
        self.images_dir = os.path.join(self.root_dir, 'JPEGImages')

        assert os.path.exists(self.annotations_dir), self.annotations_dir
        assert os.path.exists(self.images_dir), self.images_dir

        self.save_annotations_dir = os.path.join(self.save_root_dir, 'Annotations')
        self.save_images_dir = os.path.join(self.save_root_dir, 'JPEGImages')
        if not os.path.exists(self.save_annotations_dir):
            os.makedirs(self.save_annotations_dir)
        if not os.path.exists(self.save_images_dir):
            os.makedirs(self.save_images_dir)

    def get_annotations(self):
        annotations_file_list = glob.glob(os.path.join(self.annotations_dir, '*.xml'))
        return annotations_file_list

    def resize_tuple_by_rate(self, rate, image_path, xml_path, save_image_path=None, save_xml_path=None):
        """Resize a image and coresponding xml
        Arguments:
            rate: float or int, scale size.
            image_path: str, the path of image.
            xml_path: str, the path of xml file.
            save_image_path: str, image save path, default is image_path.
            save_xml_path: str, xml save path, default is xml_path.
        """
        assert os.path.exists(image_path), image_path
        assert os.path.exists(xml_path), xml_path

        save_image_path = image_path if save_image_path is None else save_image_path
        save_xml_path = xml_path if save_xml_path is None else save_xml_path

        # resize image and save
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, None, fx=rate, fy=rate)
        cv2.imwrite(save_image_path, image_resized)

        # resize annotation and save
        xml_file = XmlReader(xml_path)
        xml_file.set_object_bndbox(rate, save_path=save_xml_path)
        return 1

    def resize_tuple_by_min_size(self, min_size, image_path, xml_path, save_image_path=None, save_xml_path=None):
        assert os.path.exists(image_path), image_path
        assert os.path.exists(xml_path), xml_path

        save_image_path = image_path if save_image_path is None else save_image_path
        save_xml_path = xml_path if save_xml_path is None else save_xml_path

        image = cv2.imread(image_path)
        height, width = image.shape[0:2]
        rate = min_size / min(height, width)
        new_height = int(height * rate)
        new_width = int(width * rate)

        self.resize_tuple_by_rate(rate, image_path, xml_path, save_image_path, save_xml_path)

    def resize_dataset_by_rate(self, rate):
        """Resize the whole dataset
        Arguments:
            rate: float or int, scale size.
        """
        annotations_file_list = self.get_annotations()
        
        print('Resizing dataset ...')
        for xml_path in tqdm.tqdm(annotations_file_list):
            image_path = self.get_image_path_by_xml_path(xml_path)
            save_xml_path = self.get_save_path(xml_path)
            save_image_path = self.get_save_path(image_path)

            self.resize_tuple_by_rate(rate, image_path, xml_path, save_image_path, save_xml_path)

    def resize_dataset_by_min_size(self, min_size):
        annotations_file_list = self.get_annotations()
    
        print('Resizing dataset ...')
        for xml_path in tqdm.tqdm(annotations_file_list):
            image_path = self.get_image_path_by_xml_path(xml_path)
            save_xml_path = self.get_save_path(xml_path)
            save_image_path = self.get_save_path(image_path)\

            self.resize_tuple_by_min_size(min_size, image_path, xml_path, save_image_path, save_xml_path)

    def get_save_path(self, path):
        name = os.path.basename(path)
        save_dir = self.save_annotations_dir if name[-4:] == '.xml' else self.save_images_dir
        save_path = os.path.join(save_dir, name)
        return save_path
        
    def get_image_path_by_xml_path(self, xml_path):
        xml_name = os.path.basename(xml_path)
        image_path = os.path.join(self.images_dir, xml_name.replace('.xml', '.jpg'))
        return image_path

    def copy_imagesets(self, imagesets_dir=None):
        """Copy some text file in Main dir from root dir to save dir
        Arguments:
            images_dir: str, the path of ImageSets/Main.
        """
        if imagesets_dir is None:
            imagesets_dir = os.path.join(self.root_dir, 'ImageSets/Main')

        # make save dir        
        save_dir = os.path.join(self.save_root_dir, 'ImageSets/Main')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for file_name in ['train.txt', 'val.txt', 'trainval.txt', 'test.txt']:
            file_path = os.path.join(imagesets_dir, file_name)
            if os.path.exists(file_path):
                shutil.copy2(file_path, save_dir)
            else:
                print('Can not find path: {}'.format(file_path))


def test():
    root_dir = '/home/wangtf/ShareDataset/dataset/RebarDataset/rebar-20181229-black/VOC2007'
    save_root_dir = '/home/wangtf/ShareDataset/dataset/RebarDataset/rebar-test/VOC2007'
    min_size = 2000
    resizer = DatasetResize(root_dir, save_root_dir)
    resizer.resize_dataset_by_min_size(min_size)
    resizer.copy_imagesets()


if __name__ == '__main__':
    test()

