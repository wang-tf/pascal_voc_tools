# -*- coding:utf-8 -*-
"""
Some tools for VOC data set dir.
"""
import os
import glob
import cv2
import shutil
import tqdm
import logging

from ._xml_parser import PascalXml
from .image_utils import Image
from .annotation_tools import Annotations
from .images_tools import JPEGImages
logger = logging.getLogger(__name__)
from .tools import bb_intersection_over_union as iou


class Main(object):
    def __init__(self, main_dir):
        self.dir = main_dir
        self.name_list_map = {}

    def __str__(self):
        return f"Main(main_dir={self.dir})"

    def load(self, set_name):
        file_path = os.path.join(self.dir, set_name + '.txt')
        assert os.path.isfile(file_path), file_path

        with open(file_path) as f:
            name_lines = f.read().strip().split('\n')
        logger.info(f'There are {len(name_lines)} names in the file.')

        self.name_list_map[set_name] = name_lines

        return self

    def save(self):
        for set_name in self.name_list_map:
            save_path = os.path.join(self.dir, set_name + '.txt')
            with open(save_path, 'w') as f:
                f.write('\n'.join(self.name_list_map[set_name]))
        return self


class VOCTools(object):
    def __init__(self, root_dir: str):
        self.root_dir = voc_root_dir
        self.year = self.get_year()

        self.annotations = Annotations(os.path.join(root_dir, 'Annotations'))
        self.jpegimages = JPEGImages(os.path.join(root_dir, 'JPEGImages'))
        self.main = Main(os.path.join(root_dir, 'ImageSets/Main'))

    def get_year(self):
        year = None
        if self.voc_root_dir and 'VOC' == self.voc_root_dir[:3]:
            yaer = self.voc_root_dir[3:]
        return year

    def gen_format_dir(self):
        ann_dir = self.annotations.dir
        jpg_dir = self.jpegimages.dir
        main_dir = self.main.dir
        if not os.path.exists(ann_dir):
            os.makedirs(ann_dir)
        if not os.path.exists(jpg_dir):
            os.makedirs(jpg_dir)
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)
        return self

    def resize_by_size(self, width, height):
        """Resize whole dataset by set a fix image size.

        Args:
            width: int, new image width;
            height: intn new image height.
        """
        self.annotations.load()
        save_voc = VOCTools(save_root_dir)
        save_voc.gen_format_dir(save_root_dir)

        print('Resizing dataset ...')
        for xml_path in tqdm.tqdm(self.annotations.ann_list):
            image_path = self.get_image_path_by_xml_path(xml_path)
            xml_name = os.path.basename(xml_path)
            save_xml_path = os.path.join(save_voc.annotations.dir, xml_name)
            image_name = os.path.basename(image_path)
            save_image_path = os.path.join(save_voc.jpegimages.dir, image_name)

            # resize image
            image = Image().load(image_path)
            rate = image.resize_letter_box(width, height)
            image.save(save_image_path)

            # resize annotation and save
            xml_data = PascalXml().load(xml_path).resize_obj_by_rate(rate)
            xml_data.save(save_xml_path)

        return save_voc

    def get_image_path_by_xml_path(self, xml_path):
        """using xml path to inference image path

        Args:
            xml_path: str, xml file path.
        
        Returns:
            image path in current dataset.
        """
        xml_name = os.path.basename(xml_path)
        image_path = os.path.join(self.images_dir,
                                  xml_name.replace('.xml', '.jpg'))
        return image_path

    def match_jpg_xml(self):
        self.jpegimages.load()
        self.annotations.load()

        images_name = [
            os.path.basename(path).split('.')[0]
            for path in self.jpegimages.jpg_list
        ]
        xmls_name = [
            os.path.basename(path).split('.')[0]
            for path in self.annotations.ann_list
        ]

        matched_name = [name for name in xmls_name if name in images_name]

        matched_images_path = [
            os.path.join(self.jpg_dir, name + '.jpg') for name in matched_name
        ]
        matched_xmls_path = [
            os.path.join(self.xml_dir, name + '.xml') for name in matched_name
        ]
        return matched_images_path, matched_xmls_path

    def get_crop_bboxes(self, width, height, cover_thresh=0.2):
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
                bboxes.append(
                    [start_x, start_y, start_x + x_stride, start_y + y_stride])
                start_x += int(x_stride * (1 - cover_thresh))
            bboxes.append(
                [width - x_stride, start_y, width, start_y + y_stride])

        start_y = 0
        while start_y + y_stride < height:
            add_row_crop(start_y)
            start_y += int(y_stride * (1 - cover_thresh))
        add_row_crop(height - y_stride)
        return bboxes

    def crop_data(self,
                  save_root_dir,
                  set_name_list,
                  min_side=1000,
                  max_size=1333):
        new_voc = VOCTools(save_root_dir)
        new_voc.gen_format_dir()

        for set_name in set_name_list:
            self.main.load(set_name)

        new_voc.main.name_list_map = {}
        for set_name in self.main.name_list_map:
            name_list = self.main.name_list_map[set_name]
            logger.info(f'Start crop from {set_name} set')
            new_voc.main.name_list_map[set_name] = []

            for name_id in name_list:
                xml_path = os.path.join(self.annotations.dir, name_id + '.xml')
                jpg_path = os.path.join(self.jpegimages.dir, name_id + '.jpg')

                images, xml_writers = self.crop_image_annotations(
                    jpg_path, xml_path)
                for i, (image,
                        xml_writer) in enumerate(zip(images, xml_writers)):
                    new_name_id = name_id + '_{:0>2d}'
                    new_voc.main.name_list_map[set_name].append(new_name_id)
                    save_xml_path = os.path.join(new_voc.annotations.dir,
                                                 new_name_id + '.xml')
                    save_jpg_path = os.path.join(new_voc.jpegimages.dir,
                                                 new_name_id + '.jpg')
                    image.save(save_jpg_path)
                    xml_writer.folder = new_voc.jpegimages
                    xml_writer.path = save_jpg_path
                    xml_writer.save(save_xml_path)
        
        new_voc.main.save()
        return new_voc

    def crop_image_annotations(self,
                               jpg_path,
                               xml_path,
                               cover_thresh=0.2,
                               iou_thresh=1.0,
                               database='Unknown',
                               segmented=0):
        """Split an image and it's annotation file.
        Arguments:
            jpg_path: str, image path.
            xml_path: str, xml path.
            cover_thresh: float, cover rate about each subimage, default=0.2.
            iou_thresh: float, filter annotations which one's iou is smaller than thresh, default=1.0.
            database: str, save xml database name.
            segmented: bool, save xml segmented name.
        """
        image = Image().load(jpg_path)
        split_bboxes = self.get_crop_bboxes(width=image.width,
                                            height=image.height,
                                            cover_thresh=cover_thresh)
        subimages = image.crop_image(split_bboxes)

        xml_info = PascalXml().load(xml_path)
        subannotations = xml_info.crop_annotations(split_bboxes)

        return subimages, subannotations
