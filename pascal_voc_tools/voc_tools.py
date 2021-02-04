# -*- coding:utf-8 -*-
"""
Some tools for VOC data set dir.
"""
import glob
import logging
import os

import tqdm

from .annotations_tools import Annotations
from .image_tools import ImageWrapper
from .jpegimages_tools import JPEGImages
from .xml_tools import PascalXml

logger = logging.getLogger(__name__)


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
    """a VOC format dataset.

    Attributes:
        root_dir: a voc directory like VOC2007.
    """
    def __init__(self, voc_root_dir: str):
        self.root_dir = voc_root_dir
        self.year = self.get_year()

        self.annotations = Annotations(
            os.path.join(self.root_dir, 'Annotations'))
        self.jpegimages = JPEGImages(os.path.join(self.root_dir, 'JPEGImages'))
        self.main = Main(os.path.join(self.root_dir, 'ImageSets/Main'))

    def __str__(self):
        return f"VOCTools(voc_root_dir={self.root_dir})"

    def get_year(self):
        year = None
        sub_dir = os.path.basename(self.root_dir.rstrip('/'))
        if sub_dir and 'VOC' == sub_dir[:3]:
            year = sub_dir[3:]
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

    def resize_by_size(self, width, height, save_root_dir):
        """Resize whole dataset by set a fix image size.

        Args:
            width: int, new image width.
            height: int, new image height.
            save_root_dir: a new dir to save data.

        Returns:
            new VOCTools have new data.
        """
        self.annotations.load()
        save_voc = VOCTools(save_root_dir)
        save_voc.gen_format_dir()

        logger.info('Resizing dataset ...')
        for xml_path in tqdm.tqdm(self.annotations.ann_list):
            image_path = self.get_image_path_by_xml_path(xml_path)
            xml_name = os.path.basename(xml_path)
            save_xml_path = os.path.join(save_voc.annotations.dir, xml_name)
            image_name = os.path.basename(image_path)
            save_image_path = os.path.join(save_voc.jpegimages.dir, image_name)

            # resize image
            image = ImageWrapper().load(image_path)
            rate, biases = image.resize_letter_box(width, height)
            image.save(save_image_path)

            # resize annotation and save
            xml_data = PascalXml().load(xml_path).resize_obj_by_rate(
                rate, biases)
            # rewrite info
            xml_data.foler = save_voc.jpegimages.dir
            xml_data.filename = image_name
            xml_data.path = save_image_path
            # letter box resize will pad zero
            xml_data.size.width = width
            xml_data.size.height = height
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
        image_path = os.path.join(self.jpegimages.dir,
                                  xml_name.replace('.xml', '.jpg'))
        return image_path

    def match_jpg_xml(self):
        self.jpegimages.load(self.jpegimages.dir)
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

    def get_crop_bboxes(self,
                        width,
                        height,
                        min_side,
                        max_side,
                        cover_thresh=0.2):
        bboxes = []
        if width >= height:
            x_stride = max_side
            y_stride = min_side
        else:
            x_stride = min_side
            y_stride = max_side

        def add_row_crop(start_y):
            start_x = 0
            while start_x + x_stride < width:
                bboxes.append(
                    [start_x, start_y, start_x + x_stride, start_y + y_stride])
                start_x += int(x_stride * (1 - cover_thresh))
            bboxes.append([
                max(0, width - x_stride), start_y, width,
                min(height, start_y + y_stride)
            ])

        start_y = 0
        while start_y + y_stride < height:
            add_row_crop(start_y)
            start_y += int(y_stride * (1 - cover_thresh))
        add_row_crop(max(0, height - y_stride))
        return bboxes

    def crop_data(self,
                  save_root_dir,
                  set_name_list,
                  min_side=1000,
                  max_side=1333,
                  cover_thresh=0.2,
                  iou_thresh=0.7):
        """split image and annotation to some sub data.

        Arguments:
            save_root_dir: a new dir to save new data.
            set_name_list: a list like [train, val].
            min_side: a int of sub image min side.
            max_side: a int of sub image max side.
            cover_thresh: sub image overlap rate.
            iou_thresh: bndbox with sub image overlap rate.

        Returns:
            a new VOCTools including new data.
        """
        new_voc = VOCTools(save_root_dir)
        new_voc.gen_format_dir()

        for set_name in set_name_list:
            self.main.load(set_name)

        new_voc.main.name_list_map = {}
        for set_name in self.main.name_list_map:
            name_list = self.main.name_list_map[set_name]
            logger.info(f'Start crop from {set_name} set')
            new_voc.main.name_list_map[set_name] = []

            for name_id in tqdm.tqdm(name_list):
                xml_path = os.path.join(self.annotations.dir, name_id + '.xml')
                jpg_path = os.path.join(self.jpegimages.dir, name_id + '.jpg')

                images, xml_writers = self.crop_image_annotations(
                    jpg_path,
                    xml_path,
                    min_side,
                    max_side,
                    cover_thresh=cover_thresh,
                    iou_thresh=iou_thresh)
                for i, (image,
                        xml_writer) in enumerate(zip(images, xml_writers)):
                    new_name_id = name_id + '_{:0>2d}'.format(i)
                    new_voc.main.name_list_map[set_name].append(new_name_id)
                    save_xml_path = os.path.join(new_voc.annotations.dir,
                                                 new_name_id + '.xml')
                    save_jpg_path = os.path.join(new_voc.jpegimages.dir,
                                                 new_name_id + '.jpg')
                    image.save(save_jpg_path)
                    xml_writer.folder = new_voc.jpegimages.dir
                    xml_writer.path = save_jpg_path
                    xml_writer.save(save_xml_path)

        new_voc.main.save()
        return new_voc

    def crop_image_annotations(self,
                               jpg_path,
                               xml_path,
                               min_side,
                               max_side,
                               cover_thresh=0.2,
                               iou_thresh=0.7):
        """Split an image and it's annotation file.

        Arguments:
            jpg_path: str, image path.
            xml_path: str, xml path.
            cover_thresh: float, cover rate about each subimage, default=0.2.
            iou_thresh: float, filter annotations which one's iou is smaller
                than thresh, default=1.0.

        Returns:
            subimages: a list of ImageWrapper.
            subannotations: a list of PascalXml
        """
        image = ImageWrapper().load(jpg_path)
        split_bboxes = self.get_crop_bboxes(width=image.width,
                                            height=image.height,
                                            min_side=min_side,
                                            max_side=max_side,
                                            cover_thresh=cover_thresh)
        subimages = image.crop_image(split_bboxes)

        xml_info = PascalXml().load(xml_path)
        subannotations = xml_info.crop_annotations(split_bboxes,
                                                   iou_thresh=iou_thresh)

        return subimages, subannotations

    def check_jpg_xml_match(self):
        """Check matching degree about xml files and jpeg files.
        """
        # arguemnts check
        assert os.path.exists(self.annotations.dir), self.annotations.dir
        assert os.path.exists(self.jpegimages.dir), self.jpegimages.dir

        # get name list
        xml_file_list = glob.glob(os.path.join(self.annotations.dir, '*.xml'))
        jpeg_file_list = glob.glob(os.path.join(self.jpegimages.dir, '*.jpg'))
        xml_name_list = [
            os.path.basename(path).split('.')[0] for path in xml_file_list
        ]
        jpeg_name_list = [
            os.path.basename(path).split('.')[0] for path in jpeg_file_list
        ]

        inter = list(set(xml_name_list).intersection(set(jpeg_name_list)))
        xml_diff = list(set(xml_name_list).difference(set(jpeg_name_list)))
        jpeg_diff = list(set(jpeg_name_list).difference(set(xml_name_list)))

        # print result and return matched list
        print('Find {} xml, {} jpg, matched {}.'.format(
            len(xml_file_list), len(jpeg_file_list), len(inter)))
        if len(xml_diff):
            print("Only have xml file: {}\n{}".format(len(xml_diff), xml_diff))
        if len(jpeg_diff):
            print("Only have jpg file: {}\n{}".format(len(jpeg_diff),
                                                      jpeg_diff))

        return inter
