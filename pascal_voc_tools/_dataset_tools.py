# -*- coding:utf-8 -*-

import os
import datetime
import cv2
import glob
import random
from pascal_voc_tools import XmlParser

TIME_NOW = datetime.datetime.now()


class COCODataset():
    def __init__(self):
        self.image_init_id = 0
        self.annotation_init_id = 0
        self.category_init_id = 0
        self.time_now = datetime.datetime.now()

        self.image_next_new_id = self.image_init_id
        self.annotation_next_new_id = self.annotation_init_id
        self.category_next_new_id = self.category_init_id

        self.image_name_id_map = {}
        self.category_name_id_map = {}

        self.data = {
            "info": self.info(),
            "licenses": [self.license()],
            "images": [],
            "annotations": [],
            "categories": []
        }

    @staticmethod
    def info(self,
             year=TIME_NOW.year,
             version='v1',
             description='dataset',
             contributor='',
             url='',
             date_crtated=TIME_NOW):
        coco_info = {
            'year': year,
            "version": version,
            "description": description,
            "contributor": contributor,
            "url": url,
            "date_created": str(date_crtated)
        }
        return coco_info

    @staticmethod
    def license(self, license_id=1, name='', url=''):
        coco_license = {"id": license_id, "name": name, "url": url}
        return coco_license

    @staticmethod
    def image(self,
              coco_url='',
              date_captured='',
              file_name='',
              flickr_url='',
              image_id='',
              height=0,
              width=0,
              image_license=1):
        coco_image = {
            "coco_url": coco_url,
            "date_captured": date_captured,
            "file_name": file_name,
            "flickr_url": flickr_url,
            "id": image_id,
            "height": height,
            "width": width,
            "license": image_license
        }
        return coco_image

    @staticmethod
    def annotation(self,
                   annotation_id=0,
                   image_id=0,
                   category_id=0,
                   segmentation=[],
                   bbox=[],
                   iscrowd=0):
        coco_annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": bbox[2] * bbox[3],  # width * height
            "bbox": bbox,  # [xmin, ymin, width, height]
            "iscrowd": iscrowd
        }
        return coco_annotation

    @staticmethod
    def category(self, category_id=0, name='', supercategory=''):
        if not supercategory:
            supercategory = name
        coco_category = {
            "id": category_id,
            "name": name,
            "supercategory": supercategory
        }
        return coco_category

    def add_image(self, image_name, image_id, width, height, image_license=1):
        new_image = self.image(file_name=image_name,
                               image_id=image_id,
                               width=width,
                               height=height,
                               image_license=image_license)
        self.data['images'].append(new_image)
        # add info to name id map
        self.image_name_id_map[image_name] = image_id

    def add_annotation(self, annotation_id, image_id, category_id, bbox):
        new_annotation = self.annotation(annotation_id=annotation_id,
                                         image_id=image_id,
                                         category_id=category_id,
                                         bbox=bbox)
        self.data['annotations'].append(new_annotation)

    def add_category(self, category_id, name):
        new_category = self.category(category_id=category_id, name=name)
        self.data['categories'].append(new_category)

    def get_category_id_by_name(self, category_name):
        if category_name not in self.category_name_id_map:
            category_id = self.category_next_new_id
            self.category_next_new_id += 1

            self.add_category(category_id, category_name)

        category_id = self.category_name_id_map[category_name]
        return category_id

    def get_image_id_by_image_path(self, image_path):
        """Finding image id in image_name_id_map. If not find,
        add image and return a new image id.

        Arguments:
            image_path: str, the image to save info.
        Returns:
            image_id in image_name_id_map
        Raises:
            AssertionError: can not find image
        """
        assert os.path.exists(image_path), image_path

        image_name = os.path.basename(image_path)
        if image_name not in self.image_name_id_map:
            # add image to dataset
            image_id = self.image_next_new_id
            self.image_next_new_id += 1

            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            self.add_image(image_name, image_id, width, height)

        image_id = self.image_name_id_map[image_name]

        return image_id
    
    def get_annotation_next_new_id_and_refresh(self):
        annotation_next_new_id = self.annotation_next_new_id
        self.annotaiton_next_new_id += 1
        return annotation_next_new_id


class DarknetDataset():
    def __init__(self, label_file_path):
        self.category_init_id = 0

        self.category_next_new_id = self.category_init_id

        self.label_file_path = label_file_path
        self.category_name_id_map = {}
        self.category_id_name_map = {}
        self._get_category_id_map()

        self.data = {key: [] for key in self.category_name_id_map.keys()}

    def _get_category_id_map(self):
        with open(self.label_file_path) as f:
            lines = f.read().strip().split('\n')
        for line in lines:
            category_name = line.strip()
            if category_name not in self.category_name_id_map:
                category_id = self.category_next_new_id
                self.category_next_new_id += 1

                self.category_name_id_map[category_name] = category_id
                self.category_id_name_map[category_id] = category_name

    def add_annotatoin(self, image_name_id, category_id, confidence, bbox):
        """
        Arguments:
            bbox: [xmin, ymin, width, height]
        """
        category_name = self.category_id_name_map[category_id]
        self.data[category_name].append([image_name_id, confidence] + bbox)


class PascalVOCDataset():
    def __init__(self,
                 voc_root_dir,
                 voc_sets=[('2007', 'train'), ('2007', 'val'),
                           ('2007', 'test')]):
        """
        Args:
            root_dir: str, the directory path including JPEGImages
                      and Annotations;
        """
        self.root_dir = voc_root_dir
        self.voc_sets = voc_sets

        self.sub_root_dirs = []
        for year, main_set in voc_sets:
            sub_root_dir = os.path.join(voc_root_dir, 'VOC{}'.format(year))
            if sub_root_dir not in self.sub_root_dirs:
                self.sub_root_dirs.append(sub_root_dir)

            main_dir = os.path.join(sub_root_dir, 'ImageSets/Main')
            xmls_dir = os.path.join(sub_root_dir, 'Annotations')
            images_dir = os.path.join(sub_root_dir, 'JPEGImages')
            assert os.path.exists(xmls_dir), xmls_dir
            assert os.path.exists(images_dir), images_dir

            if not os.path.exists(main_dir):
                os.makedirs(main_dir)

        self.classes = []

    def get_classes(self, sub_root_dir):
        """ read the file labels.list in voc_root_path

        Args:
            voc_root_path: str, like VOCdevkit.
        """
        # The labels.list file mast be in voc_root_path dir.
        label_path = os.path.join(sub_root_dir, 'labels/classes.txt')
        assert os.path.exists(label_path), label_path

        with open(label_path) as f:
            labels = f.read().strip().split('\n')
        self.classes = [label.strip() for label in labels]
        print('Classes change to: {}'.format(self.classes))

    def xml2label(self, image_id, sub_root_dir):
        from pascal_voc_tools.utils import xyxy2xcycwh
        from pascal_voc_tools.utils import bbox_absolute2relative

        xmls_dir = os.path.join(sub_root_dir, 'Annotations')
        labels_dir = os.path.join(sub_root_dir, 'lables')

        xml_file_path = os.path.join(xmls_dir, '{}.xml'.format(image_id))
        label_save_path = os.path.join(labels_dir, '{}.txt'.format(image_id))

        xml_data = XmlParser().load(xml_file_path)
        width = int(xml_data['size']['width'])
        height = int(xml_data['size']['height'])

        objects = []
        for obj in xml_data['object']:
            difficult = obj['difficult']
            category = obj['name']
            if category not in self.classes or int(difficult) == 1:
                continue
            category_id = self.classes.index(category)
            xmlbox = obj['bndbox']
            b = (float(xmlbox['xmin']), float(xmlbox['ymin']),
                 float(xmlbox['xmax']), float(xmlbox['ymax']))
            bbox_absolute = xyxy2xcycwh(b)
            bbox_relative = bbox_absolute2relative(bbox_absolute,
                                                   width=width,
                                                   height=height)
            objects.append(
                str(category_id) + " " +
                " ".join([str(a) for a in bbox_relative]))

        with open(label_save_path, 'w') as f:
            f.write('\n'.join(objects))

        return objects

    def match_xml_and_jpg(self, xmls_dir, images_dir):
        """Finding corresponding jpg file for xml file in xmls_dir,
        the list of file name will be returned.

        Args:
            xmls_dir: str, the directory path including xmls;
            images_dir: str, the directory path incuding images;

        Returns:
            useful_name_list: str, the list of image name which
                              have corresponding xml file.
        """
        xmls_list = sorted(glob.glob(os.path.join(xmls_dir, '*.xml')))
        useful_name_list = []
        for xml_path in xmls_list:
            name = os.path.basename(xml_path).split('.')[0]
            image_path = os.path.join(images_dir, '{}.jpg'.format(name))
            if os.path.exists(image_path):
                useful_name_list.append(name)

        return useful_name_list

    def prefix_grouping(self, prefix_list, name_list=None):
        """name_list will be splited some groups for which all name are
        started with one prefix

        Args:
            prefix_list: list, the images which have the same string
                         will save in the same list that as the value
                         of the prefix string as the key.
            name_list: list, default is None, whick have all useful
                         name as the whole dataset.
        Returns:
            groups: map, the key is the string in prefix_list, the value
                    is a list that all name in it has corresponding prefix.
        """
        from pascal_voc_tools.utils import prefix_grouping
        name_list = name_list if name_list else self.useful_name_list

        groups = prefix_grouping(prefix_list, name_list)
        return groups

    def split_group_by_rate(self,
                            groups,
                            test_rate,
                            val_rate=0.0,
                            shuffle=False):
        """
        Args:
            groups: dict like {prefix: [name, ]}, grouped name list.
            test_rate: float, the test rate of all data.
            val_rate: float, the val rate of all data, default is 0.0.
            shuffle: bool, default is False.

        Returns:
            result: dict like {'train': [], 'val': [], 'test': []}.
        """
        train_list, val_list, test_list = [], [], []

        for prefix in groups:
            if not groups[prefix]:
                continue
            split_result = self.split_by_rate(test_rate,
                                              val_rate,
                                              name_list=groups[prefix],
                                              shuffle=shuffle)
            train_list = train_list + split_result['train']
            test_list = test_list + split_result['test']
            val_list = val_list + split_result['val']

        return {'train': train_list, 'val': val_list, 'test': test_list}

    def save_main_data(self, split_name_dic, save_dir=None):
        """
        Args:
            split_name_dic: map, the splited result fo dataset.
            save_dir: str, default is None, the path using to save result.
                      if None, the result will saved in Main dir corresponding
                      root dir.
        """
        save_dir = save_dir if save_dir else self.main_dir

        if 'trainval' not in split_name_dic:
            split_name_dic[
                'trainval'] = split_name_dic['train'] + split_name_dic['val']

        for key, val in split_name_dic.items():
            save_path = os.path.join(save_dir, key + '.txt')
            with open(save_path, 'w') as f:
                f.write('\n'.join(val))
            print('INFO: saved {}'.format(save_path))

        return 0


def pascalvoc2json(voc_root_dir, main_set_name='train'):
    xml_dir = os.path.join(voc_root_dir, 'Annotations')
    jpg_dir = os.path.join(voc_root_dir, 'JPEGImages')
    main_set_dir = os.path.join(voc_root_dir, 'ImageSets/Main')

    main_set_path = os.path.join(main_set_dir, main_set_name + '.txt')

    with open(main_set_path) as f:
        lines = f.read().strip().split('\n')
    image_ids = [line.strip() for line in lines]

    coco_dataset = COCODataset()
    json_content = coco_dataset.data

    for name_id in image_ids:
        # load image
        image_path = os.path.join(jpg_dir, name_id + '.jpg')
        image_id = coco_dataset.get_image_id_by_image_path(image_path)

        # load annotation
        xml_file_path = os.path.join(xml_dir, name_id + '.xml')
        xml_data = XmlParser().load(xml_file_path)

        for obj in xml_data['object']:
            category_name = obj['name']
            #
            category_id = coco_dataset.get_category_id_by_name(category_name)

            xmin = float(obj['bndbox']['xmin'])
            ymin = float(obj['bndbox']['ymin'])
            xmax = float(obj['bndbox']['xmax'])
            ymax = float(obj['bndbox']['ymax'])

            annotation_id = coco_dataset.get_annotation_next_new_id_and_refresh()

            coco_dataset.add_annotation(image_id, annotation_id, category_id,
                                        [xmin, ymin, xmax - xmin, ymax - ymin])

    return json_content


def darknet2json(text_file_path):
    pass


def pascalvoc2darknet(voc_root_dir):
    """Convert PascalVOC dataset to darknet dataset.

    Args:
        voc_root_path: str, like VOCdevkit.
    """
    voc_dataset = PascalVOCDataset(voc_root_dir)

    for year, image_set in voc_dataset.voc_sets:
        sub_root_dir = os.path.join(voc_root_dir, 'VOC{}'.format(year))
        voc_dataset.get_classes(sub_root_dir)
        sub_root_label_dir = os.path.join(sub_root_dir, 'labels')
        if not os.path.exists():
            os.makedirs(sub_root_label_dir)
        # like train.txt or test.txt
        main_set_file_path = os.path.join(
            sub_root_dir, 'ImageSets/Main/{}.txt'.format(image_set))

        # pass can not fine file
        if not os.path.exists(main_set_file_path):
            continue

        image_ids = open(main_set_file_path).read().strip().split('\n')
        main_set_save_file_path = os.path.join(voc_root_dir,
                                               '%s_%s.txt' % (year, image_set))
        image_file_list = []
        for image_id in image_ids:
            image_file_path = os.path.abspath(
                os.path.join(sub_root_dir,
                             'JPEGImages/{}.jpg'.format(image_id)))
            image_file_list.append(image_file_path)
            voc_dataset.xml2label(image_id, sub_root_dir)

        with open(main_set_save_file_path, 'w') as list_file:
            list_file.write('\n'.join(image_file_list))
