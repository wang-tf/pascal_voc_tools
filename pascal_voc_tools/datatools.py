# -*- coding:utf-8 -*-
"""
@File: datasplit.py
@Time: 2019-12-02
@Author: ternencewang2015@outlook.com
@Description:
This script used to split pascal voc format dataset and generate
train.txt, val.txt and test.txt.
"""

import os
import glob
import random
import xml.etree.ElementTree as ET


class DataSplit():
    """
    This script used to split pascal voc format dataset and generate
    train.txt, val.txt and test.txt.
    """
    def __init__(self, root_dir):
        """
        Args:
            root_dir: str, the directory path including JPEGImages
                      and Annotations;
        """
        self.root_dir = root_dir

        self.txt_dir = os.path.join(self.root_dir, 'ImageSets/Main')
        self.xmls_dir = os.path.join(self.root_dir, 'Annotations')
        self.images_dir = os.path.join(self.root_dir, 'JPEGImages')
        assert os.path.exists(self.xmls_dir), self.xmls_dir
        assert os.path.exists(self.images_dir), self.images_dir

        if not os.path.exists(self.txt_dir):
            os.makedirs(self.txt_dir)

        self.useful_name_list = self.match_xml_and_jpg()

    def match_xml_and_jpg(self, xmls_dir=None, images_dir=None):
        """Finding corresponding jpg file for xml file in xmls_dir,
        the list of file name will be returned.

        Args:
            xmls_dir: str, the directory path including xmls;
            images_dir: str, the directory path incuding images;

        Returns:
            useful_name_list: str, the list of image name which
                              have corresponding xml file.
        """
        if not xmls_dir:
            xmls_dir = self.xmls_dir
        if not images_dir:
            images_dir = self.images_dir

        xmls_list = sorted(glob.glob(os.path.join(xmls_dir, '*.xml')))
        useful_name_list = []
        for xml_path in xmls_list:
            name = os.path.basename(xml_path).split('.')[0]
            image_path = os.path.join(images_dir, '{}.jpg'.format(name))
            if os.path.exists(image_path):
                useful_name_list.append(name)

        return useful_name_list

    def prefix_grouping(self, prefix_list, name_list=None):
        """
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
        if name_list is None:
            name_list = self.useful_name_list

        groups = {}
        for prefix in prefix_list:
            groups[prefix] = []

        for name in name_list:
            for prefix in prefix_list:
                if name.startswith(prefix):
                    groups[prefix].append(name)

        return groups

    def split_by_rate(self,
                      test_rate,
                      val_rate=0.0,
                      name_list=None,
                      shuffle=False):
        """
        Args:
            test_rate: float, the test data rate for all data;
            val_rate: float, default is 0.0, the val data rate for all data;
            name_list: list, all useful name in this data.
            shuffle: bool, default is False, The name_list will be shuffled
                     if it is true.

        Returns:
            splited_data: map, the key is str in ['train', 'val', 'test'],
                          the value is the list of names. 
        """
        if name_list is None:
            name_list = self.useful_name_list

        assert test_rate < 1, 'Error: test_rate {} not in range.'.format(
            test_rate)
        assert len(
            name_list) > 2, 'Error: name_list length is needed more than 2.'

        if shuffle:
            random.shuffle(name_list)

        test_number = int(test_rate * len(name_list))
        test_number = test_number if test_number > 0 else 1

        if val_rate > 0:
            val_number = int(val_rate * len(name_list))
            val_number = val_number if val_number > 0 else 1
        else:
            val_number = 0

        train_number = len(name_list) - test_number - val_number
        assert train_number > 0, 'Error: train_number is needed more than 0.'

        train_list = name_list[0:train_number]
        test_list = name_list[train_number:train_number + test_number]
        if val_number > 0:
            val_list = name_list[-val_number:]
        else:
            val_list = []
        return {'train': train_list, 'val': val_list, 'test': test_list}

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

    def save(self, split_name_dic, save_dir=None):
        """
        Args:
            split_name_dic: map, the splited result fo dataset.
            save_dir: str, default is None, the path using to save result.
                      if None, the result will saved in Main dir corresponding
                      root dir.
        """
        if save_dir is None:
            save_dir = self.txt_dir

        if 'trainval' not in split_name_dic:
            split_name_dic[
                'trainval'] = split_name_dic['train'] + split_name_dic['val']

        for key, val in split_name_dic.items():
            with open(os.path.join(save_dir, key + '.txt'), 'w') as f:
                f.write('\n'.join(val))

        return 0


class DarknetDataset():
    """Convert PascalVOC dataset to darknet dataset.

    Attributes:
        voc_sets: list, like [('2007', 'train')]

    """
    def __init__(self, voc_sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]):
        self.voc_sets = voc_sets
        self.classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
            "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
            "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]

    def get_classes(self, voc_root_path):
        """ read the file labels.list in voc_root_path

        Args:
            voc_root_path: str, like VOCdevkit.
        """
        # The labels.list file mast be in voc_root_path dir.
        label_path = os.path.join(voc_root_path, 'labels.list')
        assert os.path.exists(label_path), label_path

        with open(label_path) as f:
            labels = f.read().strip().split('\n')
            labels = [label.strip() for label in labels]
        self.classes = labels
        print('Classes change to: {}'.format(self.classes))
        return self.classes

    def convert(self, size, box):
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[1]) / 2.0 - 1
        y = (box[2] + box[3]) / 2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    def convert_annotation(self, year, image_id, voc_root_path):
        in_file = open(
            os.path.join(voc_root_path,
                         'VOC%s/Annotations/%s.xml' % (year, image_id)))
        out_file = open(
            os.path.join(voc_root_path,
                         'VOC%s/labels/%s.txt' % (year, image_id)), 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.classes or int(difficult) == 1:
                continue
            cls_id = self.classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text),
                 float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = self.convert((w, h), b)
            out_file.write(
                str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    def voc2(self, voc_root_path):
        """Convert PascalVOC dataset to darknet dataset.

        Args:
            voc_root_path: str, like VOCdevkit.
        """
        # change labels
        self.get_classes(voc_root_path)

        for year, image_set in self.voc_sets:
            if not os.path.exists(
                    os.path.join(voc_root_path, 'VOC%s/labels/' % (year))):
                os.makedirs(
                    os.path.join(voc_root_path, 'VOC%s/labels/' % (year)))
            if not os.path.exists(
                    os.path.join(
                        voc_root_path, 'VOC%s/ImageSets/Main/%s.txt' %
                        (year, image_set))):
                continue

            image_ids = open(
                os.path.join(voc_root_path, 'VOC%s/ImageSets/Main/%s.txt' %
                             (year, image_set))).read().strip().split()
            with open(
                    os.path.join(voc_root_path,
                                 '%s_%s.txt' % (year, image_set)),
                    'w') as list_file:
                image_file_list = []
                for image_id in image_ids:
                    image_file_path = os.path.abspath(
                        os.path.join(
                            voc_root_path,
                            'VOC{}/JPEGImages/{}.jpg'.format(year, image_id)))
                    image_file_list.append(image_file_path)
                    self.convert_annotation(year, image_id, voc_root_path)
                list_file.write('\n'.join(image_file_list))

        #os.system("cat {0}/2007_train.txt {0}/2007_val.txt > {0}/train.txt".format(voc_root_path))
        #os.system("cat {0}/2007_train.txt {0}/2007_val.txt {0}/2007_test.txt > {0}/train.all.txt".format(voc_root_path))


def test():
    root_dir = '/home/wangtf/ShareDataset/dataset/RebarDataset/rebar-test/VOC2007'
    prefix_list = ['12', '14', '16', '18', '20', '22', '25', '32']
    test_rate = 0.2
    val_rate = 0.2

    spliter = DataSplit(root_dir)
    groups = spliter.prefix_grouping(prefix_list)
    split_result = spliter.split_group_by_rate(groups,
                                               test_rate,
                                               val_rate=val_rate)
    spliter.save(split_result)


if __name__ == '__main__':
    test()
