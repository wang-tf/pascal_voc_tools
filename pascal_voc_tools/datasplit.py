#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@File: datasplit.py
@Time: 2019-01-11
@Author: ternencewang2015@outlook.com
@Direct: generate txt file in Main
"""

import os
import glob
import random


class DataSplit():
    def __init__(self, root_dir):
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
        if xmls_dir == None:
            xmls_dir = self.xmls_dir
        if images_dir == None:
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

    def split_by_rate(self, test_rate, val_rate=0.0, name_list=None, shuffle=False):
        if name_list is None:
            name_list = self.useful_name_list

        assert test_rate < 1, 'Error: test_rate {} not in range.'.format(test_rate)
        assert len(name_list) > 2, 'Error: name_list length is needed more than 2.'

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
        test_list = name_list[train_number:train_number+test_number]
        if val_number > 0:
            val_list = name_list[-val_number:]
        else:
            val_list = []
        return {'train': train_list, 'val': val_list, 'test': test_list}

    def split_group_by_rate(self, groups, test_rate, val_rate=0.0, shuffle=False):
        """
        Arguments:
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
            split_result = self.split_by_rate(test_rate, val_rate, name_list=groups[prefix], shuffle=shuffle)
            train_list = train_list + split_result['train']
            test_list = test_list + split_result['test']
            val_list = val_list + split_result['val']

        return {'train': train_list, 'val': val_list, 'test': test_list}

    def save(self, split_name_dic, save_dir=None):
        if save_dir is None:
            save_dir = self.txt_dir

        if 'trainval' not in split_name_dic:
            split_name_dic['trainval'] = split_name_dic['train'] + split_name_dic['val']

        for key, val in split_name_dic.items():
            with open(os.path.join(save_dir, key+'.txt'), 'w') as f:
                f.write('\n'.join(val))

def test():
    root_dir = '/home/wangtf/ShareDataset/dataset/RebarDataset/rebar-test/VOC2007'
    prefix_list = ['12', '14', '16', '18', '20', '22', '25', '32']
    test_rate = 0.2
    val_rate = 0.2

    spliter = DataSplit(root_dir)
    groups = spliter.prefix_grouping(prefix_list)
    split_result = spliter.split_group_by_rate(groups, test_rate, val_rate=val_rate)
    spliter.save(split_result)


if __name__ == '__main__':
    test()

