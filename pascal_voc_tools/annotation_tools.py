#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File : annotation_tools.py
@Time : 2019/03/01 11:14:56
@Author : wangtf
@Version : 1.0
@Desc : None
'''

# here put the import lib
import os
import glob
from .xmlreader import XmlReader


class AnnotationTools():
    def __init__(self, ann_dir, name_set=None):
        self.ann_dir = ann_dir
        self.name_set = name_set
        self.ann_list = self.get_ann_list()

    def get_ann_list(self):
        if self.name_set is None:
            ann_list = glob.glob(os.path.join(self.ann_dir, '*.xml'))
        else:
            name_set_path = os.path.join(self.ann_dir, '../ImageSets/Main/{}.txt'.format(self.name_set))
            assert os.path.exists(name_set_path), 'Can not find file: {}'.format(name_set_path)
            with open(name_set_path) as f:
                name_list = f.read().strip().split('\n')
            ann_list = [os.path.join(self.ann_dir, '{}.xml'.format(name)) for name in name_list]
        return ann_list

    def get_class_dict(self):
        name_dict = {}
        for xml_path in self.ann_list:
            xml_data = XmlReader(xml_path).load()
            xml_name_list = [obj['name'] for obj in xml_data['object']]
            for name in xml_name_list:
                if name not in name_dict:
                    name_dict[name] = {'count': 0, 'included_file': []}
                name_dict[name]['count'] += 1
                file_name = os.path.basename(xml_path)
                if file_name not in name_dict[name]['included_file']:
                    name_dict[name]['included_file'].append(file_name)

        return name_dict
