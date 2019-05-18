#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@File: xmlwriter.py
@Time: 2019-01-14
@Author: ternencewang
@Direc: write a xml file about pascal voc annotation.
<annotation>
    <folder>{{ folder }}</folder>
    <filename>{{ filename }}</filename>
    <path>{{ path }}</path>
    <source>
        <database>{{ database }}</database>
    </source>
    <size>
        <width>{{ size.width }}</width>
        <height>{{ size.height }}</height>
        <depth>{{ size.depth }}</depth>
    </size>
    <segmented>{{ segmented }}</segmented>{% for obj in object %}
    <object>
        <name>{{ obj.name }}</name>
        <pose>{{ obj.pose }}</pose>
        <truncated>{{ obj.truncated }}</truncated>
        <difficult>{{ obj.difficult }}</difficult>
        <bndbox>
            <xmin>{{ obj.bndbox.xmin }}</xmin>
            <ymin>{{ obj.bndbox.ymin }}</ymin>
            <xmax>{{ obj.bndbox.xmax }}</xmax>
            <ymax>{{ obj.bndbox.ymax }}</ymax>
        </bndbox>
    </object>{% endfor %}
</annotation>
"""
import os
import cv2
from jinja2 import Environment, PackageLoader


class XmlWriter():
    """Write a xml file about pascal voc annotation."""
    def __init__(self, path='', width=0, height=0, depth=3, database='Unknown', segmented=0):
        """Generate a xml file
        Arguments:
            path: str, arg in xml about image.
            width: int or str, image width.
            height: int or str, image height.
            depth: int or str, image channle.
            database: str, default='Unknown'.
            segmented: int or str, default=0.
        """
        environment = Environment(loader=PackageLoader('pascal_voc_tools', 'templates'), keep_trailing_newline=True)
        self.annotation_template = environment.get_template('annotation.xml')

        abspath = os.path.abspath(path)
        try:
            assert os.path.exists(abspath)
            image = cv2.imread(abspath)
            height = image.shape[0]
            width = image.shape[1]
            depth = image.shape[3] if len(image.shape) == 3 else 1
        except AssertionError:
            print('Warrning: can not fine path: {}'.format(abspath))

        self.template_parameters = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'size': {'width': width, 'height': height, 'depth': depth, },
            'source': {'database': database, },
            'segmented': segmented,
            'object': []
        }

    def add_object(self, name, xmin, ymin, xmax, ymax, pose='Unspecified', truncated=0, difficult=0):
        """add an object info
        Arguments:
            name: str, class name.
            xmin: int, left.
            ymin: int, top.
            xmax: int, right.
            ymax: int, bottom.
            pose: str, default is 'Unspecified'.
            truncated: str, default is 0.
            difficult: int, default is 0.
        """
        self.template_parameters['object'].append({
            'name': name,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
            'bndbox': {
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
            },
        })

    def save(self, annotation_path, image_parameters=None):
        """Write a xml file to save info.
        Arguments:
            annotation_path: str, the path of xml to save.
        """
        if image_parameters is not None:
            self.check_format(image_parameters)
            for k, v in image_parameters.items():
                self.template_parameters[k] = v

        with open(annotation_path, 'w') as xml_file:
            content = self.annotation_template.render(**self.template_parameters)
            xml_file.write(content)

    def check_format(self, template_parameters):
        for key in ['path', 'filename', 'folder', 'size', 'source', 'segmented', 'object']:
            assert key in template_parameters, "Can not find the key: {}".format(key)
        for key in ['width', 'height', 'depth']:
            assert key in template_parameters['size'], "Can not find the key in size: {}".format(key)
        
        assert isinstance(template_parameters['object'], list)
        for one_obj in template_parameters['object']:
            for key in ['name', 'pose', 'truncated', 'difficult', 'bndbox']:
                assert key in one_obj, "Can not find the key in object: {}".format(key)
                for key in ['xmin', 'ymin','xmax', 'ymax']:
                    assert key in one_obj['bndbox'], "Can not find the key in bndbox: {}".format(key)
