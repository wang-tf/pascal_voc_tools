#!/usr/bin/env python3

import os
import unittest
import sys
import glob
from PIL import Image
sys.path.append('./')
from pascal_voc_tools import file


class TestFile(unittest.TestCase):
    voc_dir = os.path.join(os.path.dirname(__file__), 'VOC2007')
    jpeg_dir = os.path.join(voc_dir, 'JPEGImages')
    xmls_dir = os.path.join(voc_dir, 'Annotations')

    def setUp(self):
        os.makedirs(self.jpeg_dir)
        os.makedirs(self.xmls_dir)

    def test_check_image_format(self):
        pass
