#!/usr/bin/env python3

import os
import shutil
import glob
import unittest
from pascal_voc_tools import DataScorce
from pascal_voc_tools import PascalXml


class TestPascalXml(unittest.TestCase):

    def test_init(self):
        folder = './tests'
        filename = 'test.xml'
        path = './tests/test.jpg'
        scource = 
        xml = PascalXml(folder)
        self.assertTrue(result)

    def test_check_voc_format(self):
        voc_dir = os.path.join(self.save_dir, 'VOC2007')
        voc_dir = format.gen_voc_format(self.ann_dir, self.jpeg_dir, voc_dir)
        result = format.check_voc_format(voc_dir)
        self.assertTrue(result)
