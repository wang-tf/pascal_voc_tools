#!/usr/bin/env python3

import os
import shutil
import glob
import unittest
import sys
sys.path.append('./')
from pascal_voc_tools import format


class TestFormat(unittest.TestCase):
    test_dev_dir = os.path.join(os.path.dirname(__file__), 'devkit')
    save_dir = os.path.join(os.path.dirname(__file__), 'gen_devkit')

    def setUp(self):
        os.makedirs(self.test_dev_dir)
        self.ann_dir = os.path.join(self.test_dev_dir, 'Annotations')
        self.jpeg_dir = os.path.join(self.test_dev_dir, 'JPEGImages')
        os.makedirs(self.ann_dir)
        os.makedirs(self.jpeg_dir)
        
    def tearDown(self):
        shutil.rmtree(self.test_dev_dir)
        shutil.rmtree(self.save_dir)

    def test_gen_voc_format(self):
        voc_dir = os.path.join(self.save_dir, 'VOC2007')
        voc_dir = format.gen_voc_format(self.ann_dir, self.jpeg_dir, voc_dir)

        # dir exists assert
        save_ann_dir = os.path.join(voc_dir, 'Annotations')
        save_jpeg_dir = os.path.join(voc_dir, 'JPEGImages')
        self.assertTrue(os.path.exists(voc_dir))
        self.assertTrue(os.path.exists(save_ann_dir))
        self.assertTrue(os.path.exists(save_jpeg_dir))

        # xml file names assert
        input_xmls = [os.path.basename(path) for path in glob.glob(os.path.join(self.ann_dir, '*.xml'))]
        output_xmls = [os.path.basename(path) for path in glob.glob(os.path.join(save_ann_dir, '*.xml'))]
        self.assertEqual(input_xmls, output_xmls)

        # jpeg file names assert
        input_jpegs = [os.path.basename(path) for path in glob.glob(os.path.join(self.jpeg_dir, '*.jpg'))]
        output_jpegs = [os.path.basename(path) for path in glob.glob(os.path.join(save_jpeg_dir, '*.jpg'))]
        self.assertEqual(input_jpegs, output_jpegs)

    def test_check_devkit_format(self):
        voc_dir = os.path.join(self.save_dir, 'VOC2007')
        voc_dir = format.gen_voc_format(self.ann_dir, self.jpeg_dir, voc_dir)
        result = format.check_devkit_format(self.save_dir)
        self.assertTrue(result)

    def test_check_voc_format(self):
        voc_dir = os.path.join(self.save_dir, 'VOC2007')
        voc_dir = format.gen_voc_format(self.ann_dir, self.jpeg_dir, voc_dir)
        result = format.check_voc_format(voc_dir)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
