#!/usr/bin/env python3

import os
import shutil
import glob
import unittest
from pascal_voc_tools._format import Format


class TestFormat(unittest.TestCase):
    test_dev_dir = os.path.join(os.path.dirname(__file__), 'devkit')
    save_dir = os.path.join(os.path.dirname(__file__), 'gen_devkit/VOC2007')

    def setUp(self):
        if not os.path.exists(self.test_dev_dir):
            os.makedirs(self.test_dev_dir)
        self.ann_dir = os.path.join(self.test_dev_dir, 'Annotations')
        self.jpeg_dir = os.path.join(self.test_dev_dir, 'JPEGImages')
        if not os.path.exists(self.ann_dir):
            os.makedirs(self.ann_dir)
        if not os.path.exists(self.jpeg_dir):
            os.makedirs(self.jpeg_dir)
        
    # def tearDown(self):
    #     shutil.rmtree(self.test_dev_dir)
    #     if os.path.exists(self.save_dir):
    #         shutil.rmtree(self.save_dir)

    def test_gen_voc_format(self):
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        f_func = Format()
        self.save_dir = f_func.gen_voc_format(self.ann_dir, self.jpeg_dir, self.save_dir)

        # dir exists assert
        self.save_ann_dir = os.path.join(self.save_dir, 'Annotations')
        self.save_jpeg_dir = os.path.join(self.save_dir, 'JPEGImages')
        self.assertTrue(os.path.exists(self.save_dir))
        self.assertTrue(os.path.exists(self.save_ann_dir))
        self.assertTrue(os.path.exists(self.save_jpeg_dir))

        # xml file names assert
        input_xmls = [os.path.basename(path) for path in glob.glob(os.path.join(self.ann_dir, '*.xml'))]
        output_xmls = [os.path.basename(path) for path in glob.glob(os.path.join(self.save_ann_dir, '*.xml'))]
        self.assertEqual(input_xmls, output_xmls)

        # jpeg file names assert
        input_jpegs = [os.path.basename(path) for path in glob.glob(os.path.join(self.jpeg_dir, '*.jpg'))]
        output_jpegs = [os.path.basename(path) for path in glob.glob(os.path.join(self.save_jpeg_dir, '*.jpg'))]
        self.assertEqual(input_jpegs, output_jpegs)

    def test_check_devkit_format(self):
        f_func = Format()
        result = f_func.check_devkit_format(os.path.join(self.save_dir, '..'))
        self.assertTrue(result)

    def test_check_voc_format(self):
        f_func = Format()
        result = f_func.check_voc_format(self.save_dir)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
