import os
import unittest
import shutil
from pascal_voc_tools import VOCTools


class TestVOCTools(unittest.TestCase):
    save_dir = os.path.join(os.path.dirname(__file__), 'gen_devkit')

    def setUp(self):
        os.makedirs(self.save_dir)

    def tearDown(self):
        shutil.rmtree(self.save_dir)

    def test_get_year(self):
        voc_dir = os.path.join(self.save_dir, 'VOC2007')

        voc = VOCTools(voc_dir)
        self.assertTrue(voc.year == '2007')

    def test_gen_format_dir(self):
        voc_dir = os.path.join(self.save_dir, 'VOC2007')
        voc = VOCTools(voc_dir).gen_format_dir()

        # dir exists assert
        save_ann_dir = os.path.join(voc_dir, 'Annotations')
        save_jpeg_dir = os.path.join(voc_dir, 'JPEGImages')
        save_main_dir = os.path.join(voc_dir, 'ImageSets/Main')
        self.assertTrue(os.path.exists(voc_dir))
        self.assertTrue(os.path.exists(save_ann_dir))
        self.assertTrue(os.path.exists(save_jpeg_dir))
        self.assertTrue(os.path.exists(save_main_dir))
