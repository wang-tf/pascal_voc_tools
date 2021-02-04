import os
import unittest

from pascal_voc_tools import DataSource, ImageSize, PascalXml


class TestPascalXml(unittest.TestCase):
    save_path = './tests/test.xml'

    def tearDown(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)

    def test_init(self):
        folder = './tests'
        filename = 'test.xml'
        path = './tests/test.jpg'
        source = DataSource('dataset')
        size = ImageSize(width=10, height=10, depth=3)
        segmented = 0
        xml = PascalXml(folder, filename, path, source, size, segmented)
        self.assertEqual(xml.folder, folder)
        self.assertEqual(xml.filename, filename)
        self.assertEqual(xml.path, path)
        self.assertEqual(xml.source, source)
        self.assertEqual(xml.size, size)
        self.assertEqual(xml.segmented, segmented)

    def test_save(self):

        save_data = "<annotation><folder></folder><filename></filename>" + \
            "<path></path><source><database>Unknown</database></source>" + \
            "<segmented>0</segmented><size><width>0</width>" + \
            "<height>0</height><depth>0</depth></size></annotation>"
        voc = PascalXml()
        voc.save(self.save_path)

        self.assertTrue(os.path.exists(self.save_path))

        with open(self.save_path) as f:
            data = f.read().strip()
            data = ''.join([line.strip() for line in data.split('\n')])
        self.assertEqual(data, save_data)
