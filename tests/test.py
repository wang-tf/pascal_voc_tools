
#!/usr/bin/env python3

import unittest
from xmlreader import XmlReader

class TestXmlReader(unittest.TestCase):
    def test_load(self):
        xml_reader = XmlReader('test.xml')
        xml_info = {}
        self.assertEqual(xml_info, xml_reader.load())
    
    def test_call(self):
        xml_reader = XmlReader('test.xml')
        xml_info = {}
        self.assertEqual(xml_info, xml_reader())

