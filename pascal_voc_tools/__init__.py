"""
Pascal Voc Tools
===============

This package provide some tools using for 
pascal voc format dataset and some usrful
functions.

some function in the below list:
1. XmlReader
2. XmlWriter
3. DatasetResize
4. DataSplit
5. DarknetDataset
6. SplitImageAnnotation
7. AnnotationTools
8. AnchorsKMeans
9. DarknetConfig
"""

from ._version import version
from .xmlreader import XmlReader
from .xmlwriter import XmlWriter
from .resize import DatasetResize
from .datatools import DataSplit
from .datatools import DarknetDataset
from .image_annotation_split import SplitImageAnnotation
from .annotation_tools import AnnotationTools
from .anchors_kmeans import AnchorsKMeans
from .darknet_config import DarknetConfig
from ._xml_parser import XmlParser

__all__ = ['version', 'XmlReader', 'XmlWriter', 'DatasetResize',
           'DataSplit', 'DarknetDataset', 'SplitImageAnnotation',
           'AnnotationTools', 'AnchorsKMeans', 'DarknetConfig', 'XmlParser']

name = 'pascal_voc_tools'
__version__ = version

