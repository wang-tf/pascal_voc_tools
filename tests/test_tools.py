import unittest
from pascal_voc_tools.tools import bb_intersection_over_union


class TestTools(unittest.TestCase):
    def test_bb_intersection_over_union(self):
        bboxA = [0, 0, 10, 10]
        bboxB = [0, 0, 10, 10]
        iou = bb_intersection_over_union(bboxA, bboxB)
        self.assertEqual(iou, 1.0)
