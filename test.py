
def save_element_info(save_dict, tag, text):
    if tag not in save_dict:
        save_dict[tag] = text
    else:
        if not isinstance(save_dict[tag], list):
            save_dict[tag] = [save_dict[tag]]
        save_dict[tag].append(text)
   
         
def parse_element(element, save_dict=None):
    if save_dict is None:
        save_dict = {}

    for child in element:

        if len(child) == 0:
            save_element_info(save_dict, child.tag, child.text)
        else:
            save_element_info(save_dict, child.tag, parse_element(child))

    return save_dict

import xml.etree.cElementTree as ET
import pprint

xml = '/home/wangtf/ShareDataset/dataset/RebarDataset/VOCdevkit_rebar_v10_0-12-14-16-18-20-22-25-32/VOC2007/Annotations/12_1_000.xml'
tree = ET.parse(xml)
root = tree.getroot()

annotations = parse_element(root)
pprint.pprint(annotations)
