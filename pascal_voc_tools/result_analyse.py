#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import numpy as np
import glob
import cv2
from pascal_voc_tools import XmlParser
import openpyxl
from openpyxl.styles import Alignment, colors
import argparse
import tqdm

from .show_comp_result import save2xmlandshow


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('manual', help='voc Annotatins dir')
    parser.add_argument('yolo', help='')
    parser.add_argument('--main_name', default='test', type=str)
    parser.add_argument('--comp', default='./inference/comp4_det_test_SteelPipe.txt', type=str)
    parser.add_argument('--confthresh', default=0.5, type=float)
    parser.add_argument('--save_dir', default='./', type=str)
    args = parser.parse_args()
    return args


def compute_overlap(bbox, annotations):
    x1 = annotations[:, 0]  
    y1 = annotations[:, 1]  
    x2 = annotations[:, 2]  
    y2 = annotations[:, 3]  
  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
    bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)

    xx1 = np.maximum(bbox[0], x1[:])  
    yy1 = np.maximum(bbox[1], y1[:])  
    xx2 = np.minimum(bbox[2], x2[:])  
    yy2 = np.minimum(bbox[3], y2[:])  

    w = np.maximum(0.0, xx2 - xx1 + 1)  
    h = np.maximum(0.0, yy2 - yy1 + 1)  
    inter = w * h

    ovr = inter / (bbox_area + areas[:] - inter)
    return ovr


def get_objects(annotation_path):
    xml_reader = XmlParser()
    data = xml_reader.load(annotation_path)
    objects = data['object'] if 'object' in data else []
        
    class_bbox = {}
    for ob in objects:
        class_name = ob['name']
        bbox = ob['bndbox']
        if class_name not in class_bbox:
            class_bbox[class_name] = []
        class_bbox[class_name].append([int(bbox['xmin']),int(bbox['ymin']),int(bbox['xmax']),int(bbox['ymax'])])

    return class_bbox


def load_annotations(annotations_dir, main_name):
    test_file = os.path.join(annotations_dir, '../ImageSets/Main', main_name+'.txt')
    assert os.path.exists(test_file), test_file

    with open(test_file) as f:
        names = f.read().strip().split('\n')
    image_path_list = sorted([os.path.join(annotations_dir, '../JPEGImages', name+'.jpg') for name in names])

    annotations_data = {}
    for image_path in tqdm.tqdm(image_path_list):
        name = os.path.basename(image_path).replace('.jpg', '')
        annotation_path = os.path.join(annotations_dir, name+'.xml')

        class_bbox = get_objects(annotation_path)
        annotations_data[name] = class_bbox
    return annotations_data


def compare(main_name, detections_dir, annotations_dir, iou_threshold=0.3, save_path=None):

    annotations_data = load_annotations(annotations_dir, main_name)

    countthing_results = {}
    countthing_results['Total'] = {}

    for name in tqdm.tqdm(annotations_data.keys()):
        class_bbox = annotations_data[name]

        detection_path = os.path.join(detections_dir, name+'.xml')
        detection_class_bbox = get_objects(detection_path)

        detected_annotations = []
        wrong_detected = []
        image_false_positives = 0
        image_true_positives = 0

        countthing_results[name] = {}

        for class_name in class_bbox.keys():
            countthing_results[name][class_name] = class_results = {}

            image_path = os.path.join(annotations_dir, '../JPEGImages', name+'.jpg')
            image = cv2.imread(image_path)

            if class_name not in detection_class_bbox:
                detection_class_bbox[class_name] = []

            for d_index, d in enumerate(detection_class_bbox[class_name]):

                overlaps = compute_overlap(np.array(d), np.array(class_bbox[class_name]))
                assigned_annotation = np.argmax(overlaps)
                max_overlap = overlaps[assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    detected_annotations.append(assigned_annotation)
                    image_true_positives += 1
                else:
                    image_false_positives += 1
                    wrong_detected.append(d_index)

            if save_path:
                # draw
                img = cv2.imread(image_path)
                forget_detected = [i for i in range(len(class_bbox[class_name])) if i not in detected_annotations]
                for ann_index in forget_detected:
                    draw_object(img, class_bbox[class_name][ann_index], (0, 255, 255))
                for d_index in range(len(detection_class_bbox[class_name])):
                    if d_index in wrong_detected:
                        draw_object(img, detection_class_bbox[class_name][d_index], (0, 0, 255))
                    else:
                        draw_object(img, detection_class_bbox[class_name][d_index], (0, 255, 0), point=True)
                cv2.imwrite(os.path.join(save_path, os.path.basename(image_path).split('.')[0]+'-'+str(len(forget_detected)+len(wrong_detected))+'.jpg'), img)    

            image_recall = float(image_true_positives) / len(class_bbox[class_name])
            image_precision = float(image_true_positives) / max(image_true_positives+image_false_positives, np.finfo(np.float64).eps)
            image_f1 = 2 * (image_recall * image_precision) / max(image_recall + image_precision, np.finfo(np.float64).eps)

            class_results['DetectionsNumber'] = len(detection_class_bbox[class_name])
            class_results['AnnotationsNumber'] = len(class_bbox[class_name])
            class_results['TruePositive'] = image_true_positives
            class_results['FalsePositive'] = image_false_positives
            class_results['Recall'] = image_recall
            class_results['Precision'] = image_precision
            class_results['F1'] = image_f1
            class_results['业务识别率'] = 1-(class_results['FalsePositive']+class_results['AnnotationsNumber']-class_results['TruePositive'])/class_results['AnnotationsNumber']

            if class_name not in countthing_results['Total']:
                countthing_results['Total'][class_name] = {'TruePositive': 0, 'FalsePositive': 0, 'AnnotationsNumber': 0, 'DetectionsNumber': 0, '业务识别率': 0}
            countthing_results['Total'][class_name]['TruePositive'] += image_true_positives
            countthing_results['Total'][class_name]['FalsePositive'] += image_false_positives
            countthing_results['Total'][class_name]['AnnotationsNumber'] += len(class_bbox[class_name])
            countthing_results['Total'][class_name]['DetectionsNumber'] += len(detection_class_bbox[class_name])
            countthing_results['Total'][class_name]['业务识别率'] += class_results['业务识别率']
    
    for class_name in countthing_results['Total'].keys():
        true_positives = countthing_results['Total'][class_name]['TruePositive']
        false_positives = countthing_results['Total'][class_name]['FalsePositive']
        annotations_number = countthing_results['Total'][class_name]['AnnotationsNumber']
        recall = float(true_positives) / annotations_number
        precision = float(true_positives) / max(true_positives + false_positives, np.finfo(np.float64).eps)
        f1 = 2 * (recall * precision) / (recall + precision)

        countthing_results['Total'][class_name]['Recall'] = recall
        countthing_results['Total'][class_name]['Precision'] = precision
        countthing_results['Total'][class_name]['F1'] = f1
        countthing_results['Total'][class_name]['业务识别率'] /= len(annotations_data.keys())
    
    return countthing_results


def merge_results(results):
    output = {}
    for model_key, model_val in results.items():
        for image_key, image_val in model_val.items():
            if image_key not in output:
                output[image_key] = {}

            for class_key, class_val in image_val.items():
                if class_key not in output[image_key]:
                    output[image_key][class_key] = {}
                output[image_key][class_key][model_key] = class_val
    
    return output


def save_result(save_path, results):
    results = merge_results(results)

    wb = openpyxl.Workbook()
    sheet = wb['Sheet']
    row_index = 0
    column_index = 65

    name_column_index = chr(column_index)
    class_column_index = chr(column_index+1)
    annotations_number_column_index = column_index+2
    sheet['{}{}'.format(name_column_index, row_index+2)] = 'Name'
    sheet['{}{}'.format(class_column_index, row_index+2)] = 'Class'
    sheet['{}{}'.format(chr(annotations_number_column_index), row_index+2)] = 'Annotations Number'

    row_start = 3
    for image_key in sorted(results.keys()):
        image_val = results[image_key]
        # name
        sheet['{}{}'.format(name_column_index, row_start)] = image_key

        for class_key in sorted(image_val.keys()):
            class_val = image_val[class_key]
            # classm, annotations number
            sheet['{}{}'.format(class_column_index, row_start)] = class_key

            model_column_start = annotations_number_column_index + 1
            for model_key in sorted(class_val.keys()):
                model_val = class_val[model_key]
                
                if sheet['{}{}'.format(chr(annotations_number_column_index), row_start)].value == None:
                    sheet['{}{}'.format(chr(annotations_number_column_index), row_start)] = model_val['AnnotationsNumber']
                else:
                    assert sheet['{}{}'.format(chr(annotations_number_column_index), row_start)].value == model_val['AnnotationsNumber'], (
                        sheet['{}{}'.format(chr(annotations_number_column_index), row_start)],
                        model_val['AnnotationsNumber']
                    )

                # add title
                if sheet['{}{}'.format(chr(model_column_start), row_index+1)].value == None:
                    item_number = len(model_val)-1
                    sheet.merge_cells('{}{}:{}{}'.format(chr(model_column_start), row_index+1, chr(model_column_start+item_number-1), row_index+1))
                    sheet['{}{}'.format(chr(model_column_start), row_index+1)].alignment = Alignment(horizontal='center', vertical='center')
                    sheet['{}{}'.format(chr(model_column_start), row_index+1)] = model_key

                for index, item_key in enumerate(sorted(model_val.keys())):
                    item_val = model_val[item_key]
                    if item_key in ['AnnotationsNumber']:
                        continue
                    # add title
                    if sheet['{}{}'.format(chr(model_column_start), row_index+2)].value == None:
                        sheet['{}{}'.format(chr(model_column_start), row_index+2)] = item_key
        
                    sheet['{}{}'.format(chr(model_column_start), row_start)] = '{:.4f}'.format(item_val) if isinstance(item_val, float) else item_val

                    model_column_start += 1
            
        row_start += 1
    
    wb.save(save_path)


def draw_object(img, bndbox, color, text=None, point=False):
    start = (int(float(bndbox[0])), int(float(bndbox[1])))
    end = (int(float(bndbox[2])), int(float(bndbox[3])))
    if point:
        radius = min(end[0] - start[0], end[1] - start[1])
        center = (int((end[0] - start[0])/2+start[0]), int((end[1] - start[1])/2+start[1]))
        cv2.circle(img, center, int(radius/2), color, 2) 
    else:
        cv2.rectangle(img, start, end, color, 2)

    if text:
        cv2.putText(img, text, start, cv2.FONT_HERSHEY_COMPLEX, 6, color, 2)


def main(iou_threshold=0.45):
    args = get_args()
    assert os.path.exists(args.manual)
    assert os.path.exists(args.yolo)

    image_dir = os.path.join(args.manual, '../JPEGImages')
    label_path = os.path.join(image_dir, '../../labels.list')
    # TODO:(wangtf) 将comp转为xml
    save2xmlandshow(args.comp, image_dir, label_path, args.confthresh)
        
    manual_annotations_dir = args.manual  # '../VOCdevkit_rebar_test_0/VOC2007/Annotations'
    yolo_detections_dir = args.yolo  # './yolov3'

    yolo_result = compare(args.main_name, yolo_detections_dir, manual_annotations_dir, iou_threshold=iou_threshold, save_path=args.save_dir)
    
    save_dict = {#'CountThings': countthing_result,
                 #'RetinaNet_Torch': retinanet_torch_result,
                 'yolo': yolo_result,
}
    save_result('./recognition_result_compare-iou_{}.xlsx'.format(iou_threshold), save_dict)

    #print('CountThings', countthing_result['Total'])
    #print('Retinanet', retinanet_result['Total'])


if __name__ == '__main__':
    main()

