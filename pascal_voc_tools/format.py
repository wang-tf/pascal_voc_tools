#!/usr/bin/env python3

import os
import glob
import shutil
import tqdm


def check_devkit_format(root_dir):
    """
    Check file tree format in Devkit dir.

    Args:
        root_dir:, str, including some VOC**** dir.
    """
    check_result = False
    assert os.path.isdir(root_dir), root_dir
    files = glob.glob(os.path.join(root_dir, '*'))
    for file_path in files:
        if os.path.isfile(file_path):
            continue
        if 'VOC' == os.path.basename(file_path)[:3]:
            check_result = check_voc_format(file_path)
            if not check_result:
                return check_result
    return check_result


def check_voc_format(voc_path):
    """
    Check file tree format in VOC dir like VOC2007.

    Args:
        voc_path: str, the dir path should including Annotations and JPEGImages.
    """
    check_result = True
    assert os.path.exists(voc_path), voc_path
    ann_dir = os.path.join(os.path.join(voc_path, 'Annotations'))
    jpeg_dir = os.path.join(os.path.join(voc_path, 'JPEGImages'))

    if not os.path.exists(ann_dir):
        check_result = False
    if not os.path.exists(jpeg_dir):
        check_result = False

    # generate ImageSets/Main dir
    if check_result:
        main_dir = os.path.join(os.path.join(voc_path, 'ImageSets/Main'))
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)
    return check_result


def gen_voc_format(ann_dir, jpeg_dir, save_dir):
    """
    Generate a new format pascal voc data.

    Args:
        ann_dir: str, the path having xmls;
        jpeg_dir: str, the path having images;
        save_dir: str, the path saving file.
    """
    assert os.path.isdir(ann_dir), ann_dir
    assert os.path.isdir(jpeg_dir), jpeg_dir
    while os.path.exists(save_dir):
        print('The dir is existed: {}'.format(save_dir))
        save_dir = input('Please input an new folder:')

    ann_save_dir = os.path.join(save_dir, 'Annotations')
    jpeg_save_dir = os.path.join(save_dir, 'JPEGImages')
    main_save_dir = os.path.join(save_dir, 'ImageSets/Main')

    os.makedirs(ann_save_dir)
    os.makedirs(jpeg_save_dir)
    os.makedirs(main_save_dir)

    xml_file_list = glob.glob(os.path.join(ann_dir, '*.xml'))
    print('Find xml file number: {}'.format(len(xml_file_list)))

    for xml_file in tqdm.tqdm(xml_file_list):
        jpeg_file = os.path.join(
            jpeg_dir,
            os.path.basename(xml_file).replace('.xml', '.jpg'))
        assert os.path.isfile(jpeg_file), jpeg_file
        shutil.copy2(jpeg_file, jpeg_save_dir)
        shutil.copy2(xml_file, ann_save_dir)
    print('Done.')
    return save_dir