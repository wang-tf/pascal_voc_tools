# -*- coding:utf-8 -*-

import random


def split_list_by_rate(test_rate, val_rate=0.0, name_list=None, shuffle=False):
    """
    Args:
        test_rate: float, the test data rate for all data;
        val_rate: float, default is 0.0, the val data rate for all data;
        name_list: list, all useful name in this data.
        shuffle: bool, default is False, The name_list will be shuffled
                    if it is true.

    Returns:
        splited_data: map, the key is str in ['train', 'val', 'test'],
                        the value is the list of names. 
    """
    assert test_rate < 1, 'Error: test_rate {} not in range.'.format(test_rate)
    assert len(name_list) > 2, 'Error: name_list length is needed more than 2.'

    if shuffle:
        random.shuffle(name_list)

    test_number = int(test_rate * len(name_list))
    test_number = test_number if test_number > 0 else 1

    if val_rate > 0:
        val_number = int(val_rate * len(name_list))
        val_number = val_number if val_number > 0 else 1
    else:
        val_number = 0

    train_number = len(name_list) - test_number - val_number
    assert train_number > 0, 'Error: train_number is needed more than 0.'

    train_list = name_list[0:train_number]
    test_list = name_list[train_number:train_number + test_number]
    if val_number > 0:
        val_list = name_list[-val_number:]
    else:
        val_list = []
    return {'train': train_list, 'val': val_list, 'test': test_list}