# -*- coding:utf-8 -*-


def prefix_grouping(prefix_list, name_list):
    """name_list will be splited some groups for which all name are
    started with one prefix

    Args:
        prefix_list: list, the images which have the same string
                        will save in the same list that as the value
                        of the prefix string as the key.
        name_list: list, default is None, whick have all useful
                        name as the whole dataset.
    Returns:
        groups: map, the key is the string in prefix_list, the value
                is a list that all name in it has corresponding prefix.
    """
    groups = {}
    for prefix in prefix_list:
        groups[prefix] = []

    for name in name_list:
        for prefix in prefix_list:
            if name.startswith(prefix):
                groups[prefix].append(name)

    return groups
