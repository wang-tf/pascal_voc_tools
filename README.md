# Pascal VOC Tools

This package includes some tools about pascal voc format dataset like read xml, write xml, resize image.


## XmlReader
`from pascal_voc_tools import XmlReader`

som functions for reading a xml file and geting data in it.

## XmlWriter
`from pascal_voc_tools import XmlWriter`

```
>>> writer = XmlWriter(image_path, image_width, image_height, image_depth, database, segmented)
>>> writer.addObject(name, xmin, ymin, xmax, ymax, pose, truncated, difficult)
>>> writer.save(save_path)
```

## DatasetResize
`from pascal_voc_tools import DatasetResize`

```
>>> resizer = DatasetResize(root_voc_dir, save_voc_dir)
>>> resizer.resize_dataset_by_min_size(min_size)
>>> resizer.copy_imagesets()  # if the file include
```

## DataSplit
`from pascal_voc_tools import DataSplit`

```
>>> spliter = DataSplit(root_dir)
>>> result = spliter.split_by_rate(test_rate)
>>> spliter.save(result)

```

