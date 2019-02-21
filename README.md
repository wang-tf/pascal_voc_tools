# Pascal VOC Tools

This package includes some tools about pascal voc format dataset like read xml, write xml, resize image.


## XmlReader
`from pascal_voc_tools import XmlReader`
```
>>> xml_path = './test.xml'
>>> reader = XmlReader(xml_path)
>>> ann_dict = reader.load()
```
som functions for reading a xml file and geting data in it.

## XmlWriter
`from pascal_voc_tools import XmlWriter`

```
>>> writer = XmlWriter(image_path, image_width, image_height, image_depth, database, segmented)
>>> writer.add_object(name, xmin, ymin, xmax, ymax, pose, truncated, difficult)
>>> writer.save(save_path)
```

Actually, if you have a dict have the format same as loaded dict from XmlReader, you can simply used like:
```
>>> writer = XmlWriter()
>>> writer.save(save_path, ann_dict)
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

