# image_utility

Handy python scripts for image data-set processing.

## Getting Started

`count_files.py` 遍历制定文件夹，对所有文件按照后缀名进行计数操作。

`file_list_generator` 列出特定目录下指定格式的文件。

`pts_tools.py` 读取[IBUG](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)数据，并将面部特征点的结果显示在图片上。

`detect_face.py` 从视频或者摄像头中检测人脸。

`pose_estimator.py` 通过solvePnP的方法估算人头部的姿态。


### Prerequisites

- Python3
- OpenCV (Python)

### Installing

From the directory where you want to store this repo:
```
git clone https://github.com/yinguobing/image_utility
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
