使用Pyhton处理图片的工具集

* 在Pyhon3下测试通过
* 部分需要OpenCV for Python

## 说明

`convert_to_square.py` 并用黑色填充无像素区域，将图片比例变更为 1:1, 。

`count_image.py` 遍历指定文件夹，对图片进行计数操作。

`extract_hand_from_PASCAL.py` 从PASCAL 2012数据库中提取手部的位置信息，并存储为csv格式。

`get_stereo_cam_images.py` 将双目摄像机的画面保存为文件，并自动编号。

`random_split.py` 将指定文件夹下的图片按比例随机分离到两个文件夹。

`resize_image.py` 遍历指定文件夹，对图片进行尺寸变换操作。

`pts_tools.py` 读取[IBUG](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)数据，并将面部特征点的结果显示在图片上。

`count_files.py` 遍历制定文件夹，对所有文件按照后缀名进行计数操作。

`detect_face.py` 从视频或者摄像头中检测人脸