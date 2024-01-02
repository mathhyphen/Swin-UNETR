# Swin UNETR

本仓库是基本上从 [MONAI](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb) 里面改写而来的，所以有什么疑问，可以直接参考MONAI的一些教程。采用的网络也是比较新的 [Swin UNETR](https://arxiv.org/pdf/2201.01266.pdf) .

## 训练步骤

### /data/create_json_for_2021.py

这里需要修改数据集的地址，我们这里使用的是Brats2021数据集。该文件主要是生成比较好读取数据的 json文件.

### swin_unetr.py

这个文件主要是修改，读取文件的地址，保存模型文件的地址，以及相应的使用哪个CUDA训练模型等等。

### test_swin_unetr.py

测试相应的模型

## 需要安装的库
monai
nibabel
