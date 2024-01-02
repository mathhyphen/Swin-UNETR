import json
import os
import numpy as np
import nibabel as nib


# 最后一个 fold 显示 5，所以有错误还要改一下
root_dir = '/mnt/data/wanghaifeng/Data/BraTS2021'
modal = ["_t1ce", "_t1", "_t2", "_flair"]
modal = [x + ".nii.gz" for x in modal]

files = os.listdir(root_dir)
np.random.shuffle(files)

training_data = []
for i in range(len(files)):
    fold_num = i // (len(files) // 5)
    img = ["BraTS2021/" + files[i] + "/" + files[i] + x for x in modal]
    lab = "BraTS2021/" + files[i] + "/" + files[i] + "_seg.nii.gz"
    training_dict = {"fold": fold_num, "image": img, "label": lab, "name": files[i]}
    training_data.append(training_dict)

data = {"training": training_data}
with open("./brats_2021.json", "w") as f:
    json.dump(data, f)
