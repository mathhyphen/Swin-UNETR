import os
import json
import shutil
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai import data
from monai.data import decollate_batch
from functools import partial

import torch


print_config()

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = "/mnt/data/wanghaifeng/GenerativeModels-main/log/" if directory is None else directory
print(root_dir)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def datafold_read(datalist, basedir, key="t1_t2"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    val = json_data

    return val


def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, dir_add=root_dir):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def get_loader(batch_size, data_dir, json_list, roi):
    data_dir = data_dir
    datalist_json = json_list
    validation_files = datafold_read(datalist=datalist_json, basedir=data_dir)

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return val_loader


data_dir = "/home/ssd/wanghaifeng/Data/Data_bu/subjects"
json_list = "/home/ssd/wanghaifeng/GenerativeModels-main/data/BraTS_first1_test_add.json"
roi = (128, 128, 128)
batch_size = 1
sw_batch_size = 2
infer_overlap = 0.5
val_loader = get_loader(batch_size, data_dir, json_list, roi)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:4")

model = SwinUNETR(
    img_size=roi,
    in_channels=2,
    out_channels=3,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=True,
).to(device)

model.load_state_dict(torch.load("/home/ssd/wanghaifeng/GenerativeModels-main/log/0821model_t1_t2.pt")["state_dict"])
model.eval()

model_inferer_test = partial(
    sliding_window_inference,
    roi_size=[roi[0], roi[1], roi[2]],
    sw_batch_size=1,
    predictor=model,
    overlap=0.6,
)

# 读取原始图像的affine
original_image_path = "/home/ssd/wanghaifeng/Data/Data_bu/subjects/Sub_132/SRI_Sub_132-flair.nii.gz"  # 替换为原始图像的路径
original_image = nib.load(original_image_path)
affine = original_image.affine

with torch.no_grad():
    for batch_data in val_loader:
        image = batch_data["image"].to(device)
        name = batch_data["name"][0]
        seg_name = name.split("/")[-1] + "-seg.nii.gz"
        save_path = os.path.join(name, seg_name)

        prob = torch.sigmoid(model_inferer_test(image))
        seg = prob[0].detach().cpu().numpy()
        seg = (seg > 0.5).astype(np.int8)
        seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
        seg_out[seg[1] == 1] = 2
        seg_out[seg[0] == 1] = 1
        seg_out[seg[2] == 1] = 1

        nifti_img = nib.Nifti1Image(seg_out, affine)
        nib.save(nifti_img, save_path)

        print(f"{name.split('/')[-1]} has done!")