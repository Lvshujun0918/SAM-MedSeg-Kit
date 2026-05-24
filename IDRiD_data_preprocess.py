import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from pwn import log

# =====================================
# 配置
# =====================================
CONFIG = {
    "input_root": "DataSets/IDRiD",
    "output_root": "DataSets/IDRiD_pro"
}

LESION_TYPES = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
    "OD": "5. Optic Disc"
}

# =====================================
# 创建目录
# =====================================
def prepare_dirs():
    for lesion in LESION_TYPES:
        for split in ["training", "testing"]:
            out_dir = os.path.join(
                CONFIG["output_root"],
                lesion,
                split
            )
            os.makedirs(out_dir, exist_ok=True)

# =====================================
# 获取原图路径
# =====================================
def get_image_path(split, image_name):

    if split == "training":
        img_dir = os.path.join(
            CONFIG["input_root"],
            "1. Original Images",
            "a. Training Set"
        )
    else:
        img_dir = os.path.join(
            CONFIG["input_root"],
            "1. Original Images",
            "b. Testing Set"
        )

    return os.path.join(img_dir, image_name + ".jpg")

# =====================================
# 处理 lesion
# =====================================
def process_lesion(split, lesion_key, lesion_folder):

    split_name = "a. Training Set" if split == "training" else "b. Testing Set"

    gt_dir = os.path.join(
        CONFIG["input_root"],
        "2. All Segmentation Groundtruths",
        split_name,
        lesion_folder
    )
    os.makedirs(gt_dir, exist_ok=True)

    gt_files = sorted([
        f for f in os.listdir(gt_dir)
        if f.endswith(".tif")
    ])

    log.info(f"{lesion_key}-{split}: {len(gt_files)} masks")

    for gt_file in tqdm(gt_files, desc=f"{lesion_key}-{split}"):

        base_name = gt_file.replace(f"_{lesion_key}.tif", "")

        img_path = get_image_path(split, base_name)
        gt_path = os.path.join(gt_dir, gt_file)

        if not os.path.exists(img_path):
            log.warning(f"原图不存在: {img_path}")
            continue

        image = cv2.imread(img_path)
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            log.warning(f"读取失败: {base_name}")
            continue

        mask = (mask > 0).astype(np.uint8) * 255

        out_dir = os.path.join(
            CONFIG["output_root"],
            lesion_key,
            split
        )

        out_img = os.path.join(out_dir, f"{base_name}.png")
        out_mask = os.path.join(out_dir, f"{base_name}_mask.png")

        cv2.imwrite(out_img, image)
        cv2.imwrite(out_mask, mask)

# =====================================
# 主函数
# =====================================
def main():

    prepare_dirs()

    for lesion_key, lesion_folder in LESION_TYPES.items():

        process_lesion("training", lesion_key, lesion_folder)
        process_lesion("testing", lesion_key, lesion_folder)

    log.success("IDRiD preprocess完成")

# =====================================
if __name__ == "__main__":
    main()