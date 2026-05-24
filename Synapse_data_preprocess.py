import os
import cv2
import numpy as np
import nibabel as nib

from tqdm import tqdm

# =====================================
# 配置
# =====================================

ROOT = "DataSets/Synapse"

OUT_DIR = "DataSets/Synapse_pro"

IMG_SIZE = 1024

WINDOW_MIN = -125
WINDOW_MAX = 275

# =====================================
# 器官
# =====================================

TARGET_ORGANS = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    6: "liver",
    11: "pancreas",
}

# =====================================
# CT window
# =====================================

def normalize_ct(img):

    img = np.clip(
        img,
        WINDOW_MIN,
        WINDOW_MAX
    )

    img = (img - WINDOW_MIN) / (
        WINDOW_MAX - WINDOW_MIN
    )

    img = (img * 255).astype(np.uint8)

    return img

# =====================================
# resize
# =====================================

def resize_image(img):

    return cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

def resize_mask(mask):

    return cv2.resize(
        mask,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

# =====================================
# 保存
# =====================================

def save_png(path, arr):

    cv2.imwrite(path, arr)

# =====================================
# 单 case
# =====================================

def process_case(
    img_path,
    label_path,
    split="training"
):

    img_nii = nib.load(img_path)
    label_nii = nib.load(label_path)

    img = img_nii.get_fdata()
    label = label_nii.get_fdata()

    case_name = os.path.basename(
        img_path
    ).replace(".nii.gz", "")

    depth = img.shape[-1]

    print(f"\n[*] {case_name} depth={depth}")

    for z in tqdm(range(depth), leave=False):

        img_slice = img[:, :, z]
        label_slice = label[:, :, z]

        # 整张 slice 没器官
        if np.sum(label_slice) == 0:
            continue

        img_slice = normalize_ct(img_slice)

        img_slice = resize_image(img_slice)

        # =====================================
        # 每个器官单独导出
        # =====================================

        for label_id, organ_name in TARGET_ORGANS.items():

            organ_mask = (
                label_slice == label_id
            ).astype(np.uint8)

            # 当前 slice 无器官
            if organ_mask.sum() == 0:
                continue

            organ_mask = organ_mask * 255

            organ_mask = resize_mask(
                organ_mask
            )

            out_dir = os.path.join(
                OUT_DIR,
                organ_name,
                split
            )

            os.makedirs(
                out_dir,
                exist_ok=True
            )

            base_name = (
                f"{case_name}_{z:03d}"
            )

            img_save = os.path.join(
                out_dir,
                f"{base_name}.png"
            )

            mask_save = os.path.join(
                out_dir,
                f"{base_name}_mask.png"
            )

            save_png(
                img_save,
                img_slice
            )

            save_png(
                mask_save,
                organ_mask
            )

# =====================================
# 主函数
# =====================================

def main():

    train_img_dir = os.path.join(
        ROOT,
        "imagesTr"
    )

    train_label_dir = os.path.join(
        ROOT,
        "labelsTr"
    )

    img_files = sorted(
        os.listdir(train_img_dir)
    )

    for img_file in tqdm(img_files):

        img_path = os.path.join(
            train_img_dir,
            img_file
        )

        label_file = img_file.replace(
            "img",
            "label"
        )

        label_path = os.path.join(
            train_label_dir,
            label_file
        )

        if not os.path.exists(label_path):

            print(
                f"[!] Missing label: {label_file}"
            )

            continue

        process_case(
            img_path,
            label_path,
            split="training"
        )

    print("\nDONE")

# =====================================

if __name__ == "__main__":
    main()