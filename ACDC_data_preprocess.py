import os
import cv2
import numpy as np
import nibabel as nib
from pwn import log

# =====================================
# 配置
# =====================================
CONFIG = {
    "input_root": "/vol1/1000/Sync/工作流/sam论文/DataSets/ACDC",
    "output_root": "/vol1/1000/Sync/工作流/sam论文/DataSets/ACDC_pro"
}

# =====================================
# 保存 PNG
# =====================================
def save_png(img, out_path):
    img = img.astype(np.float32)

    # 防止全黑
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    img = (img * 255).astype(np.uint8)

    cv2.imwrite(out_path, img)

# =====================================
# 保存 Mask
# =====================================
def save_mask(mask, out_path):
    mask = mask.astype(np.uint8)

    # ACDC标签:
    # 0 background
    # 1 RV
    # 2 MYO
    # 3 LV
    #
    # 统一转二值mask

    mask = (mask > 0).astype(np.uint8) * 255

    cv2.imwrite(out_path, mask)

# =====================================
# 处理单个 nii.gz
# =====================================
def process_case(img_path, gt_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    case_name = os.path.basename(img_path).replace(".nii.gz", "")

    log.info(f"处理: {case_name}")

    # 读取数据
    img_nii = nib.load(img_path)
    gt_nii = nib.load(gt_path)

    img_data = img_nii.get_fdata()
    gt_data = gt_nii.get_fdata()

    # ACDC一般 shape:
    # (H, W, Slice)

    if len(img_data.shape) != 3:
        log.error(f"{case_name} 维度异常: {img_data.shape}")
        return

    num_slices = img_data.shape[-1]

    saved = 0

    for idx in range(num_slices):
        img_slice = img_data[:, :, idx]
        gt_slice = gt_data[:, :, idx]

        # 跳过空mask
        if np.sum(gt_slice) == 0:
            continue

        img_name = f"{case_name}_{idx:03d}.png"
        mask_name = f"{case_name}_{idx:03d}_mask.png"

        img_out = os.path.join(out_dir, img_name)
        mask_out = os.path.join(out_dir, mask_name)

        save_png(img_slice, img_out)
        save_mask(gt_slice, mask_out)

        saved += 1

    log.success(f"{case_name}: 保存 {saved} 个切片")

# =====================================
# 遍历patient
# =====================================
def process_dataset(split_name):
    split_dir = os.path.join(CONFIG["input_root"], split_name)

    if not os.path.exists(split_dir):
        log.error(f"{split_dir} 不存在")
        return

    patient_dirs = sorted([
        os.path.join(split_dir, d)
        for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])

    log.info(f"{split_name}: 共 {len(patient_dirs)} 个patient")

    for patient_dir in patient_dirs:

        patient_name = os.path.basename(patient_dir)

        output_dir = os.path.join(
            CONFIG["output_root"],
            split_name,
            patient_name
        )

        files = os.listdir(patient_dir)

        # 找 frameXX.nii.gz
        image_files = sorted([
            f for f in files
            if f.endswith(".nii.gz")
            and "_gt" not in f
            and "_4d" not in f
        ])

        for img_file in image_files:

            img_path = os.path.join(patient_dir, img_file)

            gt_file = img_file.replace(".nii.gz", "_gt.nii.gz")
            gt_path = os.path.join(patient_dir, gt_file)

            if not os.path.exists(gt_path):
                log.error(f"GT不存在: {gt_file}")
                continue

            process_case(
                img_path,
                gt_path,
                output_dir
            )

# =====================================
# 主函数
# =====================================
def main():

    os.makedirs(CONFIG["output_root"], exist_ok=True)

    # training
    process_dataset("training")

    # testing
    process_dataset("testing")

    log.success("ACDC 全部处理完成")

# =====================================
if __name__ == "__main__":
    main()