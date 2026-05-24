import os
import cv2
import json
import time
import numpy as np
import pandas as pd

from tqdm import tqdm
from pwn import log
from ultralytics.models.sam import SAM3SemanticPredictor

from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
from skimage import measure

# =========================
# 配置
# =========================
CONFIG = {
    "model": "Models/sam3.pt",
    "image_dir": "DataSets/Synapse_pro",
    "output_root": "DataRes/Synapse_pro"
}

EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif")

# =========================
# 不同器官配置 (Synapse数据集常见分类)
# =========================
TASKS = {
    "kidney_left": {
        "prompt": "left kidney",
        "mask_suffix": "_mask.png",
        "output_dir": "sam3_KidneyL_preds"
    },
    "kidney_right": {
        "prompt": "right kidney",
        "mask_suffix": "_mask.png",
        "output_dir": "sam3_KidneyR_preds"
    },
    "liver": {
        "prompt": "liver",
        "mask_suffix": "_mask.png",
        "output_dir": "sam3_Liver_preds"
    },
    "spleen": {
        "prompt": "spleen",
        "mask_suffix": "_mask.png",
        "output_dir": "sam3_Spleen_preds"
    },
    "pancreas": {
        "prompt": "pancreas",
        "mask_suffix": "_mask.png",
        "output_dir": "sam3_pancreas_preds"
    }
}

# =========================
# 模型
# =========================
def load_model():
    predictor = SAM3SemanticPredictor(
        overrides=dict(
            conf=0.1,
            task="segment",
            mode="predict",
            model=CONFIG["model"],
            save=False,
            verbose=False,
    
        )
    )
    log.success("SAM3模型加载成功")
    return predictor

# =========================
# GT Mask
# =========================
def load_gt_mask(img_path, mask_suffix):
    base = os.path.splitext(img_path)[0]
    mask_path = base + mask_suffix
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return (mask > 0)
    tqdm.write("[!] GT不存在")
    return None

# =========================
# 推理
# =========================
def run_inference(predictor, img_path, prompt):
    predictor.set_image(img_path)
    results = predictor(text=[prompt])
    if results[0].masks is None:
        return None, []
    masks = results[0].masks.data.cpu().numpy()
    pred_mask = masks.sum(axis=0) > 0
    if hasattr(results[0], "boxes") and results[0].boxes is not None:
        conf = results[0].boxes.conf.cpu().numpy().tolist()
    else:
        conf = [1.0] * masks.shape[0]
    return pred_mask, conf

# =========================
# CMP图
# =========================
def generate_cmp(pred, gt):
    if pred is None or gt is None:
        return None

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.shape != gt.shape:
        gt = cv2.resize(
            gt.astype(np.uint8),
            (pred.shape[1], pred.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    cmp_img = np.zeros((*pred.shape, 3), dtype=np.uint8)

    cmp_img[np.logical_and(pred, gt)] = [255, 255, 255]   # TP 白色
    cmp_img[np.logical_and(pred, ~gt)] = [0, 0, 255]      # FP 红色
    cmp_img[np.logical_and(~pred, gt)] = [0, 255, 0]      # FN 绿色

    return cmp_img

# =========================
# 分割指标
# =========================
def calculate_segmentation_metrics(pred_mask, gt_mask):
    if pred_mask is None or gt_mask is None:
        return {}

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    if pred.shape != gt.shape:
        gt = cv2.resize(
            gt.astype(np.uint8),
            (pred.shape[1], pred.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    pred_sum = pred.sum()
    gt_sum = gt.sum()

    dice = (2 * intersection) / (pred_sum + gt_sum + 1e-6)
    iou = intersection / (union + 1e-6)
    sensitivity = intersection / (gt_sum + 1e-6)
    precision = intersection / (pred_sum + 1e-6)

    hausdorff = None
    hausdorff_95 = None
    asd = None

    if pred_sum > 0 and gt_sum > 0:
        pred_contours = measure.find_contours(pred, 0.5)
        gt_contours = measure.find_contours(gt, 0.5)

        if pred_contours and gt_contours:
            pred_pts = np.vstack(pred_contours)
            gt_pts = np.vstack(gt_contours)

            h1 = directed_hausdorff(pred_pts, gt_pts)[0]
            h2 = directed_hausdorff(gt_pts, pred_pts)[0]

            hausdorff = max(h1, h2)

            tree_gt = cKDTree(gt_pts)
            tree_pred = cKDTree(pred_pts)

            d_pred = tree_gt.query(pred_pts)[0]
            d_gt = tree_pred.query(gt_pts)[0]

            hausdorff_95 = max(
                np.percentile(d_pred, 95),
                np.percentile(d_gt, 95)
            )

            asd = (d_pred.mean() + d_gt.mean()) / 2

    return {
        "dice": round(dice, 4),
        "iou": round(iou, 4),
        "sensitivity": round(sensitivity, 4),
        "precision": round(precision, 4),
        "hausdorff_distance": round(hausdorff, 2) if hausdorff else None,
        "hausdorff_distance_95": round(hausdorff_95, 2) if hausdorff_95 else None,
        "average_surface_distance": round(asd, 2) if asd else None,
        "intersection": int(intersection),
        "union": int(union),
        "pred_area": int(pred_sum),
        "gt_area": int(gt_sum)
    }

# =========================
# 收集图像
# =========================
def collect_all_images(root_dir):
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(EXTENSIONS):
                if "_mask" in fname:
                    continue
                image_paths.append(os.path.join(root, fname))
    return sorted(image_paths)

# =========================
# 单图处理
# =========================
def process_one(
        predictor,
        img_path,
        out_dir,
        prompt,
        mask_suffix
):
    rel_path = os.path.relpath(img_path, CONFIG["image_dir"])
    unique_name = rel_path.replace(os.sep, "_")
    start = time.time()
    gt = load_gt_mask(img_path, mask_suffix)
    tqdm.write(f"[*] 开始处理{unique_name}")
    pred, conf = run_inference(
        predictor,
        img_path,
        prompt
    )

    cmp_img = generate_cmp(pred, gt)
    seg_metrics = calculate_segmentation_metrics(pred, gt)

    if cmp_img is not None:
        cmp_path = os.path.join(
            out_dir,
            f"cmp_{unique_name}"
        )
        cv2.imwrite(cmp_path, cmp_img)
    else:
        tqdm.write("[!] 比较失败！")

    return {
        "filename": unique_name,
        "category": os.path.dirname(rel_path),
        "prompt": prompt,
        "has_pred": pred is not None,
        "has_gt": gt is not None,
        "num_det": len(conf),
        "avg_conf": round(np.mean(conf), 4) if conf else 0,
        "time": round(time.time() - start, 3),
        **seg_metrics
    }

# =========================
# 单任务运行
# =========================
def run_task(
        predictor,
        task_name,
        task_cfg,
        all_images
):
    log.info(f"开始任务: {task_name}")

    out_dir = os.path.join(
        CONFIG["output_root"],
        task_cfg["output_dir"]
    )

    os.makedirs(out_dir, exist_ok=True)

    results = []

    for img_path in tqdm(
            all_images,
            desc=f"{task_name}",
            ncols=100
    ):
        res = process_one(
            predictor,
            img_path,
            out_dir,
            task_cfg["prompt"],
            task_cfg["mask_suffix"]
        )
        results.append(res)

    df = pd.DataFrame(results)

    csv_path = os.path.join(
        out_dir,
        "results_full.csv"
    )
    df.to_csv(csv_path, index=False)

    json_path = os.path.join(
        out_dir,
        "results_full.json"
    )
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    valid = df[df["has_gt"] & df["has_pred"]]

    if len(valid) > 0:
        log.success(f"==== {task_name} 总体指标 ====")
        log.info(f"Dice: {valid['dice'].mean():.4f}")
        log.info(f"IoU : {valid['iou'].mean():.4f}")
        log.info(f"HD95: {valid['hausdorff_distance_95'].mean():.2f}")
        log.info(f"ASD : {valid['average_surface_distance'].mean():.2f}")

# =========================
# 主函数
# =========================
def main():
    predictor = load_model()
    for task_name, task_cfg in TASKS.items():
        prom = task_cfg["prompt"]
        log.success(f"开始执行{task_name}: {prom}")
        all_images = collect_all_images(
            os.path.join(CONFIG["image_dir"], task_name)
        )
        log.info(f"共发现 {len(all_images)} 张图像")
        run_task(
            predictor,
            task_name,
            task_cfg,
            all_images
        )
    log.success("全部任务处理完成")

# =========================
if __name__ == "__main__":
    main()