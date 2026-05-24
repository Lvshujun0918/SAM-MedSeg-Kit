import os
import cv2
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
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
    "image_dir": "DataSets/Dataset_BUSI_with_GT/",
    "output_dir": "DataRes/Dataset_BUSI_with_GT/sam3_preds"
}

EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif")

# =========================
# 模型
# =========================
def load_model():
    predictor = SAM3SemanticPredictor(overrides=dict(
        conf=0.1,
        task="segment",
        mode="predict",
        model=CONFIG["model"],
        save=False,
    ))
    log.success("模型加载成功")
    return predictor

# =========================
# GT
# =========================
def load_gt_mask(img_path):
    base = os.path.splitext(img_path)[0]
    mask_path = base + "_mask.png"

    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return (mask > 0)
    return None

# =========================
# 推理
# =========================
def run_inference(predictor, img_path):
    predictor.set_image(img_path)
    results = predictor(text=["breast tumor"])

    if results[0].masks is None:
        return None, []

    masks = results[0].masks.data.cpu().numpy()
    pred_mask = masks.sum(axis=0) > 0

    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
        conf = results[0].boxes.conf.cpu().numpy().tolist()
    else:
        conf = [1.0] * masks.shape[0]

    return pred_mask, conf

# =========================
# CMP图（修复版）
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

    cmp = np.zeros((*pred.shape, 3), dtype=np.uint8)

    cmp[np.logical_and(pred, gt)] = [255, 255, 255]   # TP
    cmp[np.logical_and(pred, ~gt)] = [0, 0, 255]      # FP (红)
    cmp[np.logical_and(~pred, gt)] = [0, 255, 0]      # FN

    return cmp

# =========================
# 分割指标（完整）
# =========================
def calculate_segmentation_metrics(pred_mask, gt_mask):
    if pred_mask is None or gt_mask is None:
        return {}

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    if pred.shape != gt.shape:
        gt = cv2.resize(gt.astype(np.uint8),
                        (pred.shape[1], pred.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)

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
# Mask几何指标
# =========================
def calculate_mask_metrics(mask):
    if mask is None or np.sum(mask) == 0:
        return {}

    mask_uint8 = (mask.astype(np.uint8)) * 255

    area = np.sum(mask)
    total_pixels = mask.size
    area_percentage = (area / total_pixels) * 100

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {}

    largest = max(contours, key=cv2.contourArea)

    perimeter = cv2.arcLength(largest, True)

    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)

    convexity = area / (hull_area + 1e-6)

    x, y, w, h = cv2.boundingRect(largest)
    bbox_area = w * h
    extent = area / (bbox_area + 1e-6)

    M = cv2.moments(largest)
    cx = int(M["m10"] / (M["m00"] + 1e-6))
    cy = int(M["m01"] / (M["m00"] + 1e-6))

    return {
        "area": int(area),
        "area_percentage": round(area_percentage, 2),
        "perimeter": round(perimeter, 2),
        "convexity": round(convexity, 4),
        "extent": round(extent, 4),
        "bbox_area": int(bbox_area),
        "num_components": len(contours),
        "centroid_x": cx,
        "centroid_y": cy
    }

# =========================
# 单图处理
# =========================
def process_one(predictor, img_path, out_dir):
    fname = os.path.basename(img_path)

    # 获取子文件夹名称（类别）
    category = os.path.basename(os.path.dirname(img_path))

    log.info(f"处理: {fname}")

    start = time.time()

    gt = load_gt_mask(img_path)
    pred, conf = run_inference(predictor, img_path)

    cmp_img = generate_cmp(pred, gt)

    seg_metrics = calculate_segmentation_metrics(pred, gt)
    mask_metrics = calculate_mask_metrics(pred)

    if cmp_img is not None:
        cv2.imwrite(os.path.join(out_dir, f"cmp_{fname}"), cmp_img)

    return {
        "filename": fname,
        "category": category,
        "has_pred": pred is not None,
        "has_gt": gt is not None,
        "num_det": len(conf),
        "avg_conf": round(np.mean(conf), 4) if conf else 0,
        "time": round(time.time() - start, 3),
        **seg_metrics,
        **mask_metrics
    }

# =========================
# 主流程
# =========================
def collect_all_images(root_dir):
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(EXTENSIONS) and "_mask" not in fname:
                image_paths.append(os.path.join(root, fname))
    return image_paths

def main():
    predictor = load_model()

    image_dir = CONFIG["image_dir"]
    out_dir = CONFIG["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    results = []

    # 递归读取所有子文件夹图片
    all_images = collect_all_images(image_dir)

    log.info(f"共发现 {len(all_images)} 张图像")

    for path in all_images:
        res = process_one(predictor, path, out_dir)
        results.append(res)

    df = pd.DataFrame(results)

    csv_path = os.path.join(out_dir, "results_full.csv")
    df.to_csv(csv_path, index=False)

    json_path = os.path.join(out_dir, "results_full.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    valid = df[df["has_gt"] & df["has_pred"]]

    if len(valid) > 0:
        log.success("==== 总体指标 ====")
        log.info(f"Dice: {valid['dice'].mean():.4f}")
        log.info(f"IoU : {valid['iou'].mean():.4f}")
        log.info(f"HD95: {valid['hausdorff_distance_95'].mean():.2f}")
        log.info(f"ASD : {valid['average_surface_distance'].mean():.2f}")

    log.success(f"结果已保存: {out_dir}")

# =========================
if __name__ == "__main__":
    main()