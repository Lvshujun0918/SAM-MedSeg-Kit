import os
import cv2
import json
import time
import numpy as np
import pandas as pd
from pwn import log
from ultralytics import SAM

from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
from skimage import measure

# =========================
# 配置
# =========================
CONFIG = {
    # 切换为 SAM1 或 SAM2 模型 (确保对应 .pt 文件在 Models/ 目录下)
    # SAM1: "Models/sam_b.pt"
    # SAM2: "Models/sam2_b.pt"
    "model": "Models/sam2_l.pt", 
    "image_dir": "DataSets/Dataset_BUSI_with_GT/",
    "output_dir": "DataRes/Dataset_BUSI_with_GT/sam2_preds"
}

EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif")

# =========================
# 纯视觉算法提取 Prompt
# =========================
def extract_prompt_from_vision(img_path):
    """
    不依赖于文本提示或任何其他大模型，通过纯计算机视觉算法（形态学 + Otsu 阈值）
    自动提取目标候选区域的 bounding box，作为 SAM1/2 的输入 Prompt。
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    # 高斯滤波去噪
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 采用形态学梯度凸显边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
    
    # Otsu 自适应阈值二值化
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 闭运算符连接断裂边缘，填充空洞
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    
    # 提取轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 若无轮廓，Fallback: 取图像中心点
    if not contours:
        h, w = img.shape
        return {"points": [[w//2, h//2]]}
        
    # 假设最大轮廓即为病灶主体
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Box 外括一定的 Margin，保证 SAM 能接收到完整目标的上下文
    margin = 10
    img_h, img_w = img.shape
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_w, x + w + margin)
    y2 = min(img_h, y + h + margin)
    
    return {"bboxes": [x1, y1, x2, y2]}


# =========================
# 模型
# =========================
def load_model():
    # Ultralytics 对 SAM1/SAM2 使用相同的包接口
    model = SAM(CONFIG["model"])
    log.success(f"模型 {CONFIG['model']} 加载成功")
    return model

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
def run_inference(model, img_path):
    # 1. 自动提取 prompt
    prompt = extract_prompt_from_vision(img_path)
    if prompt is None:
        return None, [], None

    # 2. SAM1/SAM2 进行点框推理
    if "bboxes" in prompt:
        results = model(img_path, bboxes=prompt["bboxes"], verbose=False)
    else:
        results = model(img_path, points=prompt["points"], labels=[1], verbose=False)

    if len(results) == 0 or results[0].masks is None:
        return None, [], prompt

    # 3. 解析 masks 和 conf
    masks = results[0].masks.data.cpu().numpy()
    pred_mask = masks.sum(axis=0) > 0

    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
        conf = results[0].boxes.conf.cpu().numpy().tolist()
    else:
        conf = [1.0] * masks.shape[0]

    return pred_mask, conf, prompt

# =========================
# CMP图（修复版+提示框绘制）
# =========================
def generate_cmp(pred, gt, prompt=None):
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

    # 在图像上绘制所提取的 Prompt 提示框/点 (使用品红色)
    if prompt is not None:
        if "bboxes" in prompt:
            box = prompt["bboxes"]
            cv2.rectangle(cmp, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 2)
        elif "points" in prompt:
            pt = prompt["points"][0]
            cv2.circle(cmp, (int(pt[0]), int(pt[1])), 5, (255, 0, 255), -1)

    return cmp

# =========================
# 分割指标计算
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
# Mask 图几何指标计算
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
def process_one(model, img_path, out_dir):
    fname = os.path.basename(img_path)
    category = os.path.basename(os.path.dirname(img_path))

    log.info(f"处理: {fname}")
    start = time.time()

    gt = load_gt_mask(img_path)
    pred, conf, prompt = run_inference(model, img_path)

    cmp_img = generate_cmp(pred, gt, prompt)

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
    model = load_model()

    image_dir = CONFIG["image_dir"]
    out_dir = CONFIG["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    results = []
    all_images = collect_all_images(image_dir)

    log.info(f"共发现 {len(all_images)} 张图像")

    for path in all_images:
        res = process_one(model, path, out_dir)
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

if __name__ == "__main__":
    main()