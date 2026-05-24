import os
import cv2
import json
import time
import numpy as np
import pandas as pd
from pwn import log
from tqdm import tqdm
from ultralytics import SAM

from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
from skimage import measure

# =========================
# 配置
# =========================
CONFIG = {
    # SAM1: "Models/sam_b.pt"
    # SAM2: "Models/sam2_b.pt"
    "model": "Models/sam2_l.pt", 
    "image_dir": "DataSets/IDRiD_pro",
    "output_root": "DataRes/IDRiD_pro"
}

EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif")

# =========================
# 不同病灶配置
# =========================
TASKS = {
    "MA": {
        "mask_suffix": "_mask.png",
        "output_dir": "sam2_MA_preds"
    },
    "HE": {
        "mask_suffix": "_mask.png",
        "output_dir": "sam2_HE_preds"
    },
    "EX": {
        "mask_suffix": "_mask.png",
        "output_dir": "sam2_EX_preds"
    },
    "SE": {
        "mask_suffix": "_mask.png",
        "output_dir": "sam2_SE_preds"
    },
    "OD": {
        "mask_suffix": "_mask.png",
        "output_dir": "sam2_OD_preds"
    }
}

# =========================
# 纯视觉算法提取 Prompt
# =========================
def extract_prompt_from_vision(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        h, w = img.shape
        return {"points": [[w//2, h//2]]}
        
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
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
    model = SAM(CONFIG["model"])
    log.success(f"模型 {CONFIG['model']} 加载成功")
    return model

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
def run_inference(model, img_path):
    prompt = extract_prompt_from_vision(img_path)
    if prompt is None:
        return None, [], None

    if "bboxes" in prompt:
        results = model(img_path, bboxes=prompt["bboxes"], verbose=False)
    else:
        results = model(img_path, points=prompt["points"], labels=[1], verbose=False)

    if len(results) == 0 or results[0].masks is None:
        return None, [], prompt

    masks = results[0].masks.data.cpu().numpy()
    pred_mask = masks.sum(axis=0) > 0

    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
        conf = results[0].boxes.conf.cpu().numpy().tolist()
    else:
        conf = [1.0] * masks.shape[0]

    return pred_mask, conf, prompt

# =========================
# CMP图
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

    cmp_img = np.zeros((*pred.shape, 3), dtype=np.uint8)

    cmp_img[np.logical_and(pred, gt)] = [255, 255, 255]   # TP
    cmp_img[np.logical_and(pred, ~gt)] = [0, 0, 255]      # FP 
    cmp_img[np.logical_and(~pred, gt)] = [0, 255, 0]      # FN 

    if prompt is not None:
        if "bboxes" in prompt:
            box = prompt["bboxes"]
            cv2.rectangle(cmp_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 2)
        elif "points" in prompt:
            pt = prompt["points"][0]
            cv2.circle(cmp_img, (int(pt[0]), int(pt[1])), 5, (255, 0, 255), -1)

    return cmp_img

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
        model,
        img_path,
        out_dir,
        mask_suffix
):
    rel_path = os.path.relpath(img_path, CONFIG["image_dir"])
    unique_name = rel_path.replace(os.sep, "_")
    start = time.time()
    gt = load_gt_mask(img_path, mask_suffix)
    tqdm.write(f"[*] 开始处理{unique_name}")
    
    pred, conf, prompt = run_inference(model, img_path)

    cmp_img = generate_cmp(pred, gt, prompt)
    seg_metrics = calculate_segmentation_metrics(pred, gt)

    if cmp_img is not None:
        cmp_path = os.path.join(out_dir, f"cmp_{unique_name}")
        cv2.imwrite(cmp_path, cmp_img)
    else:
        tqdm.write("[!] 比较失败！")

    return {
        "filename": unique_name,
        "category": os.path.dirname(rel_path),
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
        model,
        task_name,
        task_cfg,
        all_images
):
    log.info(f"开始任务: {task_name}")

    out_dir = os.path.join(CONFIG["output_root"], task_cfg["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    results = []

    for img_path in tqdm(all_images, desc=f"{task_name}", ncols=100):
        res = process_one(
            model,
            img_path,
            out_dir,
            task_cfg["mask_suffix"]
        )
        results.append(res)

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "results_full.csv")
    df.to_csv(csv_path, index=False)

    json_path = os.path.join(out_dir, "results_full.json")
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
    model = load_model()
    for task_name, task_cfg in TASKS.items():
        log.success(f"开始执行{task_name} 任务")
        all_images = collect_all_images(os.path.join(CONFIG["image_dir"], task_name))
        log.info(f"共发现 {len(all_images)} 张图像")
        run_task(model, task_name, task_cfg, all_images)
        
    log.success("全部任务处理完成")

# =========================
if __name__ == "__main__":
    main()