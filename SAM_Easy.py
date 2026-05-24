"""
SAM_Easy.py - 初学者友好的 SAM1/2/3 统一接口

功能：
  - 统一 SAM1/2/3 的调用接口
  - 自动视觉提示提取（无需手工标注）
  - 自动计算 IoU, Dice, Hausdorff 等评估指标
  - 支持单张图像和批量处理
  - 自动生成处理报告

使用示例：
  from SAM_Easy import SAMSegmenter
  
  # 创建分割器
  segmenter = SAMSegmenter(model_type="SAM1", model_path="Models/sam_l.pt")
  
  # 分割单张图像
  result = segmenter.segment_image("image.png", use_auto_prompt=True)
  print(result)
  
  # 批量分割
  segmenter.segment_directory("DataSets/ACDC_pro", "DataRes/ACDC_pro/preds")
"""

import os
import cv2
import json
import glob
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from pwn import log

try:
    from ultralytics import SAM
except ImportError:
    print("请先安装: pip install ultralytics")
    exit(1)

try:
    from scipy.spatial.distance import directed_hausdorff
    from scipy.spatial import cKDTree
except ImportError:
    print("请先安装: pip install scipy")
    exit(1)

try:
    from skimage import measure
except ImportError:
    print("请先安装: pip install scikit-image")
    exit(1)

warnings.filterwarnings("ignore")

# ============================================================================
# 日志配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 核心类：SAMSegmenter
# ============================================================================
class SAMSegmenter:
    """统一的 SAM 分割接口，支持 SAM1/2/3"""

    def __init__(
        self,
        model_type: str = "SAM1",
        model_path: str = "Models/sam_l.pt",
        device: str = "cuda:0",
        conf: float = 0.25,
        iou_threshold: float = 0.7,
    ):
        """
        初始化 SAM 分割器

        参数:
            model_type (str): 模型类型，可选 "SAM1"、"SAM2"、"SAM3"
            model_path (str): 模型权重文件路径
            device (str): 计算设备，如 "cuda:0"、"cpu"
            conf (float): 预测置信度阈值 (0-1)
            iou_threshold (float): IoU 评估阈值 (0-1)
        """
        self.model_type = model_type.upper()
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.iou_threshold = iou_threshold

        # 检查模型文件
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}\n"
                f"请下载模型并放在 Models/ 目录下。"
            )

        # 加载模型
        log.info(f"🔄 正在加载 {model_type} 模型: {model_path}")
        try:
            self.model = SAM(model_path)
            self.model.to(device)
            log.success(f"✅ {model_type} 模型加载成功！")
        except Exception as e:
            log.error(f"❌ 模型加载失败: {e}")
            raise

        self.extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".nii.gz")

    # ========================================================================
    # 核心方法：自动提示提取（视觉算法）
    # ========================================================================
    def extract_bbox_prompt(self, img_path: str) -> Optional[np.ndarray]:
        """
        使用纯视觉算法（形态学 + Otsu 阈值）自动提取 Bounding Box 提示

        算法流程：
          1. 读取灰度图像
          2. 高斯滤波平滑
          3. 形态学梯度凸显边缘
          4. Otsu 自适应阈值二值化
          5. 闭运算填充空洞
          6. 轮廓检测并提取最大连通域
          7. 返回 bbox

        参数:
            img_path (str): 输入图像路径

        返回:
            np.ndarray: bbox 坐标 [x, y, w, h] 或 None（无有效目标）
        """
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                log.warn(f"⚠️  无法读取图像: {img_path}")
                return None

            # 高斯滤波去噪
            blur = cv2.GaussianBlur(img, (5, 5), 0)

            # 形态学梯度凸显边缘
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)

            # Otsu 自适应阈值二值化
            _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # 闭运算填充空洞
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

            # 轮廓检测
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                log.warn(f"⚠️  未检测到目标: {img_path}")
                return None

            # 提取最大轮廓
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)

            # 验证 bbox 大小（避免极小或极大的异常检测）
            img_area = img.shape[0] * img.shape[1]
            bbox_area = w * h
            if bbox_area < 0.01 * img_area or bbox_area > 0.99 * img_area:
                log.warn(f"⚠️  bbox 异常（面积={bbox_area/img_area:.2%}）: {img_path}")
                return None

            bbox = np.array([x, y, w, h], dtype=np.float32)
            return bbox

        except Exception as e:
            log.error(f"❌ 提示提取失败: {e}")
            return None

    # ========================================================================
    # 核心方法：图像分割
    # ========================================================================
    def segment_image(
        self,
        image_path: str,
        gt_path: Optional[str] = None,
        use_auto_prompt: bool = True,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        对单张图像进行分割

        参数:
            image_path (str): 输入图像路径
            gt_path (str): 真值标签路径（用于计算指标）
            use_auto_prompt (bool): 是否使用自动视觉提示
            output_path (str): 输出预测结果路径

        返回:
            dict: 包含预测掩膜和评估指标
                {
                    'mask': np.ndarray,  # 预测掩膜
                    'iou': float,         # IoU 指标
                    'dice': float,        # Dice 系数
                    'hausdorff': float,   # Hausdorff 距离
                    'sensitivity': float, # 灵敏度
                    'specificity': float, # 特异性
                }
        """
        try:
            # 读取图像
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                log.error(f"❌ 无法读取图像: {image_path}")
                return None

            # 生成提示
            if use_auto_prompt:
                bbox = self.extract_bbox_prompt(image_path)
                if bbox is None:
                    log.warn(f"⚠️  跳过无效图像: {image_path}")
                    return None
                bbox = bbox.reshape(1, -1)  # SAM 要求 (1, 4) 的形状
            else:
                bbox = None

            # SAM 分割
            with np.errstate(divide='ignore', invalid='ignore'):
                results = self.model.predict(
                    source=image_path,
                    conf=self.conf,
                    bboxes=bbox,
                    verbose=False,
                )

            if results is None or len(results) == 0:
                log.warn(f"⚠️  分割失败: {image_path}")
                return None

            # 提取预测掩膜
            result = results[0]
            if result.masks is not None:
                mask = result.masks.data[0].cpu().numpy()
                mask = (mask > 0.5).astype(np.uint8) * 255
            else:
                log.warn(f"⚠️  无掩膜输出: {image_path}")
                return None

            # 保存预测结果
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, mask)
                log.info(f"✅ 预测结果已保存: {output_path}")

            # 计算评估指标
            result_dict = {
                'mask': mask,
                'image_path': image_path,
            }

            if gt_path and os.path.exists(gt_path):
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt is not None:
                    gt = (gt > 128).astype(np.uint8)
                    mask_binary = (mask > 128).astype(np.uint8)

                    metrics = self.compute_metrics(mask_binary, gt)
                    result_dict.update(metrics)

            return result_dict

        except Exception as e:
            log.error(f"❌ 分割异常: {e}")
            return None

    # ========================================================================
    # 指标计算：IoU, Dice, Hausdorff, Sensitivity, Specificity
    # ========================================================================
    def compute_metrics(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> Dict[str, float]:
        """
        计算医学图像评估指标

        支持指标:
          - IoU (Intersection over Union)
          - Dice Coefficient
          - Hausdorff Distance
          - Sensitivity (True Positive Rate)
          - Specificity (True Negative Rate)

        参数:
            pred_mask (np.ndarray): 预测掩膜 (H, W)，值为 0/1 或 0/255
            gt_mask (np.ndarray): 真值掩膜 (H, W)，值为 0/1 或 0/255

        返回:
            dict: 所有指标值
        """
        try:
            # 确保二值化
            pred = (pred_mask > 128).astype(np.uint8)
            gt = (gt_mask > 128).astype(np.uint8)

            # 计算像素级别交集和并集
            intersection = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            tn = np.logical_and(~pred.astype(bool), ~gt.astype(bool)).sum()
            fp = np.logical_and(pred.astype(bool), ~gt.astype(bool)).sum()
            fn = np.logical_and(~pred.astype(bool), gt.astype(bool)).sum()

            # IoU
            iou = intersection / (union + 1e-8)

            # Dice
            dice = 2 * intersection / (pred.sum() + gt.sum() + 1e-8)

            # Hausdorff Distance
            hausdorff = self._compute_hausdorff_distance(pred, gt)

            # Sensitivity (True Positive Rate)
            sensitivity = intersection / (intersection + fn + 1e-8)

            # Specificity (True Negative Rate)
            specificity = tn / (tn + fp + 1e-8)

            return {
                'iou': float(iou),
                'dice': float(dice),
                'hausdorff': float(hausdorff),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
            }

        except Exception as e:
            log.error(f"❌ 指标计算失败: {e}")
            return {
                'iou': 0.0,
                'dice': 0.0,
                'hausdorff': np.inf,
                'sensitivity': 0.0,
                'specificity': 0.0,
            }

    @staticmethod
    def _compute_hausdorff_distance(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        计算两个掩膜之间的 Hausdorff 距离

        参数:
            mask1, mask2 (np.ndarray): 二值掩膜

        返回:
            float: Hausdorff 距离
        """
        try:
            # 提取轮廓点
            contours1 = measure.find_contours(mask1, 0.5)
            contours2 = measure.find_contours(mask2, 0.5)

            if len(contours1) == 0 or len(contours2) == 0:
                return np.inf

            points1 = np.vstack(contours1) if contours1 else np.array([[0, 0]])
            points2 = np.vstack(contours2) if contours2 else np.array([[0, 0]])

            # Hausdorff 距离
            d12 = directed_hausdorff(points1, points2)[0]
            d21 = directed_hausdorff(points2, points1)[0]
            hausdorff = max(d12, d21)

            return float(hausdorff)
        except Exception:
            return np.inf

    # ========================================================================
    # 批量处理
    # ========================================================================
    def segment_directory(
        self,
        input_dir: str,
        output_dir: str,
        gt_dir: Optional[str] = None,
        use_auto_prompt: bool = True,
        save_report: bool = True,
    ) -> pd.DataFrame:
        """
        批量分割图像目录中的所有图像

        参数:
            input_dir (str): 输入图像目录
            output_dir (str): 输出预测掩膜目录
            gt_dir (str): 真值标签目录（如有）
            use_auto_prompt (bool): 是否使用自动提示
            save_report (bool): 是否保存 CSV 报告

        返回:
            pd.DataFrame: 包含所有图像的处理结果和指标
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 查找所有图像
        image_files = []
        for ext in self.extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(input_dir, f"**/*{ext}"), recursive=True))

        image_files = sorted(set(image_files))  # 去重并排序

        if not image_files:
            log.error(f"❌ 未找到图像文件: {input_dir}")
            return None

        log.info(f"📊 找到 {len(image_files)} 个图像，开始处理...")

        results = []
        for idx, img_path in enumerate(tqdm(image_files, desc=f"{self.model_type} 分割进度")):
            # 生成输出路径
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(output_dir, f"{img_name}_mask.png")

            # 查找对应的真值标签
            gt_path = None
            if gt_dir:
                for ext in ("_mask.png", "_gt.png", "_label.png", ".png", ".jpg"):
                    potential_gt = os.path.join(gt_dir, img_name + ext)
                    if os.path.exists(potential_gt):
                        gt_path = potential_gt
                        break

            # 分割
            result = self.segment_image(
                image_path=img_path,
                gt_path=gt_path,
                use_auto_prompt=use_auto_prompt,
                output_path=output_path,
            )

            if result:
                results.append(result)

        # 转换为 DataFrame
        if results:
            df = pd.DataFrame(results)
            log.success(f"✅ 分割完成！处理了 {len(df)} 个图像")

            # 打印统计信息
            if 'iou' in df.columns:
                log.info(f"📈 IoU 统计:")
                log.info(f"   平均值: {df['iou'].mean():.4f}")
                log.info(f"   中位数: {df['iou'].median():.4f}")
                log.info(f"   范围: [{df['iou'].min():.4f}, {df['iou'].max():.4f}]")

            # 保存报告
            if save_report:
                report_path = os.path.join(output_dir, "metrics_report.csv")
                df.to_csv(report_path, index=False)
                log.success(f"✅ 报告已保存: {report_path}")

            return df
        else:
            log.error(f"❌ 所有图像处理失败")
            return None


# ============================================================================
# 示例使用
# ============================================================================
if __name__ == "__main__":
    """
    使用示例：

    1. 单张图像分割
    2. 批量目录分割
    3. 生成对比报告
    """

    # ========== 示例 1: 单张图像分割 ==========
    print("\n" + "="*60)
    print("示例 1: 单张图像分割")
    print("="*60)

    segmenter = SAMSegmenter(
        model_type="SAM1",
        model_path="Models/sam_l.pt",
        device="cuda:0",
    )

    result = segmenter.segment_image(
        image_path="DataSets/ACDC_pro/case_001_frame.png",
        gt_path="DataSets/ACDC_pro/case_001_frame_mask.png",
        use_auto_prompt=True,
        output_path="sample_output.png",
    )

    if result:
        log.info(f"✅ 分割成功！")
        if 'iou' in result:
            log.info(f"   IoU: {result['iou']:.4f}")
            log.info(f"   Dice: {result['dice']:.4f}")

    # ========== 示例 2: 批量处理 ==========
    print("\n" + "="*60)
    print("示例 2: 批量处理目录")
    print("="*60)

    df = segmenter.segment_directory(
        input_dir="DataSets/ACDC_pro",
        output_dir="DataRes/ACDC_pro/sam1_easy_results",
        gt_dir="DataSets/ACDC_pro",
        use_auto_prompt=True,
        save_report=True,
    )

    if df is not None:
        print("\n" + "="*60)
        print("处理结果摘要")
        print("="*60)
        print(df[['image_path', 'iou', 'dice', 'hausdorff']].head(10))

    print("\n" + "="*60)
    print("示例执行完毕！")
    print("="*60)
