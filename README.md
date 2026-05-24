# From SAM 1 to SAM 3: Benchmarking Cross-Modality Medical Image Segmentation

## 📖 项目概述

这是一个关于 **零样本设定下 SAM1/2/3 在医学图像分割中的跨模态演化评估** 的完整实验框架。项目实现了基于 Segment Anything Model 系列的医学图像分割管线，支持多个公开医学影像数据集。

**主要特点：**
- 支持 SAM1、SAM2、SAM3 三个版本
- 集成多个医学数据集（ACDC、BUSI、IDRiD、Synapse）
- 自动化的数据预处理流程
- 内置视觉提示自动提取（无需手工标注）
- 完整的评估指标计算
- 高质量图表生成

## 📁 项目结构

```
.
├── README.md                           # 本文件
├── SAM_Easy.py                         # ⭐ 初学者友好的 SAM 统一接口
├── 
├── 数据预处理模块
├── ├── ACDC_data_preprocess.py         # ACDC 数据集预处理
├── ├── Synapse_data_preprocess.py      # Synapse 数据集预处理
├── ├── BUSI_data_preprocess.py         # BUSI 数据集预处理
├── ├── IDRiD_data_preprocess.py        # IDRiD 数据集预处理
├── 
├── SAM 分割模块
├── ├── ACDC_data_sam1_seg.py           # ACDC 使用 SAM1 分割
├── ├── ACDC_data_sam2_seg.py           # ACDC 使用 SAM2 分割
├── ├── ACDC_data_sam3_seg.py           # ACDC 使用 SAM3 分割
├── ├── BUSI_data_sam1_seg.py           # BUSI 使用 SAM1 分割
├── ├── BUSI_data_sam2_seg.py           # BUSI 使用 SAM2 分割
├── ├── BUSI_data_sam3_seg.py           # BUSI 使用 SAM3 分割
├── ├── IDRiD_data_sam1_seg.py          # IDRiD 使用 SAM1 分割
├── ├── IDRiD_data_sam2_seg.py          # IDRiD 使用 SAM2 分割
├── ├── IDRiD_data_sam3_seg.py          # IDRiD 使用 SAM3 分割
├── ├── Synapse_data_sam1_seg.py        # Synapse 使用 SAM1 分割
├── ├── Synapse_data_sam2_seg.py        # Synapse 使用 SAM2 分割
├── └── Synapse_data_sam3_seg.py        # Synapse 使用 SAM3 分割
├── 
└── fig/                                # 图表生成模块
    ├── fig_cmp.py                      # 模型对比图表
    ├── fig_sam1.py                     # SAM1 性能图表
    ├── fig_sam2.py                     # SAM2 性能图表
    ├── fig_sam3.py                     # SAM3 性能图表
    ├── gen_dataset_figures.py          # 数据集可视化
    ├── measure_params_flops.py         # 参数/FLOPS 计算
    └── paper_figures_and_verify.py     # 论文验证与补充图表
```


## 🚀 快速开始

### 1️⃣ 环境配置

```bash
# 创建虚拟环境
conda create -n sam-med python=3.10
conda activate sam-med

# 安装核心依赖
pip install -r requirements.txt
```

**关键依赖：**
- `ultralytics>=8.0.0` - SAM 模型封装
- `opencv-python` - 图像处理
- `nibabel` - 医学影像格式支持 (.nii.gz)
- `scikit-image` - 形态学算法
- `pandas` - 数据分析
- `matplotlib`, `seaborn` - 可视化

### 2️⃣ 下载模型权重

在项目根目录创建 `Models/` 文件夹，下载以下权重：

| 模型 | 文件名 | 大小 | 来源 |
|------|--------|------|------|
| SAM1 Large | `sam_l.pt` | 1.2 GB | [Meta AI](https://github.com/facebookresearch/segment-anything) |
| SAM2 Large | `sam2_l.pt` | 429 MB | [Meta AI](https://github.com/facebookresearch/sam2) |
| SAM3 | `sam3.pt` | 3.3 GB | [Meta AI](https://github.com/facebookresearch/segment-anything-3) |

目录结构：
```
Models/
├── sam2_l.pt
├── sam3.pt
└── sam_l.pt
```

### 3️⃣ 准备数据

假设原始数据已下载到 `DataSets/` 目录：

```bash
# 以 ACDC 为例
python ACDC_data_preprocess.py

# 预处理后的输出在 DataSets/ACDC_pro/
```

### 4️⃣ 运行分割

**方式一：使用初学者友好的 `SAM_Easy.py`（推荐）**

```python
from SAM_Easy import SAMSegmenter

# 创建分割器实例
segmenter = SAMSegmenter(
    model_type="SAM1",           # 可选: SAM1, SAM2, SAM3
    model_path="Models/sam_l.pt"  # 模型权重路径
)

# 对单张图像分割
result = segmenter.segment_image(
    image_path="path/to/image.png",
    use_auto_prompt=True,  # 自动提取视觉提示
    output_path="output.png"
)
print(f"IoU: {result['iou']:.4f}")

# 批量分割图像目录
segmenter.segment_directory(
    input_dir="DataSets/ACDC_pro",
    output_dir="DataRes/ACDC_pro/predictions",
    gt_dir="DataSets/ACDC_pro"  # 用于计算评估指标
)
```

**方式二：直接运行原始脚本**

```bash
# 使用 SAM1 分割 ACDC 数据集
python ACDC_data_sam1_seg.py

# 使用 SAM2 分割 BUSI 数据集
python BUSI_data_sam2_seg.py

# 使用 SAM3 分割 Synapse 数据集
python Synapse_data_sam3_seg.py
```


## 📊 数据集支持

| 数据集 | 模态 | 器官/病变 | 图像数 | 脚本文件 |
|--------|------|---------|--------|---------|
| **ACDC** | 心脏 CT | 心室分割 (RV/MYO/LV) | ~1000 | `ACDC_data_*_seg.py` |
| **BUSI** | 乳腺超声 | 肿瘤检测 | ~780 | `BUSI_data_*_seg.py` |
| **IDRiD** | 视网膜 RGB | 糖尿病病变检测 | ~80 | `IDRiD_data_*_seg.py` |
| **Synapse** | 腹部 CT | 多器官分割 | ~30 | `Synapse_data_*_seg.py` |


## 🧠 核心功能说明

### 自动提示提取（Vision-Only）

无需额外标注，系统通过计算机视觉算法自动生成 Bounding Box 提示：

```python
def extract_prompt_from_vision(img_path):
    """
    使用形态学 + Otsu 阈值自动提取目标区域
    """
    # 高斯滤波 → 形态学梯度 → Otsu 二值化 → 轮廓检测
    # 返回: bbox (x, y, w, h) or None
```

### 评估指标

支持的评估指标：
- **IoU (Intersection over Union)** - 主要指标
- **Dice Coefficient** - 医学影像标准
- **Hausdorff Distance** - 轮廓相似度
- **Sensitivity / Specificity** - 临床指标

```python
metrics = segmenter.compute_metrics(
    prediction_mask=pred,
    ground_truth_mask=gt
)
# {'iou': 0.85, 'dice': 0.92, 'hausdorff': 15.3, ...}
```

### 批量处理

自动化处理大量图像并生成报告：

```python
results_df = segmenter.segment_directory(
    input_dir="DataSets/ACDC_pro",
    output_dir="DataRes/ACDC_results",
    gt_dir="DataSets/ACDC_pro",
    save_report=True  # 生成 CSV 报告
)
# 输出: results_df (包含所有图像的 IoU, Dice 等指标)
```


## 📈 图表生成

所有图表保存在 `Fig/` 目录，支持多种格式（PNG/SVG/PDF）：

```bash
# 生成论文所需的所有图表
python fig/paper_figures_and_verify.py

# 生成特定模型的性能图表
python fig/fig_sam1.py
python fig/fig_sam2.py
python fig/fig_sam3.py

# 生成模型对比图表
python fig/fig_cmp.py

# 计算参数量和 FLOPS
python fig/measure_params_flops.py
```

生成的图表包括：
- 各模型在不同数据集上的 IoU 对比
- Dice/Hausdorff 分布箱线图
- 器官/病变级别的细粒度评估
- 参数量与性能权衡


## 📝 常见工作流

### 工作流 1：快速评估新数据集

```bash
# 1. 预处理
python {DATASET}_data_preprocess.py

# 2. 分割（使用最强模型 SAM3）
python {DATASET}_data_sam3_seg.py

# 3. 查看结果
# → DataRes/{DATASET}_pro/sam3_preds/ 中有预测结果
# → CSV 报告中有详细指标
```

### 工作流 2：三模型对比

```bash
# 运行三个版本
python ACDC_data_sam1_seg.py
python ACDC_data_sam2_seg.py
python ACDC_data_sam3_seg.py

# 生成对比图表
python fig/fig_cmp.py
```

### 工作流 3：从头开始（完整流程）

```bash
# 1. 数据预处理
python ACDC_data_preprocess.py
python BUSI_data_preprocess.py
python IDRiD_data_preprocess.py
python Synapse_data_sam1_seg.py

# 2. 使用 SAM1 分割各数据集
for dataset in ACDC BUSI IDRiD Synapse; do
    python ${dataset}_data_sam1_seg.py
done

# 3. 使用 SAM2/3 进行升级
python ACDC_data_sam2_seg.py
python ACDC_data_sam3_seg.py
# ... 其他数据集

# 4. 生成完整报告和图表
python fig/paper_figures_and_verify.py
```


## 🔧 配置调整

所有脚本的配置在 `CONFIG` 字典中，可按需修改：

```python
CONFIG = {
    "model": "Models/sam_l.pt",  # 模型权重路径
    "image_dir": "DataSets/ACDC_pro",  # 输入目录
    "output_dir": "DataRes/ACDC_pro/sam1_preds",  # 输出目录
    "batch_size": 4,  # GPU 批处理大小
    "device": "cuda:0",  # GPU ID
}
```

**常用调整：**
| 配置项 | 用途 | 建议值 |
|--------|------|--------|
| `model` | 模型权重 | `sam_b.pt` (快) 或 `sam_l.pt` (强) |
| `batch_size` | GPU 并行 | 4-8 (取决于 GPU 显存) |
| `device` | 计算设备 | `cuda:0` (GPU) 或 `cpu` (CPU) |
| `iou_threshold` | 预测置信度 | 0.5-0.9 |


## ⚙️ 依赖版本参考

```
ultralytics>=8.0.0
torch>=2.0.0,<3
torchvision>=0.15.0
opencv-python>=4.8.0
nibabel>=5.1.0
scikit-image>=0.21.0
scipy>=1.11.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
```


## 📊 输出结构

运行完成后，输出目录结构如下：

```
DataRes/ACDC_pro/sam1_preds/
├── case_001_mask.png           # 预测分割掩膜
├── case_002_mask.png
├── ...
└── metrics_report.csv          # 评估报告
    ├── image_id | IoU | Dice | Hausdorff | Sensitivity | Specificity
    └── ...

Fig/
├── fig_sam1_iou_comparison.png
├── fig_sam2_dice_boxplot.png
├── fig_cmp_all_models.png
└── params_flops_analysis.pdf
```


## 🎓 对初学者的建议

1. **从 `SAM_Easy.py` 开始**
   - 这是最简单的接口，适合快速原型开发
   - 支持自动提示提取，无需手工标注

2. **先用 SAM1（速度最快）**
   - 快速验证流程是否正确
   - 然后升级到 SAM2/3 获得更好效果

3. **查看日志输出**
   - 所有脚本都使用 `pwn.log` 打印进度
   - 遇到问题时，查看错误日志定位原因


## 📬 常见问题

**Q1: 模型下载太慢？**
A: 推荐使用国内镜像或代理，或者提前下载并放在 `Models/` 目录下。

**Q2: GPU 显存不足？**
A: 修改 `batch_size` 为 1-2，或使用 `sam_b.pt`（较小模型）。

**Q3: 预测效果差？**
A: 
- 检查输入图像预处理是否正确（归一化范围）
- 尝试升级到 SAM2/3
- 调整 `iou_threshold` 参数

**Q4: 如何使用 CPU？**
A: 设置 `CONFIG["device"] = "cpu"`，但会明显变慢。

## 🤝 反馈与支持

如有问题、建议或需要技术支持，请：
- 查阅本 README 的常见问题部分
- 检查脚本输出的日志信息
- 参考原始数据论文和 SAM 官方文档

**更新日志：**
- v1.0 (2026-05-16): 初始发布，支持 SAM1/2/3 三个版本

---

**祝你使用愉快！** 🎉
