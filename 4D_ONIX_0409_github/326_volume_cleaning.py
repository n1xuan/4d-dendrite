"""
================================================================================
Tomoscopy 体数据伪影去除 Pipeline (工程级)
================================================================================

适用数据:
    AG10合金 Tomoscopy 凝固数据, 300帧时间序列
    原始格式: (280, 528, 528) uint16 TIFF stacks
    已知伪影: 环形伪影(ring artifacts), 容器壁, 空气背景, 边缘伪影,
              帧间轻微位移, 不均匀照明(beam hardening)

处理目标:
    输出干净的 float32 体数据, 样品内部保留连续灰度信息(用于DRR投影),
    样品外部置零, 伪影尽可能消除.

输出模式:
    MODE_DIFF:          target - liquid_ref (推荐, 环形伪影自动抵消)
    MODE_DIFF_POSITIVE: max(0, -(target - liquid_ref))  (纯枝晶信号)
    MODE_RAW_CLEAN:     样品内原始灰度, 外部置零 (保留全部信息)
    MODE_ALL:           同时输出以上三种

依赖:
    numpy, scipy, scikit-image, tifffile, matplotlib

使用:
    # 处理单帧 (测试)
    python volume_cleaning.py

    # 处理指定帧
    python volume_cleaning.py --frame 01650

    # 处理全部300帧
    python volume_cleaning.py --all

    # 指定输出模式
    python volume_cleaning.py --mode diff --all

作者: 协助 dh524 的4D-ONIX重建项目
================================================================================
"""

import numpy as np
import tifffile
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from skimage import measure, morphology
import matplotlib.pyplot as plt
import os
import sys
import time
import logging
import argparse
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 常量定义
# ============================================================
MODE_DIFF = 'diff'
MODE_DIFF_POSITIVE = 'diff_positive'
MODE_RAW_CLEAN = 'raw_clean'
MODE_ALL = 'all'

VALID_MODES = [MODE_DIFF, MODE_DIFF_POSITIVE, MODE_RAW_CLEAN, MODE_ALL]


# ============================================================
# 配置类
# ============================================================
class CleaningConfig:
    """体数据清理配置参数, 集中管理便于复现"""

    def __init__(self):
        # --- 路径 ---
        self.timesteps_dir = '/home/dh524/data/dendrite_tiff/timesteps'
        self.output_base = '/home/dh524/data/dendrite_tiff/clean_volumes'

        # --- 液相参考帧 ---
        self.liquid_avg_n_frames = 5        # 取前N帧平均作为液相参考
        self.liquid_smooth_sigma = 0.3      # 液相参考额外平滑 (降低残留噪声)

        # --- 样品掩膜 ---
        self.sample_threshold = 25000       # 灰度阈值: 高于此值 = 样品区域
        self.mask_erosion_pixels = 3        # 边缘腐蚀: 去除容器壁残留和边缘伪影
        self.mask_smooth_sigma = 1.0        # 掩膜平滑: 消除锯齿边缘
        self.mask_reuse = True              # 是否复用同一掩膜(加速, 适用于样品不移动的情况)

        # --- 环形伪影校正 ---
        self.ring_correction = True         # 是否做环形伪影校正
        self.ring_median_width = 51         # 径向中值滤波宽度 (奇数)
        self.ring_correction_strength = 0.8 # 校正强度 [0, 1], 1=完全去除

        # --- 径向不均匀性校正 (beam hardening) ---
        self.radial_correction = True       # 是否做径向补偿
        self.radial_n_bins = 30             # 径向分区数
        self.radial_correction_range = (0.80, 1.20)  # 补偿系数裁剪范围

        # --- 差分预处理 ---
        self.diff_smooth_sigma = 0.5        # 差分图平滑 (去高频噪声, 保留结构)
        self.diff_clip_percentile = 99.5    # 差分值裁剪百分位 (去极端异常值)

        # --- 原始数据清理 ---
        self.raw_smooth_sigma = 0.5         # 原始数据轻度平滑
        self.raw_normalize_percentile = (1, 99)  # 归一化百分位范围

        # --- 输出 ---
        self.output_dtype = np.float32      # 输出精度
        self.output_mode = MODE_DIFF        # 默认输出模式
        self.save_visualization = True      # 是否保存验证图
        self.save_npy = True                # 是否额外保存.npy格式

        # --- 帧范围 ---
        self.frame_start = 1680    # 起始帧编号
        self.frame_end = 1800      # 结束帧编号


# ============================================================
# 日志配置
# ============================================================
def setup_logger(output_dir):
    """配置日志: 同时输出到终端和文件"""
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger('volume_cleaning')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 终端输出
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    # 文件输出
    fh = logging.FileHandler(os.path.join(output_dir, 'cleaning_log.txt'), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    return logger


# ============================================================
# 核心处理模块
# ============================================================

class LiquidReferenceBuilder:
    """
    液相参考帧构建器
    
    策略: 取时间序列最早的N帧做平均, 因为这些帧在凝固开始前,
    样品处于纯液态, 灰度分布均匀. 平均可以有效降低随机噪声,
    同时保留系统性伪影模式(环形伪影等), 使得后续差分能抵消这些伪影.
    """

    def __init__(self, config: CleaningConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._reference = None

    def build(self) -> np.ndarray:
        """构建液相参考帧"""
        self.logger.info("=" * 60)
        self.logger.info("[液相参考] 构建中...")

        all_files = self._list_tiff_files()
        n_avg = min(self.config.liquid_avg_n_frames, len(all_files))
        self.logger.info(f"  使用前 {n_avg} 帧平均")

        # 逐帧累加 (避免一次加载所有帧爆内存)
        accumulator = None
        for i in range(n_avg):
            fpath = os.path.join(self.config.timesteps_dir, all_files[i])
            frame = tifffile.imread(fpath).astype(np.float64)

            if accumulator is None:
                accumulator = frame
                self.logger.info(f"    帧 {all_files[i]}: shape={frame.shape}, "
                                 f"dtype={frame.dtype}")
            else:
                accumulator += frame

            # 检查帧间一致性 (shape, 数值范围)
            if i == 0:
                ref_shape = frame.shape
                ref_mean = frame.mean()
            else:
                assert frame.shape == ref_shape, \
                    f"帧 {all_files[i]} shape {frame.shape} != 参考 {ref_shape}"
                mean_diff = abs(frame.mean() - ref_mean) / ref_mean
                if mean_diff > 0.1:
                    self.logger.warning(f"    警告: 帧 {all_files[i]} 均值偏差 "
                                        f"{mean_diff:.1%} (可能非液相帧)")

        # 平均
        self._reference = (accumulator / n_avg).astype(np.float32)

        # 可选: 额外轻度平滑
        if self.config.liquid_smooth_sigma > 0:
            self._reference = gaussian_filter(
                self._reference, sigma=self.config.liquid_smooth_sigma)
            self.logger.info(f"  额外平滑: sigma={self.config.liquid_smooth_sigma}")

        self.logger.info(f"  参考帧统计: mean={self._reference.mean():.1f}, "
                         f"std={self._reference.std():.1f}, "
                         f"range=[{self._reference.min():.0f}, {self._reference.max():.0f}]")

        return self._reference

    @property
    def reference(self) -> np.ndarray:
        if self._reference is None:
            raise RuntimeError("液相参考尚未构建, 请先调用 build()")
        return self._reference

    def _list_tiff_files(self):
        """列出时间步tiff文件, 按文件名排序"""
        files = [f for f in sorted(os.listdir(self.config.timesteps_dir))
                 if f.endswith('.tif') and not os.path.isdir(
                     os.path.join(self.config.timesteps_dir, f))]
        if len(files) == 0:
            raise FileNotFoundError(f"在 {self.config.timesteps_dir} 中未找到tiff文件")
        self.logger.info(f"  找到 {len(files)} 帧: {files[0]} ... {files[-1]}")
        return files


class SampleMaskExtractor:
    """
    样品区域掩膜提取器

    策略: 对每个Z切片做灰度阈值 → 填充孔洞 → 保留最大连通域
    → 腐蚀去除边缘 → 可选平滑. 同时计算样品的几何参数
    (中心坐标, 半径) 用于后续径向校正.

    注意: 凝固过程中样品几何基本不变, 因此掩膜可以复用.
    但如果样品有明显位移/变形, 应为每帧单独计算.
    """

    def __init__(self, config: CleaningConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._mask = None
        self._center_y = None
        self._center_x = None
        self._radius = None

    def extract(self, volume: np.ndarray) -> np.ndarray:
        """
        从体数据中提取样品区域掩膜

        Parameters
        ----------
        volume : np.ndarray, shape (D, H, W)
            3D体数据 (可以是原始数据或平滑后的数据)

        Returns
        -------
        mask : np.ndarray, shape (D, H, W), dtype bool
        """
        self.logger.info("=" * 60)
        self.logger.info("[样品掩膜] 提取中...")

        D, H, W = volume.shape
        mask = np.zeros((D, H, W), dtype=bool)

        # 对输入做轻度平滑, 减少噪声对阈值的干扰
        vol_smooth = gaussian_filter(volume, sigma=self.config.mask_smooth_sigma)

        # 逐切片处理
        centers_y = []
        centers_x = []
        radii = []

        for z in range(D):
            sl = vol_smooth[z]

            # 步骤1: 灰度阈值
            binary = sl > self.config.sample_threshold

            # 步骤2: 填充孔洞 (样品内部可能有低灰度区域)
            binary = ndimage.binary_fill_holes(binary)

            # 步骤3: 连通域分析, 只保留最大的 (即圆形样品)
            labeled = measure.label(binary)
            if labeled.max() == 0:
                continue

            regions = measure.regionprops(labeled)
            largest = max(regions, key=lambda r: r.area)
            binary = (labeled == largest.label)

            # 步骤4: 腐蚀边缘
            if self.config.mask_erosion_pixels > 0:
                binary = ndimage.binary_erosion(
                    binary, iterations=self.config.mask_erosion_pixels)

            mask[z] = binary

            # 记录几何参数 (中间层)
            if binary.any():
                props = measure.regionprops(binary.astype(np.uint8))
                if props:
                    cy, cx = props[0].centroid
                    r = np.sqrt(props[0].area / np.pi)
                    centers_y.append(cy)
                    centers_x.append(cx)
                    radii.append(r)

        # 汇总几何参数 (取中位数, 鲁棒)
        if len(centers_y) > 0:
            self._center_y = int(np.median(centers_y))
            self._center_x = int(np.median(centers_x))
            self._radius = int(np.median(radii))
        else:
            raise ValueError("未能检测到样品区域, 请检查 sample_threshold 设置")

        self._mask = mask
        sample_fraction = mask.sum() / mask.size * 100

        self.logger.info(f"  样品中心: y={self._center_y}, x={self._center_x}")
        self.logger.info(f"  样品半径: ≈{self._radius} pixels")
        self.logger.info(f"  掩膜体素: {mask.sum():,} ({sample_fraction:.1f}% of total)")

        # 一致性检查
        std_y = np.std(centers_y) if len(centers_y) > 1 else 0
        std_x = np.std(centers_x) if len(centers_x) > 1 else 0
        if std_y > 5 or std_x > 5:
            self.logger.warning(f"  警告: 样品中心在Z方向有波动 "
                                f"(std_y={std_y:.1f}, std_x={std_x:.1f}), "
                                f"可能存在样品位移")

        return mask

    @property
    def mask(self) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("掩膜尚未提取, 请先调用 extract()")
        return self._mask

    @property
    def center(self):
        return self._center_y, self._center_x

    @property
    def radius(self):
        return self._radius


class RingArtifactCorrector:
    """
    环形伪影校正器

    原理: 环形伪影在sinogram空间中是垂直条纹, 在重建后的CT切片中
    表现为以旋转中心为圆心的同心环. 在极坐标下, 这些环变成水平条纹,
    可以用沿角度方向(θ)的中值滤波检测并减去.

    方法:
      1. 将每个切片从笛卡尔坐标转换到极坐标 (r, θ)
      2. 沿θ方向做中值滤波, 得到"只有环的图像"
      3. 从原图减去环模式
      4. 转换回笛卡尔坐标

    注: 如果使用差分方法(target - liquid), 环形伪影会自然抵消,
    此校正主要用于 raw_clean 模式.
    """

    def __init__(self, config: CleaningConfig, logger: logging.Logger,
                 center_y: int, center_x: int):
        self.config = config
        self.logger = logger
        self.center_y = center_y
        self.center_x = center_x

    def correct(self, volume: np.ndarray, sample_mask: np.ndarray) -> np.ndarray:
        """
        对体数据做环形伪影校正

        Parameters
        ----------
        volume : np.ndarray, shape (D, H, W), float32
        sample_mask : np.ndarray, shape (D, H, W), bool

        Returns
        -------
        corrected : np.ndarray, same shape and dtype
        """
        if not self.config.ring_correction:
            return volume

        self.logger.info("=" * 60)
        self.logger.info("[环形伪影校正] 处理中...")

        D, H, W = volume.shape
        corrected = volume.copy()

        # 预计算极坐标映射 (对所有切片复用)
        Y, X = np.mgrid[:H, :W]
        Y_centered = Y - self.center_y
        X_centered = X - self.center_x
        R = np.sqrt(Y_centered**2 + X_centered**2)
        Theta = np.arctan2(Y_centered, X_centered)  # [-pi, pi]

        max_r = int(np.sqrt(H**2 + W**2) / 2) + 1
        n_theta = 360  # 角度采样数

        # 极坐标网格
        r_grid = np.arange(0, max_r)
        theta_grid = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)

        processed = 0
        for z in range(D):
            if not sample_mask[z].any():
                continue

            sl = volume[z]

            # 转极坐标
            from scipy.interpolate import RegularGridInterpolator
            # 构建插值器
            y_coords = np.arange(H)
            x_coords = np.arange(W)
            interp = RegularGridInterpolator(
                (y_coords, x_coords), sl, method='linear',
                bounds_error=False, fill_value=0)

            # 在极坐标网格上采样
            RR, TT = np.meshgrid(r_grid, theta_grid, indexing='ij')
            Y_polar = self.center_y + RR * np.sin(TT)
            X_polar = self.center_x + RR * np.cos(TT)
            points = np.stack([Y_polar.ravel(), X_polar.ravel()], axis=-1)
            polar = interp(points).reshape(RR.shape)

            # 沿θ方向中值滤波 → 检测环模式
            ring_pattern = median_filter(polar, size=(1, self.config.ring_median_width))

            # 环模式 = 中值 - 原图 沿θ的均值
            # 实际上直接用中值滤波结果做减法
            ring_only = ring_pattern - polar.mean(axis=1, keepdims=True)

            # 从极坐标转回笛卡尔: 对每个像素查找对应的ring值
            r_img = R.astype(int)
            r_img = np.clip(r_img, 0, max_r - 1)

            # 环的校正量只依赖r (沿θ平均)
            ring_radial = ring_only.mean(axis=1)  # shape (max_r,)
            correction_2d = ring_radial[r_img]

            # 应用校正
            corrected[z] = sl - correction_2d * self.config.ring_correction_strength

            processed += 1

        self.logger.info(f"  处理了 {processed} 个切片")
        self.logger.info(f"  校正强度: {self.config.ring_correction_strength}")

        return corrected


class RadialNonuniformityCorrector:
    """
    径向不均匀性校正器 (Beam Hardening Correction)

    原理: 同步辐射X射线穿过圆柱形样品时, 边缘路径短、中心路径长,
    加上多色X射线的beam hardening效应, 导致样品中心和边缘的重建灰度
    存在系统性差异. 表现为从中心到边缘的径向灰度渐变.

    方法:
      1. 对样品内部计算径向平均灰度曲线 f(r)
      2. 以中心灰度为基准, 计算校正因子 c(r) = f(0) / f(r)
      3. 对每个体素: corrected = original * c(r)
      4. 裁剪校正因子到合理范围, 避免极端值
    """

    def __init__(self, config: CleaningConfig, logger: logging.Logger,
                 center_y: int, center_x: int, radius: int):
        self.config = config
        self.logger = logger
        self.center_y = center_y
        self.center_x = center_x
        self.radius = radius
        self._correction_map = None

    def compute_correction_map(self, volume: np.ndarray,
                                sample_mask: np.ndarray) -> np.ndarray:
        """
        从体数据中计算径向校正图

        使用中间层切片估计径向曲线 (最具代表性)
        """
        self.logger.info("=" * 60)
        self.logger.info("[径向校正] 计算校正图...")

        D, H, W = volume.shape

        # 构建距离图
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((Y - self.center_y)**2 + (X - self.center_x)**2)
        dist_norm = dist / self.radius  # 归一化: 0=中心, 1=边缘

        # 取多个Z层估计径向曲线 (更鲁棒)
        z_samples = [D // 4, D // 2, D * 3 // 4]
        all_radial_means = []

        n_bins = self.config.radial_n_bins

        for z in z_samples:
            if not sample_mask[z].any():
                continue

            sl = volume[z]
            mask_sl = sample_mask[z]

            radial_means = np.zeros(n_bins)
            for i in range(n_bins):
                r_low = i / n_bins
                r_high = (i + 1) / n_bins
                ring = mask_sl & (dist_norm >= r_low) & (dist_norm < r_high)
                if ring.sum() > 50:
                    radial_means[i] = sl[ring].mean()
                else:
                    radial_means[i] = np.nan

            all_radial_means.append(radial_means)

        # 多层平均
        stacked = np.array(all_radial_means)
        radial_curve = np.nanmean(stacked, axis=0)

        # 插值填补 NaN
        valid = ~np.isnan(radial_curve)
        if valid.sum() < 3:
            self.logger.warning("  径向采样点不足, 跳过径向校正")
            self._correction_map = np.ones((H, W), dtype=np.float32)
            return self._correction_map

        from scipy.interpolate import interp1d
        r_centers = (np.arange(n_bins) + 0.5) / n_bins
        interp_func = interp1d(r_centers[valid], radial_curve[valid],
                                kind='linear', fill_value='extrapolate')
        radial_curve_full = interp_func(r_centers)

        # 校正因子: 中心为基准
        center_value = radial_curve_full[0]
        if center_value == 0:
            center_value = radial_curve_full[radial_curve_full > 0].min()

        correction_1d = center_value / (radial_curve_full + 1e-8)

        # 裁剪到合理范围
        lo, hi = self.config.radial_correction_range
        correction_1d = np.clip(correction_1d, lo, hi)

        # 映射到2D
        correction_map = np.ones((H, W), dtype=np.float32)
        for i in range(n_bins):
            r_low = i / n_bins
            r_high = (i + 1) / n_bins
            ring = (dist_norm >= r_low) & (dist_norm < r_high)
            correction_map[ring] = correction_1d[i]

        self._correction_map = correction_map

        self.logger.info(f"  径向曲线: center={center_value:.1f}")
        self.logger.info(f"  校正范围: [{correction_map.min():.3f}, "
                         f"{correction_map.max():.3f}]")

        return correction_map

    def apply(self, volume: np.ndarray) -> np.ndarray:
        """将径向校正应用到体数据"""
        if not self.config.radial_correction:
            return volume

        if self._correction_map is None:
            raise RuntimeError("校正图尚未计算, 请先调用 compute_correction_map()")

        corrected = np.zeros_like(volume)
        for z in range(volume.shape[0]):
            corrected[z] = volume[z] * self._correction_map

        return corrected


class VolumeOutputBuilder:
    """
    清洁体数据输出构建器

    根据不同模式, 从预处理后的数据构建最终输出:
      - diff:          target - liquid, 归一化到 [0, 1]
      - diff_positive: 只保留枝晶信号 (diff<0 部分的绝对值)
      - raw_clean:     原始灰度归一化到 [0, 1]
    """

    def __init__(self, config: CleaningConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def build_diff(self, target: np.ndarray, liquid_ref: np.ndarray,
                   sample_mask: np.ndarray) -> np.ndarray:
        """
        差分模式: target - liquid_ref

        差分值含义:
          < 0: 灰度降低 = 固相(枝晶)形成
          ≈ 0: 无变化 = 液相区域
          > 0: 灰度升高 = 溶质富集的液相

        归一化策略: 以0为中心, 对称裁剪到[-clip, +clip], 映射到[0, 1]
        这样0.5对应无变化, <0.5对应枝晶, >0.5对应富溶质
        """
        self.logger.info("  [diff] 构建差分体数据...")

        diff = target - liquid_ref

        # 平滑去高频噪声
        if self.config.diff_smooth_sigma > 0:
            diff = gaussian_filter(diff, sigma=self.config.diff_smooth_sigma)

        # 样品外置零
        diff[~sample_mask] = 0

        # 统计
        sample_vals = diff[sample_mask]
        self.logger.info(f"    差分统计 (样品内): mean={sample_vals.mean():.1f}, "
                         f"std={sample_vals.std():.1f}")
        self.logger.info(f"    P1={np.percentile(sample_vals, 1):.0f}, "
                         f"P50={np.percentile(sample_vals, 50):.0f}, "
                         f"P99={np.percentile(sample_vals, 99):.0f}")

        # 对称裁剪
        clip_val = np.percentile(np.abs(sample_vals), self.config.diff_clip_percentile)
        diff = np.clip(diff, -clip_val, clip_val)

        # 归一化到 [0, 1]
        output = (diff + clip_val) / (2 * clip_val + 1e-8)
        output = np.clip(output, 0, 1).astype(self.config.output_dtype)
        output[~sample_mask] = 0

        self.logger.info(f"    裁剪值: ±{clip_val:.0f}")
        self.logger.info(f"    输出范围: [{output[sample_mask].min():.4f}, "
                         f"{output[sample_mask].max():.4f}]")

        return output

    def build_diff_positive(self, target: np.ndarray, liquid_ref: np.ndarray,
                             sample_mask: np.ndarray) -> np.ndarray:
        """
        差分正值模式: 只保留枝晶信号

        枝晶 = diff < 0 的部分, 取绝对值
        液相 = 0
        """
        self.logger.info("  [diff_positive] 构建纯枝晶信号...")

        diff = target - liquid_ref

        if self.config.diff_smooth_sigma > 0:
            diff = gaussian_filter(diff, sigma=self.config.diff_smooth_sigma)

        # 只保留负值 (枝晶), 取绝对值
        dendrite_signal = np.clip(-diff, 0, None)
        dendrite_signal[~sample_mask] = 0

        # 统计
        positive_vals = dendrite_signal[sample_mask & (dendrite_signal > 0)]
        if len(positive_vals) > 0:
            clip_val = np.percentile(positive_vals, self.config.diff_clip_percentile)
            dendrite_signal = np.clip(dendrite_signal, 0, clip_val)
            output = dendrite_signal / (clip_val + 1e-8)
        else:
            output = dendrite_signal

        output = np.clip(output, 0, 1).astype(self.config.output_dtype)
        output[~sample_mask] = 0

        dendrite_fraction = (output[sample_mask] > 0.05).sum() / sample_mask.sum() * 100
        self.logger.info(f"    枝晶体素 (>0.05): {dendrite_fraction:.1f}% of sample")

        return output

    def build_raw_clean(self, target: np.ndarray,
                         sample_mask: np.ndarray) -> np.ndarray:
        """
        原始清理模式: 保留原始灰度, 外部置零, 归一化到[0,1]
        """
        self.logger.info("  [raw_clean] 构建清理后原始数据...")

        clean = target.copy()

        if self.config.raw_smooth_sigma > 0:
            clean = gaussian_filter(clean, sigma=self.config.raw_smooth_sigma)

        clean[~sample_mask] = 0

        # 基于样品内部统计归一化
        sample_vals = clean[sample_mask]
        p_lo, p_hi = self.config.raw_normalize_percentile
        vmin = np.percentile(sample_vals, p_lo)
        vmax = np.percentile(sample_vals, p_hi)

        output = (clean - vmin) / (vmax - vmin + 1e-8)
        output = np.clip(output, 0, 1).astype(self.config.output_dtype)
        output[~sample_mask] = 0

        self.logger.info(f"    归一化范围: [{vmin:.0f}, {vmax:.0f}]")

        return output


class VisualizationGenerator:
    """验证图生成器"""

    @staticmethod
    def generate_frame_report(target_raw, liquid_ref, clean_volumes: dict,
                               sample_mask, frame_name, output_dir, logger):
        """生成单帧的完整验证报告"""
        logger.info("  生成验证图...")

        D = target_raw.shape[0]
        slices_z = [D // 5, D * 2 // 5, D // 2, D * 3 // 5, D * 4 // 5]
        n_modes = len(clean_volumes)
        n_cols = 3 + n_modes  # original, diff_raw, mask, + 各模式

        fig, axes = plt.subplots(len(slices_z), n_cols,
                                  figsize=(5 * n_cols, 5 * len(slices_z)))

        vmin_raw = np.percentile(target_raw, 5)
        vmax_raw = np.percentile(target_raw, 95)
        diff_raw = target_raw - liquid_ref

        for row, z in enumerate(slices_z):
            # 原始数据
            axes[row, 0].imshow(target_raw[z], cmap='gray',
                                vmin=vmin_raw, vmax=vmax_raw)
            axes[row, 0].set_title(f'Original z={z}', fontsize=10)

            # 原始差分
            axes[row, 1].imshow(diff_raw[z], cmap='RdBu_r',
                                vmin=-3000, vmax=3000)
            axes[row, 1].set_title(f'Raw diff z={z}', fontsize=10)

            # 样品掩膜
            axes[row, 2].imshow(sample_mask[z], cmap='gray')
            axes[row, 2].set_title(f'Mask z={z}', fontsize=10)

            # 各输出模式
            for col_offset, (mode_name, vol) in enumerate(clean_volumes.items()):
                c = 3 + col_offset
                axes[row, c].imshow(vol[z], cmap='gray', vmin=0, vmax=1)
                axes[row, c].set_title(f'{mode_name} z={z}', fontsize=10)

        for ax in axes.flat:
            ax.axis('off')

        plt.suptitle(f'Frame: {frame_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'verification.png'), dpi=150)
        plt.close()


# ============================================================
# 主 Pipeline
# ============================================================

class VolumeCleaner:
    """
    体数据清理主Pipeline

    流程:
      1. 构建液相参考帧 (多帧平均)
      2. 提取样品掩膜 (灰度阈值 + 形态学)
      3. 环形伪影校正 (极坐标中值滤波) [可选, 仅raw模式]
      4. 径向不均匀性校正 (beam hardening) [可选]
      5. 构建输出体数据 (diff / diff_positive / raw_clean)
      6. 保存 + 可视化
    """

    def __init__(self, config: CleaningConfig):
        self.config = config
        self.logger = setup_logger(config.output_base)
        self.liquid_builder = LiquidReferenceBuilder(config, self.logger)
        self.mask_extractor = SampleMaskExtractor(config, self.logger)
        self.output_builder = VolumeOutputBuilder(config, self.logger)

        self._liquid_ref = None
        self._sample_mask = None

    def initialize(self):
        """初始化: 构建液相参考 + 样品掩膜 (只需执行一次)"""
        self.logger.info("=" * 60)
        self.logger.info("体数据清理 Pipeline 初始化")
        self.logger.info("=" * 60)

        t0 = time.time()

        # 液相参考
        self._liquid_ref = self.liquid_builder.build()

        # 样品掩膜 (用液相参考帧提取, 因为此时样品边界最清晰)
        self._sample_mask = self.mask_extractor.extract(self._liquid_ref)

        # 径向校正器
        cy, cx = self.mask_extractor.center
        r = self.mask_extractor.radius
        self.radial_corrector = RadialNonuniformityCorrector(
            self.config, self.logger, cy, cx, r)

        # 预计算径向校正图
        if self.config.radial_correction:
            self.radial_corrector.compute_correction_map(
                self._liquid_ref, self._sample_mask)

        # 环形伪影校正器
        self.ring_corrector = RingArtifactCorrector(
            self.config, self.logger, cy, cx)

        self.logger.info(f"\n初始化完成, 耗时 {time.time() - t0:.1f}s")

    def process_frame(self, frame_filename: str) -> dict:
        """
        处理单帧

        Parameters
        ----------
        frame_filename : str
            帧文件名 (如 '043_AG10_C1mm_0s_rotate_01800.tif')

        Returns
        -------
        results : dict
            键为模式名, 值为清理后的 np.ndarray (D, H, W), float32, [0, 1]
        """
        if self._liquid_ref is None:
            raise RuntimeError("Pipeline未初始化, 请先调用 initialize()")

        frame_path = os.path.join(self.config.timesteps_dir, frame_filename)
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"帧文件不存在: {frame_path}")

        frame_num = frame_filename.split('_')[-1].replace('.tif', '')

        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"[处理帧] {frame_filename} (#{frame_num})")
        self.logger.info("=" * 60)

        t0 = time.time()

        # 读取目标帧
        target_raw = tifffile.imread(frame_path).astype(np.float32)
        self.logger.info(f"  读取完成: shape={target_raw.shape}, "
                         f"mean={target_raw.mean():.1f}")

        # 径向校正 (对原始数据)
        if self.config.radial_correction:
            target_corrected = self.radial_corrector.apply(target_raw)
            liquid_corrected = self.radial_corrector.apply(self._liquid_ref)
        else:
            target_corrected = target_raw
            liquid_corrected = self._liquid_ref

        # 确定输出模式
        if self.config.output_mode == MODE_ALL:
            modes = [MODE_DIFF, MODE_DIFF_POSITIVE, MODE_RAW_CLEAN]
        else:
            modes = [self.config.output_mode]

        # 构建各模式输出
        results = {}
        for mode in modes:
            if mode == MODE_DIFF:
                results[mode] = self.output_builder.build_diff(
                    target_corrected, liquid_corrected, self._sample_mask)
            elif mode == MODE_DIFF_POSITIVE:
                results[mode] = self.output_builder.build_diff_positive(
                    target_corrected, liquid_corrected, self._sample_mask)
            elif mode == MODE_RAW_CLEAN:
                # raw模式额外做环形伪影校正
                target_ring_corrected = target_corrected
                if self.config.ring_correction:
                    target_ring_corrected = self.ring_corrector.correct(
                        target_corrected, self._sample_mask)
                results[mode] = self.output_builder.build_raw_clean(
                    target_ring_corrected, self._sample_mask)

        # 保存
        frame_dir = os.path.join(self.config.output_base, f'frame_{frame_num}')
        os.makedirs(frame_dir, exist_ok=True)

        for mode_name, vol in results.items():
            # TIFF (float32)
            tiff_path = os.path.join(frame_dir, f'volume_{mode_name}.tif')
            tifffile.imwrite(tiff_path, vol)
            self.logger.info(f"  保存: {tiff_path}")

            # NPY (可选)
            if self.config.save_npy:
                npy_path = os.path.join(frame_dir, f'volume_{mode_name}.npy')
                np.save(npy_path, vol)

        # 保存样品掩膜 (首帧时)
        mask_path = os.path.join(frame_dir, 'sample_mask.tif')
        if not os.path.exists(mask_path):
            tifffile.imwrite(mask_path, self._sample_mask.astype(np.uint8) * 255)

        # 可视化
        if self.config.save_visualization:
            VisualizationGenerator.generate_frame_report(
                target_raw, self._liquid_ref, results,
                self._sample_mask, frame_filename, frame_dir, self.logger)

        elapsed = time.time() - t0
        self.logger.info(f"  帧处理完成, 耗时 {elapsed:.1f}s")

        return results

    def process_all_frames(self, frame_list=None):
        """
        批量处理多帧

        Parameters
        ----------
        frame_list : list of str, optional
            要处理的帧文件名列表. None = 全部帧
        """
        if frame_list is None:
            frame_list = sorted([
                f for f in os.listdir(self.config.timesteps_dir)
                if f.endswith('.tif') and not os.path.isdir(
                    os.path.join(self.config.timesteps_dir, f))
            ])

        total = len(frame_list)
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"批量处理: {total} 帧")
        self.logger.info(f"{'=' * 60}")

        t_total = time.time()
        for i, fname in enumerate(frame_list):
            self.logger.info(f"\n[{i + 1}/{total}] {fname}")
            try:
                self.process_frame(fname)
            except Exception as e:
                self.logger.error(f"  错误: {e}")
                continue

        elapsed_total = time.time() - t_total
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"批量处理完成: {total} 帧, "
                         f"总耗时 {elapsed_total:.0f}s "
                         f"({elapsed_total / total:.1f}s/帧)")
        self.logger.info(f"输出目录: {self.config.output_base}")


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Tomoscopy体数据伪影去除Pipeline')
    parser.add_argument('--all', action='store_true',
                        help='处理全部300帧')
    parser.add_argument('--frame', type=str, default=None,
                        help='处理指定帧编号 (如 01650)')
    parser.add_argument('--mode', type=str, default=MODE_DIFF,
                        choices=VALID_MODES,
                        help=f'输出模式: {VALID_MODES}')
    parser.add_argument('--output', type=str, default=None,
                        help='自定义输出目录')
    parser.add_argument('--no-ring', action='store_true',
                        help='禁用环形伪影校正')
    parser.add_argument('--no-radial', action='store_true',
                        help='禁用径向不均匀校正')
    parser.add_argument('--no-vis', action='store_true',
                        help='不生成验证图 (加速批量处理)')
    return parser.parse_args()


def main():
    args = parse_args()

    # 配置
    config = CleaningConfig()
    config.output_mode = args.mode

    if args.output:
        config.output_base = args.output
    if args.no_ring:
        config.ring_correction = False
    if args.no_radial:
        config.radial_correction = False
    if args.no_vis:
        config.save_visualization = False

    # 初始化
    cleaner = VolumeCleaner(config)
    cleaner.initialize()

    # 确定处理帧
    all_files = sorted([
        f for f in os.listdir(config.timesteps_dir)
        if f.endswith('.tif') and not os.path.isdir(
            os.path.join(config.timesteps_dir, f))
    ])

    if args.frame:
        frames = [f for f in all_files if args.frame in f]
        if not frames:
            print(f"未找到包含 '{args.frame}' 的帧")
            return
    elif args.all:
        # frames = all_files
        frames = [f for f in all_files
            if config.frame_start <= int(f.split('_')[-1].replace('.tif', '')) <= config.frame_end]
    else:
        # 默认: 5个关键帧
        indices = [0, 74, 149, 224, len(all_files) - 1]
        frames = [all_files[i] for i in indices if i < len(all_files)]
        print(f"默认处理 {len(frames)} 个测试帧, 使用 --all 处理全部")

    # 处理
    if len(frames) == 1:
        cleaner.process_frame(frames[0])
    else:
        cleaner.process_all_frames(frames)

    print(f"\n完成! 输出: {config.output_base}")


if __name__ == '__main__':
    main()