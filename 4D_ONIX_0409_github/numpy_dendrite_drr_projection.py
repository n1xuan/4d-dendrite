"""
================================================================================
枝晶平行束投影生成
================================================================================

同步辐射Tomoscopy是平行束成像, 投影 = 沿射线方向的体素值线积分.
对于正交角度(0°, 90°), 直接沿轴numpy求和即可精确计算, 无需plastimatch.

与旧版本(基于plastimatch锥束DRR)的区别:
  1. 精确平行束, 无锥束近似误差
  2. 无外部依赖 (不需要plastimatch)
  3. 速度快 (~0.1s/帧 vs ~10s/帧)
  4. 直接读取cleaned TIFF, 无需MHA中间格式

输入: volume_cleaning.py 输出的 float32 TIFF 体数据
输出: 每帧多角度投影图 (PNG + float32 NPY)

使用:
    python numpy_dendrite_drr_projection.py --frame 01800
    python numpy_dendrite_drr_projection.py --all
    python dendrite_drr_projection.py --frame 01800 --angles 0 90 45 135

依赖:
    numpy, scipy, tifffile, matplotlib, opencv-python
================================================================================
"""

import os
import time
import argparse
import logging
import numpy as np
import tifffile
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as ndimage_rotate


# ============================================================
# 配置
# ============================================================
class ProjectionConfig:
    """投影配置"""

    def __init__(self):
        # --- 路径 ---
        self.clean_volumes_dir = '/home/dh524/data/dendrite_tiff/clean_volumes'
        self.projection_output_dir = '/home/dh524/data/dendrite_tiff/drr_projections'
        self.tiff_filename = 'volume_diff_positive.tif'

        # --- 体素参数 ---
        self.voxel_spacing_mm = 0.00275  # 2.75 μm

        # --- 投影角度 ---
        # 0° = 沿X轴看 (检测器显示Y-Z), 90° = 沿Y轴看 (检测器显示X-Z)
        self.angles = [0.0, 90.0]

        # --- 帧范围 ---
        self.frame_start = 1680
        self.frame_end = 1800

        # --- 输出 ---
        self.save_png = True
        self.save_npy = True
        self.save_preview = True


# ============================================================
# 核心投影函数
# ============================================================

def parallel_beam_projection(volume, angle_deg, spacing_mm):
    """
    平行束投影 (ray-sum 线积分).

    对于同步辐射Tomoscopy, 平行束投影 = 沿射线方向的体素值求和 × 步长.
    正交角度(0°, 90°, 180°, 270°)直接沿轴求和, 无插值误差.
    一般角度先绕Z轴旋转体数据, 再沿X轴求和.

    Parameters
    ----------
    volume : ndarray, shape (Z, Y, X)
        三维体数据, 背景=0, 信号>0
    angle_deg : float
        投影角度(度), XY平面内旋转:
          0°   = 沿X轴正方向看, 检测器显示 Y(水平) × Z(垂直)
          90°  = 沿Y轴正方向看, 检测器显示 X(水平) × Z(垂直)
    spacing_mm : float
        各向同性体素尺寸 (mm)

    Returns
    -------
    projection : ndarray, shape (Z, H), float32
        二维投影图, 值 = 线积分 (value·mm)
    """
    angle = angle_deg % 360

    # 正交角度: 直接沿轴求和 (快速精确)
    if angle == 0.0:
        proj = volume.sum(axis=2)           # 沿X求和 → (Z, Y)
    elif angle == 90.0:
        proj = volume.sum(axis=1)           # 沿Y求和 → (Z, X)
    elif angle == 180.0:
        proj = volume.sum(axis=2)[:, ::-1]  # 沿-X求和
    elif angle == 270.0:
        proj = volume.sum(axis=1)[:, ::-1]  # 沿-Y求和
    else:
        # 一般角度: 绕Z轴旋转体数据后沿X求和
        # 旋转-θ使投影方向对齐X轴
        rotated = ndimage_rotate(
            volume, -angle_deg, axes=(2, 1),
            reshape=True, order=1, mode='constant', cval=0.0
        )
        proj = rotated.sum(axis=2)

    return proj.astype(np.float32) * spacing_mm


def save_projection_png(projection, png_path):
    """
    将float32投影图归一化保存为uint8 PNG.

    使用 [min, max] → [0, 255] 归一化, 正确处理任意值域.
    """
    pmin, pmax = projection.min(), projection.max()

    if pmax - pmin < 1e-10:
        img = np.zeros_like(projection, dtype=np.uint8)
    else:
        img = ((projection - pmin) / (pmax - pmin) * 255).astype(np.uint8)

    cv2.imwrite(png_path, img)
    return img


# ============================================================
# 投影器
# ============================================================

class DendriteDRRProjector:
    """枝晶平行束投影生成器"""

    def __init__(self, config: ProjectionConfig):
        self.config = config
        self.logger = self._setup_logger()
        os.makedirs(config.projection_output_dir, exist_ok=True)

    def _setup_logger(self):
        logger = logging.getLogger('dendrite_projection')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
        return logger

    def _load_volume(self, frame_num):
        """加载清洗后的TIFF体数据"""
        frame_dir = os.path.join(self.config.clean_volumes_dir, f'frame_{frame_num}')
        tiff_path = os.path.join(frame_dir, self.config.tiff_filename)

        if not os.path.exists(tiff_path):
            raise FileNotFoundError(
                f"TIFF文件不存在: {tiff_path}\n"
                f"请先运行: python 326_volume_cleaning.py --frame {frame_num} --mode diff_positive"
            )

        volume = tifffile.imread(tiff_path).astype(np.float32)
        return volume

    def project_single_frame(self, frame_num: str):
        """
        对单帧生成所有角度的平行束投影.

        Parameters
        ----------
        frame_num : str
            帧编号, 如 '01800'

        Returns
        -------
        projections : dict
            {angle_deg: projection_array}
        """
        # 加载体数据
        self.logger.info(f"  加载体数据...")
        volume = self._load_volume(frame_num)
        D, H, W = volume.shape
        self.logger.info(
            f"  形状: ({D}, {H}, {W}), "
            f"值域: [{volume.min():.4f}, {volume.max():.4f}], "
            f"非零: {(volume > 0).sum() / volume.size * 100:.1f}%"
        )

        # 输出目录
        output_dir = os.path.join(self.config.projection_output_dir, f'frame_{frame_num}')
        os.makedirs(output_dir, exist_ok=True)

        # 逐角度投影
        projections = {}
        spacing = self.config.voxel_spacing_mm

        for angle in self.config.angles:
            angle_label = f"{angle:05.1f}"
            self.logger.info(f"  投影 {angle:.1f}°...")

            t0 = time.time()
            proj = parallel_beam_projection(volume, angle, spacing)
            dt = time.time() - t0

            self.logger.info(
                f"    形状: {proj.shape}, "
                f"值域: [{proj.min():.6f}, {proj.max():.6f}], "
                f"耗时: {dt:.3f}s"
            )

            # 保存NPY (float32, 用于模型训练)
            if self.config.save_npy:
                npy_path = os.path.join(output_dir, f'proj_angle{angle_label}.npy')
                np.save(npy_path, proj)

            # 保存PNG (uint8, 用于可视化)
            if self.config.save_png:
                png_path = os.path.join(output_dir, f'proj_angle{angle_label}.png')
                save_projection_png(proj, png_path)

            projections[angle] = proj

        # 预览图
        if self.config.save_preview and projections:
            self._save_preview(projections, frame_num, output_dir)

        return projections

    def _save_preview(self, projections, frame_num, output_dir):
        """保存投影预览图 (所有角度并排对比)"""
        n = len(projections)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
        if n == 1:
            axes = [axes]

        for ax, (angle, data) in zip(axes, sorted(projections.items())):
            ax.imshow(data, cmap='gray', aspect='auto')
            ax.set_title(f'Frame {frame_num} — {angle:.1f}°', fontsize=14)
            ax.set_xlabel(f'shape: {data.shape[1]}×{data.shape[0]}')
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        plt.tight_layout()
        preview_path = os.path.join(output_dir, 'preview.png')
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  预览: {preview_path}")

    def project_batch(self, frame_list):
        """批量处理多帧"""
        total = len(frame_list)
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"批量平行束投影: {total} 帧")
        self.logger.info(f"角度: {self.config.angles}")
        self.logger.info(f"体素: {self.config.voxel_spacing_mm} mm")
        self.logger.info(f"{'=' * 60}")

        t_total = time.time()
        success = 0
        failed = []

        for i, frame_num in enumerate(frame_list):
            self.logger.info(f"\n[{i + 1}/{total}] frame_{frame_num}")
            t0 = time.time()

            try:
                self.project_single_frame(frame_num)
                success += 1
            except Exception as e:
                self.logger.error(f"  错误: {e}")
                failed.append(frame_num)

            elapsed = time.time() - t0
            self.logger.info(f"  帧耗时: {elapsed:.1f}s")

        elapsed_total = time.time() - t_total
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"批量投影完成")
        self.logger.info(f"  成功: {success}/{total}")
        if failed:
            self.logger.info(f"  失败: {failed}")
        self.logger.info(
            f"  总耗时: {elapsed_total:.0f}s "
            f"({elapsed_total / max(total, 1):.1f}s/帧)"
        )
        self.logger.info(f"  输出: {self.config.projection_output_dir}")


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='枝晶平行束投影生成')
    parser.add_argument('--frame', type=str, default=None,
                        help='帧编号 (如 01800)')
    parser.add_argument('--all', action='store_true',
                        help='处理全部帧')
    parser.add_argument('--angles', nargs='+', type=float, default=None,
                        help='投影角度列表 (如 --angles 0 90 45 135)')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='TIFF输入目录 (clean_volumes)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='投影输出目录')
    parser.add_argument('--tiff-name', type=str, default=None,
                        help='输入TIFF文件名 (默认 volume_diff_positive.tif)')
    parser.add_argument('--spacing', type=float, default=None,
                        help='体素尺寸 (μm, 如 2.75)')

    args = parser.parse_args()

    # 配置
    config = ProjectionConfig()

    if args.input_dir:
        config.clean_volumes_dir = args.input_dir
    if args.output_dir:
        config.projection_output_dir = args.output_dir
    if args.tiff_name:
        config.tiff_filename = args.tiff_name
    if args.spacing:
        config.voxel_spacing_mm = args.spacing / 1000.0
    if args.angles:
        config.angles = args.angles

    # 创建投影器
    projector = DendriteDRRProjector(config)

    # 确定帧列表
    if args.frame:
        print(f"处理单帧: {args.frame}")
        projector.project_single_frame(args.frame)

    elif args.all:
        frame_dirs = sorted([
            d for d in os.listdir(config.clean_volumes_dir)
            if d.startswith('frame_') and os.path.isdir(
                os.path.join(config.clean_volumes_dir, d))
        ])

        frame_nums = []
        for d in frame_dirs:
            num_str = d.replace('frame_', '')
            try:
                num = int(num_str)
                if config.frame_start <= num <= config.frame_end:
                    frame_nums.append(num_str)
            except ValueError:
                continue

        if not frame_nums:
            print(f"在 {config.clean_volumes_dir} 中未找到帧目录")
            return

        projector.project_batch(frame_nums)

    else:
        print("默认处理 frame_01800")
        projector.project_single_frame('01800')

    print(f"\n完成! 输出: {config.projection_output_dir}")


if __name__ == '__main__':
    main()
