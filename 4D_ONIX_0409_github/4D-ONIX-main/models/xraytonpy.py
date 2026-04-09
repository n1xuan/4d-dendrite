"""
================================================================================
枝晶投影数据 → 4D-ONIX 输入格式转换器
================================================================================

将 numpy_dendrite_drr_projection.py 输出的逐帧NPY投影文件
打包为 4D-ONIX 模型要求的单一 NPY 文件.

4D-ONIX 输入格式:
    shape: [Time, View, 1, H, W]
    dtype: float32
    说明:
      - Time: 时间步数 (帧数)
      - View: 每帧的投影视角数 (2 = 正交双角度)
      - 1:    通道维度 (灰度图, 单通道)
      - H:    投影图高度 (像素)
      - W:    投影图宽度 (像素)

输入目录结构:
    drr_projections/
      frame_01680/
        proj_angle000.0.npy    (0°投影, float32)
        proj_angle090.0.npy    (90°投影, float32)
      frame_01681/
        ...
      frame_01800/
        ...

输出:
    dataset/dendrite_121frames_2views.npy
        shape: (121, 2, 1, H, W)

使用:
    # 默认参数
    python xraytonpy.py

    # 自定义尺寸 (缩放到指定分辨率)
    python xraytonpy.py --target-size 128 64

    # 自定义路径
    python xraytonpy.py --input-dir /path/to/projections --output /path/to/output.npy

    # 指定帧范围
    python xraytonpy.py --frame-start 1700 --frame-end 1750

依赖:
    numpy, opencv-python (cv2)
================================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import logging


# ============================================================
# 配置
# ============================================================
class OnixConvertConfig:
    """4D-ONIX 数据转换配置"""

    def __init__(self):
        # --- 输入路径 ---
        self.projection_dir = '/root/autodl-tmp/data/dendrite_drr_projections'

        # --- 输出路径 ---
        self.output_path = '/root/autodl-tmp/4D_ONIX_0402/4D-ONIX-main/models/dataset/dendrite_2views.npy'

        # --- 帧范围 ---
        self.frame_start = 1680
        self.frame_end = 1800

        # --- 视角配置 ---
        # NPY文件名中的角度标签, 按顺序排列
        # 顺序决定了输出中 View 维度的排列
        self.angle_labels = ['000.0', '090.0']

        # --- 图像尺寸 ---
        # None = 保持原始尺寸, (W, H) = 缩放到指定尺寸
        # 4D-ONIX 原始SLM数据用的是 (128, 64)
        # 你的数据原始投影尺寸是 (528, 280) 或 (280, 528)
        self.target_size = None  # 设为 (128, 64) 可匹配原始4D-ONIX

        # --- 缩放插值方法 ---
        # cv2.INTER_AREA: 适合缩小 (抗锯齿)
        # cv2.INTER_LINEAR: 双线性插值
        self.interpolation = cv2.INTER_AREA

        # --- 归一化 ---
        # 'none':      不做归一化, 保持原始float32值
        # 'per_frame': 每帧独立归一化到 [0, 1]
        # 'global':    全局归一化到 [0, 1] (推荐, 保持帧间相对强度)
        self.normalization = 'global'

        # --- 负噪点处理 ---
        self.clip_negative = True  # 将负值裁剪为0

        # --- 验证 ---
        self.save_verification = True  # 保存验证图


# ============================================================
# 日志
# ============================================================
def setup_logger():
    logger = logging.getLogger('dendritexray_to_npy')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    return logger


# ============================================================
# 核心转换器
# ============================================================
class DendriteToOnixConverter:
    """
    将逐帧投影NPY文件转换为4D-ONIX输入格式.

    处理流程:
      1. 扫描投影目录, 按帧编号排序
      2. 逐帧加载所有角度的NPY投影
      3. 可选: 缩放到目标尺寸
      4. 可选: 归一化
      5. 堆叠为 [Time, View, 1, H, W]
      6. 保存为单一NPY文件
    """

    def __init__(self, config: OnixConvertConfig):
        self.config = config
        self.logger = setup_logger()

    def _discover_frames(self):
        """
        扫描投影目录, 找到所有有效帧.

        有效帧 = 目录存在 + 所有角度的NPY文件都存在 + 帧编号在范围内.

        Returns
        -------
        frame_nums : list of str
            排序后的帧编号列表, 如 ['01680', '01681', ..., '01800']
        """
        proj_dir = self.config.projection_dir

        if not os.path.isdir(proj_dir):
            raise FileNotFoundError(f"投影目录不存在: {proj_dir}")

        # 扫描所有 frame_XXXXX 目录
        all_dirs = sorted([
            d for d in os.listdir(proj_dir)
            if d.startswith('frame_') and os.path.isdir(os.path.join(proj_dir, d))
        ])

        valid_frames = []
        skipped_range = 0
        skipped_missing = 0

        for d in all_dirs:
            num_str = d.replace('frame_', '')

            # 解析帧编号
            try:
                num = int(num_str)
            except ValueError:
                self.logger.warning(f"  跳过无效目录: {d}")
                continue

            # 帧范围过滤
            if num < self.config.frame_start or num > self.config.frame_end:
                skipped_range += 1
                continue

            # 检查所有角度的NPY文件是否存在
            frame_dir = os.path.join(proj_dir, d)
            all_exist = True
            for angle_label in self.config.angle_labels:
                npy_path = os.path.join(frame_dir, f'proj_angle{angle_label}.npy')
                if not os.path.exists(npy_path):
                    self.logger.warning(f"  帧 {num_str}: 缺少 proj_angle{angle_label}.npy")
                    all_exist = False
                    break

            if all_exist:
                valid_frames.append(num_str)
            else:
                skipped_missing += 1

        self.logger.info(f"  发现帧: {len(valid_frames)} 有效, "
                         f"{skipped_range} 超出范围, "
                         f"{skipped_missing} 缺少文件")

        if len(valid_frames) == 0:
            raise RuntimeError(
                f"未找到有效帧. 请确认:\n"
                f"  1. 投影目录: {proj_dir}\n"
                f"  2. 帧范围: {self.config.frame_start} - {self.config.frame_end}\n"
                f"  3. 角度文件: proj_angle{self.config.angle_labels[0]}.npy 等"
            )

        return valid_frames

    def _load_single_frame(self, frame_num):
        """
        加载单帧的所有角度投影.

        Parameters
        ----------
        frame_num : str

        Returns
        -------
        views : list of ndarray
            每个元素是一个角度的投影, shape (H, W), float32
        """
        frame_dir = os.path.join(self.config.projection_dir, f'frame_{frame_num}')
        views = []

        for angle_label in self.config.angle_labels:
            npy_path = os.path.join(frame_dir, f'proj_angle{angle_label}.npy')
            proj = np.load(npy_path).astype(np.float32)

            # 验证维度
            if proj.ndim != 2:
                raise ValueError(
                    f"帧 {frame_num} 角度 {angle_label}: "
                    f"期望2D数组, 实际维度 {proj.ndim}, shape {proj.shape}"
                )

            # 负值裁剪
            if self.config.clip_negative:
                proj = np.clip(proj, 0, None)

            # 缩放
            if self.config.target_size is not None:
                target_w, target_h = self.config.target_size
                proj = cv2.resize(
                    proj, (target_w, target_h),
                    interpolation=self.config.interpolation
                )

            views.append(proj)

        # 验证所有视角尺寸一致
        shapes = [v.shape for v in views]
        if len(set(shapes)) > 1:
            raise ValueError(
                f"帧 {frame_num}: 不同角度的投影尺寸不一致: {shapes}"
            )

        return views

    def convert(self):
        """
        执行完整转换流程.

        Returns
        -------
        output_path : str
            输出NPY文件的路径
        """
        self.logger.info("=" * 60)
        self.logger.info("枝晶投影 → npy格式转换")
        self.logger.info("=" * 60)

        t_start = time.time()

        # --- 1. 发现有效帧 ---
        self.logger.info("\nStep 1: 扫描投影目录...")
        frame_nums = self._discover_frames()
        n_frames = len(frame_nums)
        n_views = len(self.config.angle_labels)
        self.logger.info(f"  有效帧: {n_frames}")
        self.logger.info(f"  视角数: {n_views} ({self.config.angle_labels})")
        self.logger.info(f"  帧范围: {frame_nums[0]} - {frame_nums[-1]}")

        # --- 2. 加载第一帧确定尺寸 ---
        self.logger.info("\nStep 2: 确定数据尺寸...")
        first_views = self._load_single_frame(frame_nums[0])
        proj_h, proj_w = first_views[0].shape

        if self.config.target_size is not None:
            target_w, target_h = self.config.target_size
            self.logger.info(f"  原始尺寸: {proj_h} × {proj_w}")
            self.logger.info(f"  目标尺寸: {target_h} × {target_w}")
            final_h, final_w = target_h, target_w
        else:
            self.logger.info(f"  投影尺寸: {proj_h} × {proj_w} (保持原始)")
            final_h, final_w = proj_h, proj_w

        # --- 3. 预分配输出数组 ---
        # shape: [Time, View, 1, H, W]
        output_shape = (n_frames, n_views, 1, final_h, final_w)
        self.logger.info(f"\nStep 3: 预分配数组 {output_shape}...")
        memory_mb = np.prod(output_shape) * 4 / (1024 * 1024)  # float32 = 4 bytes
        self.logger.info(f"  预计内存: {memory_mb:.1f} MB")

        data = np.zeros(output_shape, dtype=np.float32)

        # --- 4. 逐帧加载 ---
        self.logger.info(f"\nStep 4: 加载 {n_frames} 帧...")

        global_min = float('inf')
        global_max = float('-inf')

        for t, frame_num in enumerate(frame_nums):
            if t % 20 == 0 or t == n_frames - 1:
                self.logger.info(f"  [{t + 1}/{n_frames}] frame_{frame_num}")

            views = self._load_single_frame(frame_num)

            for v, proj in enumerate(views):
                # 尺寸验证
                if proj.shape != (final_h, final_w):
                    raise ValueError(
                        f"帧 {frame_num} 视角 {v}: "
                        f"尺寸 {proj.shape} != 期望 ({final_h}, {final_w})"
                    )
                # 单张2D数据填入5D大张量中特定坐标位置
                data[t, v, 0, :, :] = proj

                # 跟踪全局统计
                global_min = min(global_min, proj.min())
                global_max = max(global_max, proj.max())

        self.logger.info(f"  加载完成")
        self.logger.info(f"  全局值域: [{global_min:.6f}, {global_max:.6f}]")

        # --- 5. 归一化 ---
        self.logger.info(f"\nStep 5: 归一化 (模式: {self.config.normalization})...")

        if self.config.normalization == 'global':
            # 全局归一化: 保持帧间相对强度
            if global_max - global_min > 1e-10:
                data = (data - global_min) / (global_max - global_min)
            self.logger.info(f"  全局归一化: [{global_min:.6f}, {global_max:.6f}] → [0, 1]")

        elif self.config.normalization == 'per_frame':
            # 逐帧归一化: 每帧独立
            for t in range(n_frames):
                frame_data = data[t]
                fmin, fmax = frame_data.min(), frame_data.max()
                if fmax - fmin > 1e-10:
                    data[t] = (frame_data - fmin) / (fmax - fmin)
            self.logger.info(f"  逐帧独立归一化 → [0, 1]")

        elif self.config.normalization == 'none':
            self.logger.info(f"  不做归一化, 保持原始值")

        else:
            raise ValueError(f"未知归一化模式: {self.config.normalization}")

        # --- 6. 最终验证 ---
        self.logger.info(f"\nStep 6: 数据验证...")

        assert data.shape == output_shape, \
            f"形状不匹配: {data.shape} != {output_shape}"
        assert data.dtype == np.float32, \
            f"类型不匹配: {data.dtype} != float32"
        assert np.isfinite(data).all(), \
            "数据包含 NaN 或 Inf"

        self.logger.info(f"  形状: {data.shape}")
        self.logger.info(f"  类型: {data.dtype}")
        self.logger.info(f"  值域: [{data.min():.6f}, {data.max():.6f}]")
        self.logger.info(f"  均值: {data.mean():.6f}")
        self.logger.info(f"  非零: {(data > 0).sum() / data.size * 100:.1f}%")
        self.logger.info(f"  验证通过 ✓")

        # --- 7. 保存 ---
        self.logger.info(f"\nStep 7: 保存...")
        output_dir = os.path.dirname(self.config.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        np.save(self.config.output_path, data)
        file_size_mb = os.path.getsize(self.config.output_path) / (1024 * 1024)
        self.logger.info(f"  路径: {self.config.output_path}")
        self.logger.info(f"  大小: {file_size_mb:.1f} MB")

        # --- 8. 验证图 ---
        if self.config.save_verification:
            self._save_verification(data, frame_nums, output_dir)

        # --- 完成 ---
        elapsed = time.time() - t_start
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"转换完成! 耗时 {elapsed:.1f}s")
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"\n输出: {self.config.output_path}")
        self.logger.info(f"形状: {data.shape}")
        self.logger.info(f"  dim 0: Time  = {data.shape[0]} 帧")
        self.logger.info(f"  dim 1: View  = {data.shape[1]} 视角")
        self.logger.info(f"  dim 2: Chan  = {data.shape[2]} 通道")
        self.logger.info(f"  dim 3: H     = {data.shape[3]} 像素")
        self.logger.info(f"  dim 4: W     = {data.shape[4]} 像素")
        self.logger.info(f"\n4D-ONIX 预期格式: (Time, View, 1, H, W) ✓")

        return self.config.output_path

    def _save_verification(self, data, frame_nums, output_dir):
        """保存验证图: 首帧/中间帧/末帧的所有视角"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("  matplotlib未安装, 跳过验证图")
            return

        n_frames = data.shape[0]
        n_views = data.shape[1]

        # 选择首/中/末三帧
        sample_indices = [0, n_frames // 2, n_frames - 1]
        sample_indices = [i for i in sample_indices if i < n_frames]

        fig, axes = plt.subplots(
            len(sample_indices), n_views,
            figsize=(7 * n_views, 6 * len(sample_indices))
        )

        if len(sample_indices) == 1:
            axes = axes[np.newaxis, :]
        if n_views == 1:
            axes = axes[:, np.newaxis]

        for row, t_idx in enumerate(sample_indices):
            frame_num = frame_nums[t_idx]
            for col in range(n_views):
                img = data[t_idx, col, 0]  # (H, W)
                axes[row, col].imshow(img, cmap='gray', aspect='auto')
                angle = self.config.angle_labels[col]
                axes[row, col].set_title(
                    f'Frame {frame_num} | {angle}° | '
                    f'[{img.min():.3f}, {img.max():.3f}]',
                    fontsize=11
                )
                axes[row, col].axis('off')

        plt.suptitle(
            f'4D-ONIX Input Verification\n'
            f'Shape: {data.shape} | dtype: {data.dtype}',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()

        verify_dir = output_dir if output_dir else '.'
        verify_path = os.path.join(verify_dir, 'onix_input_verification.png')
        plt.savefig(verify_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  验证图: {verify_path}")


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='枝晶投影 → 4D-ONIX 输入格式转换',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python dendrite_to_onix.py
  python dendrite_to_onix.py --target-size 128 64
  python dendrite_to_onix.py --frame-start 1700 --frame-end 1750
  python dendrite_to_onix.py --norm per_frame
        """
    )
    parser.add_argument('--input-dir', type=str, default=None,
                        help='投影NPY输入目录 (drr_projections/)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出NPY文件路径')
    parser.add_argument('--frame-start', type=int, default=None,
                        help='起始帧编号')
    parser.add_argument('--frame-end', type=int, default=None,
                        help='结束帧编号')
    parser.add_argument('--target-size', nargs=2, type=int, default=None,
                        metavar=('W', 'H'),
                        help='目标尺寸 宽 高 (如 128 64)')
    parser.add_argument('--norm', type=str, default=None,
                        choices=['none', 'per_frame', 'global'],
                        help='归一化模式')
    parser.add_argument('--angles', nargs='+', type=str, default=None,
                        help='角度标签列表 (如 000.0 090.0)')
    parser.add_argument('--no-verify', action='store_true',
                        help='不生成验证图')

    args = parser.parse_args()

    # 配置
    config = OnixConvertConfig()

    if args.input_dir:
        config.projection_dir = args.input_dir
    if args.output:
        config.output_path = args.output
    if args.frame_start:
        config.frame_start = args.frame_start
    if args.frame_end:
        config.frame_end = args.frame_end
    if args.target_size:
        config.target_size = tuple(args.target_size)  # (W, H)
    if args.norm:
        config.normalization = args.norm
    if args.angles:
        config.angle_labels = args.angles
    if args.no_verify:
        config.save_verification = False

    # 执行转换
    converter = DendriteToOnixConverter(config)
    converter.convert()


if __name__ == '__main__':
    main()