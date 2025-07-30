# -*- coding: utf-8 -*-
# 文件路径: diffbir/dataset/dataimport.py

from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# 关闭Pillow对于超大图像文件的解压炸弹保护
Image.MAX_IMAGE_PIXELS = None


class PairedImageDataset(Dataset):
    """
    【已升级】一个用于加载成对 LQ/HQ 图像的数据集类。

    现已支持一对多的文件匹配模式，并通过 `use_dynamic_lq` 控制。
    默认输出尺寸为 512x512。
    此版本包含了详细的调试打印信息。
    """

    def __init__(
            self,
            data_sources: List[Dict[str, str]],
            split: str,
            use_dynamic_lq: bool = False,
            out_size: int = 512
    ):
        """
        构造函数。

        Args:
            data_sources (List[Dict[str, str]]): 数据源配置列表。
            split (str): 数据划分 ('train', 'validate', 'test')。
            use_dynamic_lq (bool): 是否加载一个HQ对应的所有动态LQ。默认为 False。
            out_size (int): 输出图像的统一尺寸。默认为 512。
        """
        super().__init__()
        self.data_sources = data_sources
        self.split = split
        self.use_dynamic_lq = use_dynamic_lq
        self.out_size = out_size
        self.image_pairs = []
        self._scan_files()

    def _scan_files(self):
        """
        【已重构+调试】扫描所有数据源路径，查找并匹配 LQ 和 HQ 图像对。
        采用“从HQ找LQ”的策略，以支持“一对多”的匹配。
        """
        # 打印扫描任务的初始信息
        print(f"--- [PairedImageDataset] 开始扫描 '{self.split}' 划分的图像对 ---")
        print(f"--- 动态LQ数据模式: {'启用' if self.use_dynamic_lq else '关闭 (仅使用_001)'} ---")

        supported_extensions = ['*.png', '*.tiff', '*.tif', '*.jpg', '*.jpeg']

        # 遍历在 YAML 中配置的每一个数据源
        for source in self.data_sources:
            lq_base_path = Path(source['lq_path'])
            hq_base_path = Path(source['hq_path'])
            lq_split_path = lq_base_path / self.split
            hq_split_path = hq_base_path / self.split

            # 如果路径不存在，则跳过此数据源
            if not lq_split_path.is_dir() or not hq_split_path.is_dir():
                print(f"\n[警告] 路径不存在，跳过数据源: LQ='{lq_split_path}', HQ='{hq_split_path}'")
                continue

            # --- 调试打印: 正在扫描哪个HQ目录 ---
            print(f"\n[信息] 正在扫描 HQ 目录: {hq_split_path}")

            # 1. 首先遍历 HQ 目录下的所有图像文件
            hq_files = []
            for ext in supported_extensions:
                hq_files.extend(hq_split_path.glob(ext))

            # --- 调试打印: 报告在HQ目录中找到的文件总数 ---
            if not hq_files:
                print("  > 在此HQ目录中未找到任何图像文件。")
                continue  # 如果没找到HQ文件，直接处理下一个数据源
            else:
                print(f"  > 在此HQ目录中找到 {len(hq_files)} 个图像文件。")

            # 遍历找到的每一个HQ文件，尝试为它寻找匹配的LQ文件
            for hq_file_path in hq_files:
                hq_stem = hq_file_path.stem  # 获取 HQ 文件名 (不含后缀), 例如 "imageA"

                # --- 调试打印: 当前正在处理哪个HQ文件 ---
                # 使用一个空行来分隔每个HQ文件的处理日志，使输出更清晰

                # 2. 根据 `use_dynamic_lq` 的设置，决定查找策略
                if self.use_dynamic_lq:
                    # 【策略一: 启用动态】查找所有 "imageA_*" 格式的 LQ 文件
                    search_pattern = f'{hq_stem}_*.*'
                else:
                    # 【策略二: 关闭动态】只查找 "imageA_001.*" 格式的 LQ 文件
                    search_pattern = f'{hq_stem}_001.*'

                # --- 调试打印: 打印用于查找LQ文件的搜索模式 ---
                # 这一点对于检查 `use_dynamic_lq` 开关是否按预期工作至关重要
                matched_lq_paths = list(lq_split_path.glob(search_pattern))

                # --- 调试打印: 报告查找结果 ---
                if not matched_lq_paths:
                    print("    - 结果: 未找到任何匹配的LQ文件。")
                    continue  # 继续处理下一个HQ文件


                # 3. 将所有找到的有效配对加入列表
                for lq_path in matched_lq_paths:
                    # 确保找到的文件后缀是支持的
                    if any(lq_path.name.lower().endswith(ext.strip('*')) for ext in supported_extensions):
                        # --- 调试打印: 打印每一个成功建立的配对 ---
                        # 这是最关键的调试信息，显示了最终采纳的匹配结果
                        self.image_pairs.append({
                            'lq': str(lq_path),
                            'hq': str(hq_file_path)
                        })

        # --- 调试打印: 最终总结信息 ---
        print(f"\n--- [PairedImageDataset] 扫描完成！总共建立了 {len(self.image_pairs)} 个有效的图像对。---")

        # --- 调试打印: 抽样检查已建立的配对 ---
        # 如果列表太长，只打印前5个作为样本，方便快速核对
        if self.image_pairs:
            print("--- 抽样检查前5个已建立的配对 (完整路径): ---")
            for i, pair in enumerate(self.image_pairs[:5]):
                print(f"  {i + 1}: HQ='{pair['hq']}', LQ='{pair['lq']}'")
        print("-" * 50 + "\n")

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, index: int) -> tuple:
        # __getitem__ 的内部逻辑保持不变，因为它只负责处理一对给定的路径
        pair = self.image_pairs[index]
        lq_path = pair['lq']
        hq_path = pair['hq']

        try:
            lq_pil = Image.open(lq_path)
            hq_pil = Image.open(hq_path)
            lq_np = np.array(lq_pil)
            hq_np = np.array(hq_pil)

            if lq_np.ndim == 2:
                lq_np = np.stack([lq_np] * 3, axis=-1)
            elif lq_np.shape[2] == 4:
                lq_np = lq_np[:, :, :3]

            if hq_np.ndim == 2:
                hq_np = np.stack([hq_np] * 3, axis=-1)
            elif hq_np.shape[2] == 4:
                hq_np = hq_np[:, :, :3]

            if self.out_size is not None:
                lq_np = cv2.resize(lq_np, (self.out_size, self.out_size), interpolation=cv2.INTER_LANCZOS4)
                hq_np = cv2.resize(hq_np, (self.out_size, self.out_size), interpolation=cv2.INTER_LANCZOS4)

            if lq_np.dtype == np.uint8:
                lq_max_val = 255.0
            elif lq_np.dtype == np.uint16:
                lq_max_val = 65535.0
            else:
                raise TypeError(f"不支持的LQ图像数据类型: {lq_np.dtype}")

            if hq_np.dtype == np.uint8:
                hq_max_val = 255.0
            elif hq_np.dtype == np.uint16:
                hq_max_val = 65535.0
            else:
                raise TypeError(f"不支持的HQ图像数据类型: {hq_np.dtype}")

            lq_processed = lq_np.astype(np.float32) / lq_max_val
            hq_processed = (hq_np.astype(np.float32) / hq_max_val) * 2.0 - 1.0
            prompt = ""
            return hq_processed, lq_processed, prompt

        except Exception as e:
            print(f"错误: 无法加载或处理图像对 L:{lq_path}, H:{hq_path}。错误信息: {e}")
            return self.__getitem__((index + 1) % len(self))