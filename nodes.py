"""ComfyUI 节点定义。"""

from __future__ import annotations

import numpy as np
import torch

from .core import (auto_layers, layers_to_text, parse_layers, quantize,
                   rgb_to_hex, upscale_nn)

CATEGORY = "像素/降采样"


# ────────────────────────────────────────────────────────────
#  张量 <-> numpy
# ────────────────────────────────────────────────────────────

def _to_rgba(image: torch.Tensor) -> np.ndarray:
    """ComfyUI 的 IMAGE [B,H,W,C] float 0~1 → uint8 RGBA（取第 0 张）。"""
    a = image[0].detach().cpu().numpy()
    a = np.clip(a * 255.0, 0, 255).round().astype(np.uint8)
    if a.shape[-1] >= 4:
        return a[..., :4]
    return np.dstack([a[..., :3], np.full(a.shape[:2], 255, np.uint8)])


def _to_image(arr: np.ndarray) -> torch.Tensor:
    """uint8 RGBA → [1,H,W,3] float。透明像素会被填成黑色。"""
    a = arr[..., :3].astype(np.float32) / 255.0
    return torch.from_numpy(a)[None]


# ────────────────────────────────────────────────────────────
#  节点 1：分析颜色
# ────────────────────────────────────────────────────────────

class PixelAnalyze:
    """统计图片里出现过的颜色，按面积聚类合并，
    输出一段「颜色 阈值」文本 —— 复制到下方 PixelDownscale 的 layers 框里再调。

    顺序是按面积从大到小排的（大面积通常是背景，默认放在最底层）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tolerance": ("INT", {
                    "default": 8, "min": 0, "max": 128, "step": 1,
                    "tooltip": "颜色合并容差（每通道最大差值）。8 ≈ PS 魔棒容差 8",
                }),
                "default_th": ("FLOAT", {
                    "default": 0.45, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "生成的层列表里每层用的初始阈值",
                }),
                "max_colors": ("INT", {
                    "default": 32, "min": 1, "max": 256, "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "layers_text")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "分析颜色并生成层列表文本，粘贴到 PixelDownscale 里手工调"

    def run(self, image, tolerance, default_th, max_colors):
        rgba = _to_rgba(image)
        layers = auto_layers(rgba, tolerance, default_th, 1, max_colors)
        text = layers_to_text(layers)
        return (image, text)


# ────────────────────────────────────────────────────────────
#  节点 2：分层降采样
# ────────────────────────────────────────────────────────────

class PixelDownscale:
    """把「高分辨率像素风图」变成真正的低分辨率像素画。

    layers 留空 = 自动聚类（先用这个看效果）；
    想精细控制就把 PixelAnalyze 输出的文本粘进来，然后：
      · 调整行的顺序 = 调 z 序（越靠下越靠前）
      · 改每行的阈值 = 调这一层的「要多大面积才留下」
      · 删掉整行 = 去掉这个颜色（噪点直接删，别调阈值）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale": ("INT", {
                    "default": 4, "min": 1, "max": 32, "step": 1,
                    "tooltip": "1 个大像素 = scale×scale 个源像素",
                }),
                "layers": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "每行「颜色 阈值」，顺序=z序（越靠下越靠前）。留空则自动聚类",
                }),
                "tolerance": ("INT", {
                    "default": 8, "min": 0, "max": 128, "step": 1,
                    "tooltip": "颜色匹配容差",
                }),
                "default_th": ("FLOAT", {
                    "default": 0.45, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "layers 留空时，自动生成的层用这个阈值",
                }),
                "fill": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "没达标的块用覆盖率最高的层填色。源图不透明时开；开时 th 建议 0.35~0.45",
                }),
                "min_neighbors": ("INT", {
                    "default": 0, "min": 0, "max": 8, "step": 1,
                    "tooltip": "去孤立块：周围同层邻居少于这个数的块删掉（1 = 去孤立点）",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("preview", "small", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "按颜色分层，逐层阈值降采样。preview 是放大回原尺寸的预览，small 是成品"

    def run(self, image, scale, layers, tolerance, default_th, fill, min_neighbors):
        rgba = _to_rgba(image)

        parsed = parse_layers(layers)
        used_auto = not parsed
        if used_auto:
            parsed = auto_layers(rgba, tolerance, default_th, 1)
            if not parsed:
                raise ValueError("图里没有非透明像素")

        small, stats = quantize(rgba, scale, parsed, 1, tolerance,
                                min_neighbors, fill)
        preview = upscale_nn(small, scale)
        h2, w2 = small.shape[:2]
        total = h2 * w2
        lines = [f"输出 {w2} × {h2}   (scale={scale}"
                 f"{', auto layers' if used_auto else ''}"
                 f"{', fill' if fill else ''})",
                 "",
                 f"{'颜色':<9}{'阈值':>6}{'着色':>7}{'填充':>6}   占比",
                 "-" * 38]
        for s in stats:
            lines.append(f"{s['color']:<9}{s['th']:>6.2f}{s['hit']:>7}"
                         f"{s['fill']:>6}   {s['hit'] / total * 100:5.1f}%")
        cov = int((small[..., 3] > 0).sum())
        lines += ["-" * 38,
                  f"总覆盖率 {cov / total * 100:.1f}%"]
        if used_auto:
            lines += ["", "（layers 为空，用了自动聚类。想细调就接 PixelAnalyze）"]

        return (_to_image(preview), _to_image(small), "\n".join(lines))


# ────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "PixelDownscale": PixelDownscale,
    "PixelAnalyze": PixelAnalyze,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelDownscale": "Pixel Downscale 分层降采样",
    "PixelAnalyze": "Pixel Analyze 颜色分析",
}
