"""ComfyUI 自定义节点：像素降采样（按颜色分层 + 逐层阈值）。

安装：把整个 `comfyui-pixel-downscale` 文件夹放进
      ComfyUI/custom_nodes/ ，重启 ComfyUI。
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
