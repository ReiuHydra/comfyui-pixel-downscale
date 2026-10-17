"""核心算法 —— 纯 numpy，无 IO、无框架依赖。

单独拿出去也能用：

    from core import quantize, auto_layers, parse_layers
    small, stats = quantize(rgba, scale=4, layers=auto_layers(rgba))

原理：
    按颜色分层 → 每层独立阈值 → 按 scale×scale 分块算覆盖率 →
    cov ≥ th 才着色 → 按数组顺序（= z 序，越靠后越前）叠加。
"""

from __future__ import annotations

import numpy as np


# ────────────────────────────────────────────────────────────
#  颜色
# ────────────────────────────────────────────────────────────

def hex_to_rgb(s: str) -> tuple[int, int, int]:
    s = str(s).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"不是 6 位十六进制颜色：{s!r}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def rgb_to_hex(c) -> str:
    return "#%02X%02X%02X" % (int(c[0]), int(c[1]), int(c[2]))


def parse_layers(text: str) -> list[dict]:
    """解析层列表文本。

    每行一个层：`颜色 阈值`，顺序 = z 序（越靠后越靠前）。
    `//` 后面是注释；颜色带不带 `#` 都行。

        89CEFF 0.45
        #4D91F4 0.45      // 天空
        EC475B 0.30
    """
    out: list[dict] = []
    for raw in (text or "").splitlines():
        line = raw.split("//")[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        try:
            rgb = hex_to_rgb(parts[0])
        except ValueError:
            continue
        th = 0.45
        if len(parts) > 1:
            try:
                th = float(parts[1])
            except ValueError:
                pass
        out.append({"rgb": rgb, "th": float(min(1.0, max(0.01, th)))})
    return out


def layers_to_text(layers: list[dict], note: str = "") -> str:
    head = note or "// 颜色  阈值    —— 越靠下越靠前（z 序）"
    body = [f'{rgb_to_hex(L["rgb"])} {L["th"]:.2f}' for L in layers]
    return "\n".join([head] + body)


# ────────────────────────────────────────────────────────────
#  分块统计
# ────────────────────────────────────────────────────────────

def block_mean(mask: np.ndarray, scale: int) -> np.ndarray:
    """(H, W) → (H//scale, W//scale)，每块的均值（0~1）。

    也就是「这一块里有多少比例的像素满足 mask」。
    """
    h2, w2 = mask.shape[0] // scale, mask.shape[1] // scale
    if h2 == 0 or w2 == 0:
        raise ValueError(f"图太小：{mask.shape} 除以 scale={scale} 后没有像素")
    m = mask[: h2 * scale, : w2 * scale].astype(np.float32)
    return m.reshape(h2, scale, w2, scale).mean(axis=(1, 3))


def count_neighbors(mask: np.ndarray) -> np.ndarray:
    """每个格子周围 8 邻域里有多少个 True。"""
    h, w = mask.shape
    m = np.pad(mask.astype(np.uint8), 1, constant_values=0)
    s = np.zeros((h, w), np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            s += m[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return s


def upscale_nn(img: np.ndarray, k: int) -> np.ndarray:
    """最近邻放大 k 倍（预览用）。"""
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)


# ────────────────────────────────────────────────────────────
#  颜色聚类
# ────────────────────────────────────────────────────────────

def collect_colors(rgba: np.ndarray, alpha_min: int = 1):
    """统计出现过的颜色，按像素数降序。"""
    a = rgba[..., 3]
    ys, xs = np.nonzero(a >= alpha_min)
    if len(ys) == 0:
        return np.zeros((0, 3), np.uint8), np.zeros((0,), np.int64)
    px = rgba[ys, xs, :3].reshape(-1, 3)
    colors, counts = np.unique(px, axis=0, return_counts=True)
    order = np.argsort(-counts)
    return colors[order], counts[order]


def cluster_colors(colors: np.ndarray, counts: np.ndarray, tol: int,
                   max_colors: int = 512):
    """把相近颜色合并成簇（按像素数降序贪心）。

    代表色始终是「簇里最先遇到的那个颜色」，通常就是最纯的那个。

    max_colors：只拿像素数前 N 个颜色参与聚类。
    这个是必须的 —— 照片类素材可能有几十万种颜色，
    而这里是 O(颜色数 × 簇数) 的循环，不限制会直接卡死。
    被丢掉的都是占比极小的碎色，本来就该当噪点。

    返回 (代表色列表, 每簇像素数)。
    """
    if max_colors and len(colors) > max_colors:
        colors, counts = colors[:max_colors], counts[:max_colors]

    reps: list[tuple[int, int, int]] = []
    sums: list[int] = []
    if tol <= 0:
        return ([tuple(int(v) for v in c) for c in colors],
                [int(n) for n in counts])

    for c, n in zip(colors, counts):
        if reps:
            d = np.abs(np.asarray(reps, np.int16) - c.astype(np.int16)).max(axis=1)
            j = int(np.argmin(d))
            if d[j] <= tol:
                sums[j] += int(n)
                continue
        reps.append(tuple(int(v) for v in c))
        sums.append(int(n))
    return reps, sums


def assign_pixels(rgba: np.ndarray, palette, tol: int,
                  alpha_min: int = 1) -> np.ndarray:
    """每个像素归给「最接近的代表色」，超出容差或透明 = -1。

    一次性算完，一个像素只属于一层 —— 不会出现
    「两种颜色都说这块是我的」而要靠 z 序去打的情况。
    """
    px = rgba[..., :3].astype(np.int16)
    best = np.full(px.shape[:2], 10 ** 6, np.int32)
    idx = np.full(px.shape[:2], -1, np.int32)

    for i, c in enumerate(palette):
        d = np.abs(px - np.asarray(c, np.int16)).max(axis=-1)
        m = d < best
        best[m] = d[m]
        idx[m] = i

    idx[(best > tol) | (rgba[..., 3] < alpha_min)] = -1
    return idx


def auto_layers(rgba: np.ndarray, tolerance: int = 8,
                default_th: float = 0.45, alpha_min: int = 1,
                max_colors: int = 64) -> list[dict]:
    """自动生成层列表：颜色聚类，按面积降序（= 默认 z 序，大面积在后）。"""
    colors, counts = collect_colors(rgba, alpha_min)
    if len(colors) == 0:
        return []

    reps0, _ = cluster_colors(colors, counts, tolerance)

    # 用最终代表色重新统计真实归属：
    # 贪婪聚类和「最近色分配」在容差重叠时结果不一致，以 assign_pixels 为准。
    idx = assign_pixels(rgba, reps0, tolerance, alpha_min)
    sums = [int((idx == i).sum()) for i in range(len(reps0))]
    order = sorted(range(len(reps0)), key=lambda i: -sums[i])

    return [{"rgb": reps0[i], "th": default_th} for i in order[:max_colors]]


# ────────────────────────────────────────────────────────────
#  主算法
# ────────────────────────────────────────────────────────────

def quantize(rgba: np.ndarray, scale: int, layers: list[dict],
             alpha_min: int = 1, tol: int = 0, min_neighbors: int = 0,
             fill: bool = False):
    """分层量化。

    rgba          : (H, W, 4) uint8
    scale         : 一个大像素 = scale × scale 个源像素
    layers        : [{"rgb": (r,g,b), "th": 0.5}]  顺序 = z 序（越靠后越前）
    tol           : 颜色容差（每通道最大差值），0 = 精确匹配
    min_neighbors : 一块要保留，周围至少要这么多个同层邻居（0 = 不处理）
    fill          : 没达标的块用【覆盖率最高的那层】填色

    返回 (小图, 每层统计)。
    """
    if not layers:
        raise ValueError("layers 是空的")

    h2, w2 = rgba.shape[0] // scale, rgba.shape[1] // scale
    out = np.zeros((h2, w2, 4), dtype=np.uint8)
    stats: list[dict] = []

    idx = assign_pixels(rgba, [L["rgb"] for L in layers], tol, alpha_min)

    # 诊断：大部分像素找不到归属，说明 layers 和这张图不配套
    opaque = int((rgba[..., 3] >= alpha_min).sum())
    if opaque:
        miss = float((idx == -1).sum()) / opaque
        if miss > 0.8:
            raise ValueError(
                f"{miss * 100:.0f}% 的非透明像素找不到匹配颜色。\n"
                f"常见原因：layers 是从另一张图聚出来的，或者 tolerance 太小。\n"
                f"建议：用 Pixel Analyze 对【当前这张图】重新分析，"
                f"或者把 tolerance 调大（同图内 8 够用，跨图得几十以上）。"
            )

    covs = []
    for i, L in enumerate(layers):
        cov = block_mean(idx == i, scale)
        covs.append(cov)
        hit = cov >= float(L["th"])
        raw_hit = int(hit.sum())

        if min_neighbors > 0:
            hit &= count_neighbors(hit) >= min_neighbors

        out[hit, :3] = np.asarray(L["rgb"], dtype=np.uint8)
        out[hit, 3] = 255
        stats.append({
            "color": rgb_to_hex(L["rgb"]),
            "th": L["th"],
            "hit": int(hit.sum()),
            "raw": raw_hit,
            "fill": 0,
            "total": h2 * w2,
        })

    # 填充：对还空着的块，挑覆盖率最高的那层盖上
    if fill and covs:
        empty = out[..., 3] == 0
        if empty.any():
            C = np.stack(covs, axis=0)
            best = C.argmax(axis=0)
            best_cov = C.max(axis=0)
            m = empty & (best_cov > 0)        # 源图本来就空的地方不填
            for i, L in enumerate(layers):
                sub = m & (best == i)
                n = int(sub.sum())
                if n:
                    out[sub, :3] = np.asarray(L["rgb"], dtype=np.uint8)
                    out[sub, 3] = 255
                    stats[i]["fill"] = n

    return out, stats
