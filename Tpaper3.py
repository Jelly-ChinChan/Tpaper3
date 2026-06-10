import streamlit as st
from PIL import Image
import numpy as np
from collections import deque


def otsu_threshold(arr: np.ndarray) -> float:
    values = arr.ravel().astype(np.float64)
    if values.size == 0:
        return 0.0

    v_min = values.min()
    v_max = values.max()

    if v_max == v_min:
        return float(v_min)

    scaled = ((values - v_min) / (v_max - v_min) * 255).astype(np.uint8)

    hist = np.bincount(scaled, minlength=256).astype(np.float64)
    total = scaled.size
    sum_total = np.dot(np.arange(256), hist)

    sumB = 0.0
    wB = 0.0
    maximum = -1.0
    threshold_scaled = 128

    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue

        wF = total - wB
        if wF == 0:
            break

        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF

        between = wB * wF * (mB - mF) ** 2

        if between > maximum:
            maximum = between
            threshold_scaled = t

    return float(v_min + threshold_scaled / 255 * (v_max - v_min))


def auto_find_paper_by_blue_edge(
    img_rgb: Image.Image,
    blue_b_min=70,
    blue_index_min=15,
    expand_ratio=1.03
):
    arr = np.array(img_rgb).astype(np.float32)
    h, w, _ = arr.shape

    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]

    blue_index = B - ((R + G) / 2)

    blue_edge_mask = (
        (B > blue_b_min) &
        (blue_index > blue_index_min)
    )

    ys, xs = np.where(blue_edge_mask)

    if len(xs) < 100:
        cx = w // 2
        cy = h // 2
        r = min(w, h) // 2
    else:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        cx = int((x0 + x1) / 2)
        cy = int((y0 + y1) / 2)

        diameter = max(x1 - x0, y1 - y0)
        r = int(diameter / 2 * expand_ratio)

    r = min(r, cx, cy, w - cx - 1, h - cy - 1)

    yy, xx = np.ogrid[:h, :w]
    circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2

    return circle_mask, blue_edge_mask, cx, cy, r


def binary_dilate(mask, iterations=3):
    result = mask.copy()

    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)

        result = (
            padded[1:-1, 1:-1] |
            padded[:-2, 1:-1] |
            padded[2:, 1:-1] |
            padded[1:-1, :-2] |
            padded[1:-1, 2:] |
            padded[:-2, :-2] |
            padded[:-2, 2:] |
            padded[2:, :-2] |
            padded[2:, 2:]
        )

    return result


def binary_erode(mask, iterations=3):
    result = mask.copy()

    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)

        result = (
            padded[1:-1, 1:-1] &
            padded[:-2, 1:-1] &
            padded[2:, 1:-1] &
            padded[1:-1, :-2] &
            padded[1:-1, 2:] &
            padded[:-2, :-2] &
            padded[:-2, 2:] &
            padded[2:, :-2] &
            padded[2:, 2:]
        )

    return result


def close_mask(mask, iterations=6):
    return binary_erode(binary_dilate(mask, iterations), iterations)


def fill_holes_inside_circle(mask, circle_mask):
    """
    將封閉紅色輪廓內部全部填滿。
    """
    h, w = mask.shape

    background = (~mask) & circle_mask
    visited = np.zeros_like(mask, dtype=bool)

    q = deque()

    # 從 circle_mask 邊界附近出發，找出外部背景
    boundary = circle_mask & (~binary_erode(circle_mask, 1))
    ys, xs = np.where(boundary & background)

    for y, x in zip(ys, xs):
        visited[y, x] = True
        q.append((y, x))

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    while q:
        y, x = q.popleft()

        for dy, dx in directions:
            ny = y + dy
            nx = x + dx

            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue

            if visited[ny, nx]:
                continue

            if background[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    holes = background & (~visited)

    filled = mask | holes
    filled = filled & circle_mask

    return filled


def analyze_two_zones(
    img_rgb: Image.Image,
    circle_mask: np.ndarray,
    white_threshold=None,
    white_bias=0.0,
    close_iterations=8
):
    arr = np.array(img_rgb).astype(np.float32)

    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]

    gray = 0.299 * R + 0.587 * G + 0.114 * B
    blue_index = B - ((R + G) / 2)

    white_score = gray - blue_index + white_bias

    valid_white_score = white_score[circle_mask]

    if white_threshold is None:
        white_threshold = otsu_threshold(valid_white_score)

    # 初步找出白色褪色邊界
    candidate_faded = (white_score > white_threshold) & circle_mask

    # 把紅色邊界補縫，避免中間無法填滿
    closed_faded = close_mask(candidate_faded, iterations=close_iterations)
    closed_faded = closed_faded & circle_mask

    # 把紅色輪廓內部全部填滿
    faded_mask = fill_holes_inside_circle(closed_faded, circle_mask)

    blue_mask = circle_mask & (~faded_mask)

    faded_count = int(faded_mask.sum())
    blue_count = int(blue_mask.sum())
    total = faded_count + blue_count

    faded_ratio = faded_count / total if total else 0
    blue_ratio = blue_count / total if total else 0

    return {
        "white_score": white_score,
        "white_threshold": white_threshold,
        "candidate_faded": candidate_faded,
        "closed_faded": closed_faded,
        "faded_mask": faded_mask,
        "blue_mask": blue_mask,
        "faded_count": faded_count,
        "blue_count": blue_count,
        "total": total,
        "faded_ratio": faded_ratio,
        "blue_ratio": blue_ratio,
    }


def make_overlay(
    img_rgb,
    circle_mask,
    blue_mask,
    faded_mask
):
    arr = np.array(img_rgb).astype(np.float32)
    overlay = arr.copy()

    outside = ~circle_mask
    overlay[outside] = overlay[outside] * 0.30

    overlay[blue_mask] = np.array([255, 230, 0])
    overlay[faded_mask] = np.array([255, 0, 0])

    return overlay.clip(0, 255).astype(np.uint8)


# ================= Streamlit UI =================
st.title("🧪 圓形鉬藍試紙二區面積分析")

st.write(
    "黃色=剩餘藍色區，紅色=白色褪色區。"
    "本版會自動把紅色輪廓內部全部填滿。"
)

uploaded_file = st.file_uploader(
    "請選擇一張圖片",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    st.subheader("1) 原始照片")
    st.image(img, caption="原始照片", use_container_width=True)

    st.subheader("2) 自動偵測圓形試紙")

    with st.expander("進階設定：圓形偵測", expanded=False):
        blue_b_min = st.slider("偵測外圈：B 通道最低值", 0, 255, 70)
        blue_index_min = st.slider("偵測外圈：Blue Index 最低值", -50, 150, 15)
        expand_ratio = st.slider("圓形遮罩放大比例", 0.90, 1.15, 1.03, 0.01)

    circle_mask, blue_edge_mask, cx, cy, r = auto_find_paper_by_blue_edge(
        img,
        blue_b_min=blue_b_min,
        blue_index_min=blue_index_min,
        expand_ratio=expand_ratio
    )

    st.info(f"自動偵測結果：圓心 = ({cx}, {cy})，半徑 = {r} px")

    st.subheader("3) 二區判斷參數")

    auto_result = analyze_two_zones(
        img,
        circle_mask,
        white_threshold=None
    )

    with st.expander("進階設定：褪色判斷", expanded=False):
        use_manual = st.checkbox("使用手動閾值", value=False)

        manual_white_threshold = st.slider(
            "White Score 閾值",
            0.0,
            255.0,
            float(auto_result["white_threshold"]),
            1.0
        )

        white_bias = st.slider(
            "白色判斷偏移值",
            -100.0,
            100.0,
            0.0,
            1.0
        )

        close_iterations = st.slider(
            "紅色邊界補縫強度",
            1,
            20,
            8,
            1,
            help="數值越大，越容易把紅色邊界接起來並填滿內部。"
        )

    if use_manual:
        result = analyze_two_zones(
            img,
            circle_mask,
            white_threshold=manual_white_threshold,
            white_bias=white_bias,
            close_iterations=close_iterations
        )
    else:
        result = analyze_two_zones(
            img,
            circle_mask,
            white_threshold=None,
            white_bias=white_bias,
            close_iterations=close_iterations
        )

    st.write(f"White Score 閾值：**{result['white_threshold']:.2f}**")

    overlay = make_overlay(
        img,
        circle_mask,
        result["blue_mask"],
        result["faded_mask"]
    )

    st.subheader("4) AI 自動選區與分析結果")

    st.image(
        Image.fromarray(overlay),
        caption="黃色=剩餘藍色區，紅色=白色褪色區，變暗=桌面不計算",
        use_container_width=True
    )

    st.success(
        f"✅ 剩餘藍色區：**{result['blue_ratio']:.2%}**　｜　"
        f"白色褪色區：**{result['faded_ratio']:.2%}**"
    )

    st.subheader("5) 數值摘要")

    st.write(
        f"- Blue pixels：{result['blue_count']}\n"
        f"- Faded white pixels：{result['faded_count']}\n"
        f"- Total counted pixels：{result['total']}\n"
        f"- Blue ratio：{result['blue_ratio']:.4f}\n"
        f"- Faded white ratio：{result['faded_ratio']:.4f}\n"
        f"- White Score threshold：{result['white_threshold']:.2f}"
    )
