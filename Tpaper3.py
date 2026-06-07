import streamlit as st
from PIL import Image
import numpy as np


# ---------- Otsu 閾值 ----------
def otsu_threshold(arr: np.ndarray) -> float:
    """
    arr: 任意數值陣列，會自動轉成 0~255 做 Otsu
    return: 對應回原始數值尺度的 threshold
    """
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

    threshold_original = v_min + (threshold_scaled / 255) * (v_max - v_min)

    return float(threshold_original)


# ---------- 自動偵測深藍色試紙外圈 ----------
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

    return circle_mask, blue_edge_mask, blue_index, cx, cy, r


# ---------- 分析藍色 / 褪色面積 ----------
def analyze_blue_faded_area(
    img_rgb: Image.Image,
    circle_mask: np.ndarray,
    manual_threshold=None
):
    arr = np.array(img_rgb).astype(np.float32)

    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]

    blue_index = B - ((R + G) / 2)

    valid_blue_index = blue_index[circle_mask]

    if manual_threshold is None:
        threshold = otsu_threshold(valid_blue_index)
    else:
        threshold = manual_threshold

    blue_mask = (blue_index > threshold) & circle_mask
    faded_mask = (blue_index <= threshold) & circle_mask

    blue_count = int(blue_mask.sum())
    faded_count = int(faded_mask.sum())
    total = blue_count + faded_count

    blue_ratio = blue_count / total if total else 0
    faded_ratio = faded_count / total if total else 0

    return {
        "blue_index": blue_index,
        "threshold": threshold,
        "blue_mask": blue_mask,
        "faded_mask": faded_mask,
        "blue_count": blue_count,
        "faded_count": faded_count,
        "total": total,
        "blue_ratio": blue_ratio,
        "faded_ratio": faded_ratio,
    }


# ---------- 製作分析疊圖 ----------
def make_overlay(img_rgb, circle_mask, blue_edge_mask, blue_mask, faded_mask):
    arr = np.array(img_rgb).astype(np.float32)
    overlay = arr.copy()

    alpha = 0.42

    # 圓形外部：變暗，表示桌面不計算
    outside = ~circle_mask
    overlay[outside] = overlay[outside] * 0.30

    # 藍色區：黃色標示
    overlay[blue_mask] = (
        (1 - alpha) * overlay[blue_mask]
        + alpha * np.array([255, 220, 0])
    )

    # 褪色區：紅色標示
    overlay[faded_mask] = (
        (1 - alpha) * overlay[faded_mask]
        + alpha * np.array([255, 0, 0])
    )

    # 自動偵測到的深藍外圈：加深黃色
    edge_show = blue_edge_mask & circle_mask
    overlay[edge_show] = (
        0.35 * overlay[edge_show]
        + 0.65 * np.array([255, 255, 0])
    )

    return overlay.clip(0, 255).astype(np.uint8)


# ================= Streamlit UI =================
st.title("🧪 圓形鉬藍試紙褪色面積分析")

st.write(
    "本程式會自動偵測圓形試紙，排除桌面背景，"
    "並使用 Blue Index = B - (R+G)/2 判斷藍色與褪色區。"
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

    with st.expander("進階設定：通常不需要調整", expanded=False):
        blue_b_min = st.slider(
            "偵測外圈：B 通道最低值",
            0, 255, 70
        )

        blue_index_min = st.slider(
            "偵測外圈：Blue Index 最低值",
            -50, 150, 15
        )

        expand_ratio = st.slider(
            "圓形遮罩放大比例",
            0.90, 1.15, 1.03, 0.01
        )

    circle_mask, blue_edge_mask, blue_index, cx, cy, r = auto_find_paper_by_blue_edge(
        img,
        blue_b_min=blue_b_min,
        blue_index_min=blue_index_min,
        expand_ratio=expand_ratio
    )

    st.info(f"自動偵測結果：圓心 = ({cx}, {cy})，半徑 = {r} px")

    st.subheader("3) 藍色 / 褪色判斷")

    auto_result = analyze_blue_faded_area(
        img,
        circle_mask,
        manual_threshold=None
    )

    with st.expander("進階設定：手動調整 Blue Index 閾值", expanded=False):
        use_manual = st.checkbox("使用手動閾值", value=False)

        manual_threshold = st.slider(
            "Blue Index 閾值",
            -100.0,
            200.0,
            float(auto_result["threshold"]),
            1.0
        )

    if use_manual:
        result = analyze_blue_faded_area(
            img,
            circle_mask,
            manual_threshold=manual_threshold
        )
    else:
        result = auto_result

    st.write(
        f"目前 Blue Index 閾值：**{result['threshold']:.2f}**"
    )

    overlay = make_overlay(
        img,
        circle_mask,
        blue_edge_mask,
        result["blue_mask"],
        result["faded_mask"]
    )

    st.subheader("4) AI 自動選區與分析結果")

    st.image(
        Image.fromarray(overlay),
        caption="黃色=藍色區，紅色=褪色區，變暗區=桌面不計算",
        use_container_width=True
    )

    st.success(
        f"✅ 藍色剩餘面積比例：**{result['blue_ratio']:.2%}**　｜　"
        f"褪色面積比例：**{result['faded_ratio']:.2%}**"
    )

    st.subheader("5) 數值摘要")

    st.write(
        f"- Blue pixels：{result['blue_count']}\n"
        f"- Faded pixels：{result['faded_count']}\n"
        f"- Total counted pixels：{result['total']}\n"
        f"- Blue ratio：{result['blue_ratio']:.4f}\n"
        f"- Faded ratio：{result['faded_ratio']:.4f}\n"
        f"- Blue Index threshold：{result['threshold']:.2f}"
    )
