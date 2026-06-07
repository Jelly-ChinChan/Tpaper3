import streamlit as st
from PIL import Image
import numpy as np

# ---------- Otsu 閾值 ----------
def otsu_threshold(gray_arr: np.ndarray) -> int:
    hist = np.bincount(gray_arr.ravel(), minlength=256).astype(np.float64)
    total = gray_arr.size
    if total == 0:
        return 128

    sum_total = np.dot(np.arange(256), hist)
    sumB = 0.0
    wB = 0.0
    maximum = -1.0
    threshold = 128

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
            threshold = t

    return int(threshold)


# ---------- 自動偵測深藍色試紙外圈 ----------
def auto_find_paper_by_blue_edge(
    img_rgb: Image.Image,
    blue_b_min=70,
    blue_rg_diff=15,
    blue_g_diff=5,
    expand_ratio=1.03
):
    arr = np.array(img_rgb).astype(np.int16)
    h, w, _ = arr.shape

    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]

    blue_edge_mask = (
        (B > blue_b_min) &
        (B > R + blue_rg_diff) &
        (B > G + blue_g_diff)
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


# ---------- 分析圓形試紙內藍色 / 褪色面積 ----------
def analyze_blue_faded_area(
    img_rgb: Image.Image,
    circle_mask: np.ndarray,
    mode="RGB 藍色判斷",
    blue_b_min=70,
    blue_rg_diff=15,
    blue_g_diff=5
):
    arr = np.array(img_rgb).astype(np.int16)
    gray = np.array(img_rgb.convert("L")).astype(np.uint8)

    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]

    if mode == "RGB 藍色判斷":
        blue_mask = (
            (B > blue_b_min) &
            (B > R + blue_rg_diff) &
            (B > G + blue_g_diff) &
            circle_mask
        )

        faded_mask = circle_mask & (~blue_mask)
        threshold = None

    else:
        valid_gray = gray[circle_mask]
        threshold = otsu_threshold(valid_gray)

        blue_mask = (gray <= threshold) & circle_mask
        faded_mask = (gray > threshold) & circle_mask

    blue_count = int(blue_mask.sum())
    faded_count = int(faded_mask.sum())
    total = blue_count + faded_count

    blue_ratio = blue_count / total if total else 0
    faded_ratio = faded_count / total if total else 0

    return {
        "gray": gray,
        "threshold": threshold,
        "blue_mask": blue_mask,
        "faded_mask": faded_mask,
        "blue_count": blue_count,
        "faded_count": faded_count,
        "total": total,
        "blue_ratio": blue_ratio,
        "faded_ratio": faded_ratio,
    }


# ---------- 製作預覽疊圖 ----------
def make_overlay(img_rgb, circle_mask, blue_edge_mask, blue_mask, faded_mask):
    arr = np.array(img_rgb).astype(np.float32)
    overlay = arr.copy()

    alpha = 0.38

    # 圓形外：變暗，代表桌面不計算
    outside = ~circle_mask
    overlay[outside] = overlay[outside] * 0.30

    # 褪色區：青色
    overlay[faded_mask] = (
        (1 - alpha) * overlay[faded_mask]
        + alpha * np.array([0, 255, 255])
    )

    # 藍色區：紅色
    overlay[blue_mask] = (
        (1 - alpha) * overlay[blue_mask]
        + alpha * np.array([255, 0, 0])
    )

    # 自動抓到的深藍外圈：黃色強調
    edge_show = blue_edge_mask & circle_mask
    overlay[edge_show] = (
        0.45 * overlay[edge_show]
        + 0.55 * np.array([255, 255, 0])
    )

    return overlay.clip(0, 255).astype(np.uint8)


# ================= Streamlit UI =================
st.title("🧪 圓形試紙藍色 / 褪色面積自動分析")

st.write(
    "上傳照片後，程式會先自動判定深藍色試紙外圈，建立圓形試紙遮罩，"
    "圓形外部視為桌面，不納入面積計算。"
)

uploaded_file = st.file_uploader(
    "請選擇一張圖片",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    st.subheader("1) 原始照片")
    st.image(img, caption="原始照片", use_container_width=True)

    st.subheader("2) 自動偵測參數")

    with st.expander("進階設定：通常不需要調整", expanded=False):
        blue_b_min = st.slider(
            "深藍外圈：B 通道最低值",
            0, 255, 70
        )

        blue_rg_diff = st.slider(
            "深藍外圈：B 必須大於 R 的程度",
            0, 100, 15
        )

        blue_g_diff = st.slider(
            "深藍外圈：B 必須大於 G 的程度",
            0, 100, 5
        )

        expand_ratio = st.slider(
            "圓形遮罩放大比例",
            0.90, 1.15, 1.03, 0.01
        )

    circle_mask, blue_edge_mask, cx, cy, r = auto_find_paper_by_blue_edge(
        img,
        blue_b_min=blue_b_min,
        blue_rg_diff=blue_rg_diff,
        blue_g_diff=blue_g_diff,
        expand_ratio=expand_ratio
    )

    st.info(f"自動偵測結果：圓心 = ({cx}, {cy})，半徑 = {r} px")

    st.subheader("3) 分析方式")

    mode = st.radio(
        "請選擇藍色 / 褪色判斷方式",
        ["RGB 藍色判斷", "灰階 Otsu 自動分割"],
        index=0
    )

    result = analyze_blue_faded_area(
        img,
        circle_mask,
        mode=mode,
        blue_b_min=blue_b_min,
        blue_rg_diff=blue_rg_diff,
        blue_g_diff=blue_g_diff
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
        caption="紅色=藍色面積，青色=褪色面積，黃色=自動偵測到的深藍外圈，變暗區=桌面不計算",
        use_container_width=True
    )

    st.success(
        f"✅ 藍色面積比例：**{result['blue_ratio']:.2%}**　｜　"
        f"褪色面積比例：**{result['faded_ratio']:.2%}**"
    )

    st.subheader("5) 數值摘要")

    if result["threshold"] is not None:
        st.write(f"- Otsu 閾值：{result['threshold']}")

    st.write(
        f"- Blue pixels：{result['blue_count']}\n"
        f"- Faded pixels：{result['faded_count']}\n"
        f"- Total counted pixels：{result['total']}\n"
        f"- Blue ratio：{result['blue_ratio']:.4f}\n"
        f"- Faded ratio：{result['faded_ratio']:.4f}"
    )
