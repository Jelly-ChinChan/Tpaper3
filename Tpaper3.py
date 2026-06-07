import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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


# ---------- 自動偵測圓形試紙 ----------
def auto_find_circle_mask(img_rgb: Image.Image, bg_brightness_thr=230, shrink_ratio=0.98):
    """
    自動偵測圓形試紙位置，建立圓形 mask
    bg_brightness_thr: 用灰階亮度初步找出比背景暗的區域
    shrink_ratio: 稍微縮小圓形，避免把桌面邊緣算進去
    """
    gray = np.array(img_rgb.convert("L"))
    h, w = gray.shape

    rough_mask = gray < bg_brightness_thr
    ys, xs = np.where(rough_mask)

    if len(xs) < 100:
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2
    else:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        cx = int((x0 + x1) / 2)
        cy = int((y0 + y1) / 2)
        r = int(min(x1 - x0, y1 - y0) / 2 * shrink_ratio)

    yy, xx = np.ogrid[:h, :w]
    circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2

    return circle_mask, cx, cy, r


# ---------- 分析藍色/褪色面積 ----------
def analyze_blue_faded_area(
    img_rgb: Image.Image,
    circle_mask: np.ndarray,
    method="otsu_gray",
    blue_b_threshold=90,
    blue_s_threshold=40
):
    """
    method:
    - otsu_gray: 用灰階 Otsu 分割，較暗者視為藍色，較亮者視為褪色
    - blue_rgb: 用 RGB 判斷藍色，適合藍色很明顯的試紙
    """
    arr = np.array(img_rgb).astype(np.uint8)
    gray = np.array(img_rgb.convert("L")).astype(np.uint8)

    valid_gray = gray[circle_mask]

    if method == "otsu_gray":
        t = otsu_threshold(valid_gray)
        blue_mask = (gray <= t) & circle_mask
        faded_mask = (gray > t) & circle_mask

    else:
        r = arr[:, :, 0].astype(np.int16)
        g = arr[:, :, 1].astype(np.int16)
        b = arr[:, :, 2].astype(np.int16)

        blue_strength = b - ((r + g) / 2)

        blue_mask = (
            (b > blue_b_threshold) &
            (blue_strength > blue_s_threshold) &
            circle_mask
        )

        faded_mask = circle_mask & (~blue_mask)
        t = None

    blue_count = int(blue_mask.sum())
    faded_count = int(faded_mask.sum())
    total = blue_count + faded_count

    blue_ratio = blue_count / total if total else 0
    faded_ratio = faded_count / total if total else 0

    return {
        "gray": gray,
        "threshold": t,
        "blue_mask": blue_mask,
        "faded_mask": faded_mask,
        "blue_count": blue_count,
        "faded_count": faded_count,
        "total": total,
        "blue_ratio": blue_ratio,
        "faded_ratio": faded_ratio,
    }


# ---------- 疊圖 ----------
def make_overlay(img_rgb, circle_mask, blue_mask, faded_mask):
    arr = np.array(img_rgb).astype(np.float32)
    overlay = arr.copy()

    alpha = 0.35

    # 藍色面積：紅色標示
    overlay[blue_mask] = (1 - alpha) * overlay[blue_mask] + alpha * np.array([255, 0, 0])

    # 褪色面積：青色標示
    overlay[faded_mask] = (1 - alpha) * overlay[faded_mask] + alpha * np.array([0, 255, 255])

    # 圓形外部變暗，方便確認桌子沒有被計算
    outside = ~circle_mask
    overlay[outside] = overlay[outside] * 0.35

    return overlay.clip(0, 255).astype(np.uint8)


# ================= Streamlit UI =================
st.title("🧪 圓形試紙藍色/褪色面積分析")

st.write(
    "上傳照片後，程式會自動偵測圓形試紙，只計算圓形試紙內部的藍色面積與褪色面積，"
    "不會把桌子背景納入計算。"
)

uploaded_file = st.file_uploader(
    "請選擇一張圖片...",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    w, h = img.size

    st.subheader("1) 原始圖片")
    st.image(img, caption="上傳的圖片", use_container_width=True)

    st.subheader("2) 圓形試紙偵測參數")

    col1, col2 = st.columns(2)

    with col1:
        bg_brightness_thr = st.slider(
            "背景亮度門檻",
            min_value=120,
            max_value=255,
            value=230,
            help="數值越高，越容易把較亮區域也納入試紙偵測。"
        )

    with col2:
        shrink_ratio = st.slider(
            "圓形縮小比例",
            min_value=0.80,
            max_value=1.05,
            value=0.98,
            step=0.01,
            help="建議 0.95~1.00，可避免把桌面邊緣算進去。"
        )

    circle_mask, cx, cy, r = auto_find_circle_mask(
        img,
        bg_brightness_thr=bg_brightness_thr,
        shrink_ratio=shrink_ratio
    )

    st.write(f"偵測到圓心：({cx}, {cy})，半徑：{r} px")

    st.subheader("3) 手動微調圓形 ROI")

    col3, col4, col5 = st.columns(3)

    with col3:
        cx = st.slider("圓心 X", 0, w - 1, int(cx))

    with col4:
        cy = st.slider("圓心 Y", 0, h - 1, int(cy))

    with col5:
        r = st.slider("半徑 r", 10, min(w, h) // 2, int(r))

    yy, xx = np.ogrid[:h, :w]
    circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2

    st.subheader("4) 藍色/褪色分割方法")

    method = st.radio(
        "選擇分割方法",
        ["otsu_gray", "blue_rgb"],
        format_func=lambda x: "灰階 Otsu 自動分割" if x == "otsu_gray" else "RGB 藍色判斷"
    )

    blue_b_threshold = 90
    blue_s_threshold = 40

    if method == "blue_rgb":
        col6, col7 = st.columns(2)

        with col6:
            blue_b_threshold = st.slider(
                "藍色 B 通道最低值",
                0,
                255,
                90
            )

        with col7:
            blue_s_threshold = st.slider(
                "藍色強度門檻 B - (R+G)/2",
                0,
                150,
                40
            )

    result = analyze_blue_faded_area(
        img,
        circle_mask,
        method=method,
        blue_b_threshold=blue_b_threshold,
        blue_s_threshold=blue_s_threshold
    )

    st.success(
        f"✅ 藍色面積比例：**{result['blue_ratio']:.2%}** ｜ "
        f"褪色面積比例：**{result['faded_ratio']:.2%}**"
    )

    st.subheader("5) 視覺化結果")

    overlay = make_overlay(
        img,
        circle_mask,
        result["blue_mask"],
        result["faded_mask"]
    )

    st.image(
        Image.fromarray(overlay),
        caption="分割疊圖：藍色面積=紅色標示，褪色面積=青色標示；圓形外部已變暗，不納入計算",
        use_container_width=True
    )

    st.subheader("6) 灰階直方圖")

    gray = result["gray"]
    valid_pixels = gray[circle_mask]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(valid_pixels.ravel(), bins=40, edgecolor="black")

    if result["threshold"] is not None:
        ax.axvline(result["threshold"], linestyle="--")
        ax.set_title(f"Grayscale Histogram within Circular Paper ROI, Otsu threshold = {result['threshold']}")
    else:
        ax.set_title("Grayscale Histogram within Circular Paper ROI")

    ax.set_xlabel("Grayscale 0=black, 255=white")
    ax.set_ylabel("Count")

    st.pyplot(fig)

    st.subheader("7) 數值摘要")

    st.write(
        f"- Blue pixels: {result['blue_count']}\n"
        f"- Faded pixels: {result['faded_count']}\n"
        f"- Total counted pixels: {result['total']}\n"
        f"- Blue ratio: {result['blue_ratio']:.4f}\n"
        f"- Faded ratio: {result['faded_ratio']:.4f}"
    )